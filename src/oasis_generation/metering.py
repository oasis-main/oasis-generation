"""Usage events for capability-token callers (ORG-171, interface doc 5.3).

No unmetered inference: every upstream call made for a capability token emits
one usage event to `oasis/system/gen/usage/output` (QoS 1). oasis-auth prices it
and adds it to the user's month-to-date charges.

  - UsageMeter  publishes an event; when the publish fails it appends the event
                to a JSON-lines spool on disk and resends it later (flush()).
                available() is False when the spool holds more than spool_max
                events, when its oldest event is 10 minutes old or older, or when
                the spool cannot be written. The gateway then answers new metered
                requests with 503 metering_unavailable.
  - MqttBridge  the paho-mqtt connection: service login, the usage publish, and
                the revocation / billing subscriptions that feed
                capability.AuthState. paho is imported only when the bridge
                starts, so tests and service-only deployments never need a broker.

Service-token callers (the fleet) are not metered by this path; they keep the
local JSONL ledger in main.py.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)

USAGE_TOPIC = "oasis/system/gen/usage/output"
REVOCATION_TOPIC = "oasis/system/auth/revocation/output"
BILLING_TOPIC = "oasis/system/auth/billing/output"
SPOOL_FILE = "usage-spool.jsonl"
SPOOL_MAX_AGE_S = 600
DEFAULT_SPOOL_MAX = 1000


def uuid7() -> str:
    """RFC 9562 UUIDv7: 48-bit Unix ms time, version 7, variant 10, random rest."""
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= ((rand >> 62) & 0xFFF) << 64      # rand_a, 12 bits
    value |= 0b10 << 62                        # variant
    value |= rand & ((1 << 62) - 1)            # rand_b, 62 bits
    return str(uuid.UUID(int=value))


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def build_usage_event(
    *, sub: str, jti: str, allowance_ref: str, request_id: str, client_request_id: str,
    model: str, backend: str, served_by: str, tokens_in: int, tokens_out: int,
    cache_read_tokens: int, cache_write_tokens: int, status: str, streamed: bool,
    **extra: Any,
) -> dict:
    """One usage event, exactly the fields of interface doc 5.3 (plus `extra`)."""
    ev = {
        "event_id": uuid7(), "ts": _now_iso(), "service": "gen", "capability": "inference",
        "sub": sub, "jti": jti, "allowance_ref": allowance_ref, "request_id": request_id,
        "client_request_id": client_request_id, "model": model, "backend": backend,
        "served_by": served_by, "tokens_in": int(tokens_in), "tokens_out": int(tokens_out),
        "cache_read_tokens": int(cache_read_tokens), "cache_write_tokens": int(cache_write_tokens),
        "status": status, "streamed": bool(streamed),
    }
    ev.update(extra)
    return ev


class Publisher(Protocol):
    def publish(self, topic: str, payload: bytes, qos: int) -> bool: ...


class UsageMeter:
    """Publish usage events; spool the ones that fail; tell the gateway when to stop."""

    def __init__(self, publisher: Publisher | None, spool_dir: str,
                 spool_max: int = DEFAULT_SPOOL_MAX, max_age_s: float = SPOOL_MAX_AGE_S,
                 clock: Callable[[], float] = time.time):
        self.publisher = publisher
        self.spool_dir = spool_dir
        self.spool_max = spool_max
        self.max_age_s = max_age_s
        self._clock = clock
        self._lock = threading.Lock()
        self._broken = False
        self._spooled_at: list[float] = []   # one entry per spooled line, in order
        self._load()

    @property
    def _path(self) -> str:
        return os.path.join(self.spool_dir, SPOOL_FILE)

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._spooled_at.append(float(json.loads(line)["spooled_at"]))
                    except (ValueError, KeyError, TypeError):
                        # A torn last line (crash mid-write) is kept on disk and
                        # counted as old, so it cannot hide behind the age rule.
                        self._spooled_at.append(0.0)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.error("usage spool unreadable (%s): metering unavailable", exc)
            self._broken = True

    def _publish(self, event: dict) -> bool:
        if self.publisher is None:
            return False
        try:
            return bool(self.publisher.publish(
                USAGE_TOPIC, json.dumps(event, separators=(",", ":")).encode(), 1))
        except Exception as exc:  # noqa: BLE001 - a broken publisher means "spool it"
            logger.warning("usage publish raised %s", type(exc).__name__)
            return False

    def emit(self, event: dict) -> None:
        """Publish now, or spool. Never raises."""
        if self._publish(event):
            return
        line = json.dumps({"spooled_at": self._clock(), "event": event}, separators=(",", ":"))
        with self._lock:
            try:
                os.makedirs(self.spool_dir, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                self._spooled_at.append(self._clock())
            except OSError as exc:
                # The event is lost. Refuse further metered work until the spool
                # is writable again (the operator must look at the disk).
                self._broken = True
                logger.error("usage spool write failed (%s); event %s lost; metering unavailable",
                             exc, event.get("event_id"))

    def flush(self) -> int:
        """Resend spooled events in order; stop at the first failure. Returns the number sent."""
        with self._lock:
            if not self._spooled_at:
                return 0
            try:
                with open(self._path, encoding="utf-8") as fh:
                    lines = [x for x in fh.read().splitlines() if x.strip()]
            except OSError:
                return 0
            sent = 0
            for line in lines:
                try:
                    event = json.loads(line)["event"]
                except (ValueError, KeyError, TypeError):
                    logger.error("usage spool: dropping an unreadable line")
                    sent += 1
                    continue
                if not self._publish(event):
                    break
                sent += 1
            if sent == 0:
                return 0
            rest = lines[sent:]
            tmp = self._path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write("".join(x + "\n" for x in rest))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self._path)
            except OSError as exc:
                # The sent events stay on disk and are sent again later. The
                # consumer is idempotent on event_id, so a resend costs nothing.
                logger.error("usage spool rewrite failed (%s)", exc)
                return 0
            self._spooled_at = self._spooled_at[sent:]
            return sent

    def spool_count(self) -> int:
        with self._lock:
            return len(self._spooled_at)

    def available(self) -> bool:
        with self._lock:
            if self._broken:
                return False
            if len(self._spooled_at) > self.spool_max:
                return False
            if self._spooled_at and self._clock() - self._spooled_at[0] >= self.max_age_s:
                return False
            return True


def unique_client_id(prefix: str) -> str:
    """Two gateway tasks with one client id would disconnect each other."""
    return f"{prefix}-{secrets.token_hex(4)}"


def _rc_ok(rc: Any) -> bool:
    failure = getattr(rc, "is_failure", None)
    if failure is not None:
        return not failure
    return rc == 0


def _default_client_factory(client_id: str):
    import paho.mqtt.client as mqtt  # imported only when the bridge starts

    return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                       client_id=client_id, protocol=mqtt.MQTTv311)


class MqttBridge:
    """One paho connection: publish usage, subscribe to revocation and billing.

    Connects in the background (connect_async + loop_start), so the gateway
    starts and serves even when the broker is down; usage events then go to the
    spool and the revocation set stays as it was (CONTRACT section 6).
    """

    def __init__(self, *, host: str, port: int, username: str, password: str, client_id: str,
                 state, client_factory: Callable[[str], Any] = _default_client_factory):
        self.host, self.port = host, port
        self.username, self._password = username, password
        self.client_id = client_id
        self.state = state
        self._factory = client_factory
        self._client = None
        self.connected = False

    def start(self) -> None:
        c = self._factory(self.client_id)
        c.username_pw_set(self.username, self._password)
        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_message = self._on_message
        if hasattr(c, "reconnect_delay_set"):
            c.reconnect_delay_set(min_delay=1, max_delay=60)
        self._client = c
        c.connect_async(self.host, self.port, keepalive=60)
        c.loop_start()
        logger.info("MQ bridge connecting to %s:%s as %s (client id %s)",
                    self.host, self.port, self.username, self.client_id)

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
                self._client.loop_stop()
            except Exception:  # noqa: BLE001
                pass
        self.connected = False

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if not _rc_ok(reason_code):
            logger.error("MQ bridge login refused: %s", reason_code)
            self.connected = False
            return
        self.connected = True
        # Plain (not shared) subscriptions: every gateway task needs every event.
        client.subscribe(REVOCATION_TOPIC, qos=1)
        client.subscribe(BILLING_TOPIC, qos=1)
        logger.info("MQ bridge connected; subscribed to revocation and billing topics")

    def _on_disconnect(self, client, userdata, flags, reason_code=None, properties=None):
        self.connected = False
        logger.warning("MQ bridge disconnected: %s", reason_code)

    def _on_message(self, client, userdata, msg):
        try:
            ev = json.loads(msg.payload)
        except (ValueError, TypeError):
            logger.warning("MQ bridge: non-JSON message on %s", msg.topic)
            return
        if msg.topic == REVOCATION_TOPIC:
            ok = self.state.apply_revocation(ev)
        elif msg.topic == BILLING_TOPIC:
            ok = self.state.apply_billing(ev)
        else:
            return
        if not ok:
            logger.warning("MQ bridge: refused a bad event on %s", msg.topic)

    def publish(self, topic: str, payload: bytes, qos: int) -> bool:
        if not self.connected or self._client is None:
            return False
        info = self._client.publish(topic, payload, qos=qos)
        return getattr(info, "rc", 1) == 0
