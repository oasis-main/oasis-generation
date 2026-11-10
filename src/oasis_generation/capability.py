"""Capability-token verifier for the gateway (ORG-171, contract v1).

This module implements docs/capability-token/CONTRACT.md for production use. The
test file tests/test_capability.py runs every vector in
docs/capability-token/vectors.json (token and billing) against it, so a change
here that moves any decision or reason code fails the tests.

Pieces:
  - verify()                  the contract check order, reason codes and claims.
  - check_inference_allowed() the decision table for a metered route (ORG-172 rule 2).
  - JwksCache                 the oasis-auth JWKS, TTL 600 s, one rate-limited
                              refetch on an unknown kid, last good set kept on error.
  - AuthState                 revoked jti set, per-sub not-before, and the
                              blocked-sub set from the billing topic, plus the
                              periodic full snapshot that repairs all three.

The package never imports from the repository's reference script; the logic is
copied on purpose so the image does not ship test tooling.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import json
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

AUD_GEN = "oasis-gen"
TYP = "oasis-cap+jwt"
MAX_LIFETIME_S = 900
LEEWAY_S = 30
TIERS = ("free", "personal", "pro", "max")
BILLING = ("none", "active", "suspended")
KNOWN_SCOPES = ("gen:chat", "gen:models", "gen:decide", "ws:session", "ws:fs", "ws:shell")
REQUIRED = ("iss", "aud", "sub", "tier", "billing_state", "scopes", "jti", "iat", "exp")

JWKS_TTL_S = 600
# Minimum time between two refetches caused by an unknown kid. A token with a
# random kid must not turn the gateway into a JWKS request amplifier.
JWKS_UNKNOWN_KID_MIN_INTERVAL_S = 30
JWKS_MAX_BYTES = 64 * 1024
# A revocation entry is held until every token it can match has expired: a token
# issued at most LEEWAY_S after the event lives at most MAX_LIFETIME_S, and is
# accepted LEEWAY_S after exp (CONTRACT section 6).
REVOCATION_HOLD_S = MAX_LIFETIME_S + 2 * LEEWAY_S

# ---- state snapshot (ORG-171 backstop A) ----
# A snapshot whose ts is this far ahead of our clock is refused: its ts would
# pin the per-sub ordering in the future and make every later event "older".
SNAPSHOT_FUTURE_S = 60
# A snapshot older than this is ignored (a replay or a stuck queue), and does not
# count as fresh state.
SNAPSHOT_MAX_AGE_S = 300
# A locally blocked sub is dropped as "healed" only when its deciding event is
# older than the snapshot by more than this. Auth publishes the billing event
# first and stores the signal after, so a snapshot read in between can miss a
# sub we already (rightly) blocked.
SNAPSHOT_HEAL_GRACE_S = 10
SNAPSHOT_MAX_LIST = 20_000
SNAPSHOT_MAX_PARTS = 100
SNAPSHOT_PENDING_MAX = 8
SNAPSHOT_PENDING_TTL_S = 120


# ---- strict encoding (CONTRACT section 1) -------------------------------------

_B64U_RE = re.compile(r"^[A-Za-z0-9_-]*$")


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def b64u_dec(s: str) -> bytes:
    """Strict base64url without padding; one byte string has one text form."""
    if not isinstance(s, str) or not _B64U_RE.match(s):
        raise ValueError("not base64url")
    raw = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    if b64u(raw) != s:
        raise ValueError("non-canonical base64url")
    return raw


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate JSON key")
        d[k] = v
    return d


def _loads_strict(b: bytes):
    return json.loads(b, object_pairs_hook=_no_dup_pairs)


# ---- verification ---------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    ok: bool
    reason: str               # "ok" or a stable reject code (CONTRACT section 4)
    claims: dict | None = None  # the payload, only when ok


def peek_kid(token: str) -> str | None:
    """The kid from an UNVERIFIED header, only to choose a JWKS refetch. Never trusted."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header = _loads_strict(b64u_dec(parts[0]))
        kid = header.get("kid") if isinstance(header, dict) else None
        return kid if isinstance(kid, str) else None
    except Exception:  # noqa: BLE001 - any bad input means "no kid"
        return None


def verify(
    token: str, jwks: dict, now: int, *, issuer: str, audience: str = AUD_GEN,
    required_scope: str | None = None,
    revoked_jti: frozenset[str] | set[str] = frozenset(),
    not_before_by_sub: dict[str, int] | None = None,
) -> Result:
    """Return Result(ok, reason, claims). Never raises on bad input.

    Check order (fixed by the contract): structure, alg, typ, key, signature,
    claims, revocation (jti, then per-sub not-before), scope.
    """
    def no(r: str) -> Result:
        return Result(False, r, None)

    if not isinstance(token, str):
        return no("malformed")
    parts = token.split(".")
    if len(parts) != 3:
        return no("malformed")
    try:
        header = _loads_strict(b64u_dec(parts[0]))
        payload = _loads_strict(b64u_dec(parts[1]))
        sig = b64u_dec(parts[2])
    except Exception:  # noqa: BLE001
        return no("malformed")
    if not isinstance(header, dict) or not isinstance(payload, dict):
        return no("malformed")
    # alg: exactly EdDSA. The key type is never read from the token.
    if header.get("alg") != "EdDSA":
        return no("bad_alg")
    if header.get("typ") != TYP:
        return no("bad_typ")
    kid = header.get("kid")
    if not isinstance(kid, str):
        return no("unknown_kid")
    keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
    key = next((k for k in keys if isinstance(k, dict) and k.get("kid") == kid), None)
    if key is None or key.get("kty") != "OKP" or key.get("crv") != "Ed25519":
        return no("unknown_kid")
    try:
        Ed25519PublicKey.from_public_bytes(b64u_dec(key["x"])).verify(
            sig, f"{parts[0]}.{parts[1]}".encode())
    except (InvalidSignature, ValueError, KeyError, TypeError):
        return no("bad_signature")
    # claims; nothing above trusted a claim
    if any(c not in payload for c in REQUIRED):
        return no("missing_claim")
    if payload["iss"] != issuer:
        return no("bad_iss")
    if payload["aud"] != audience:  # a list audience is refused
        return no("bad_aud")
    iat, exp = payload["iat"], payload["exp"]
    if type(iat) is not int or type(exp) is not int:
        return no("malformed")
    if exp <= iat:
        return no("bad_lifetime")
    if exp - iat > MAX_LIFETIME_S:
        return no("lifetime_too_long")
    if now >= exp + LEEWAY_S:
        return no("expired")
    if iat > now + LEEWAY_S:
        return no("not_yet_valid")
    if payload["tier"] not in TIERS:
        return no("bad_tier")
    if payload["billing_state"] not in BILLING:
        return no("bad_billing_state")
    sc = payload["scopes"]
    if not (isinstance(sc, list) and all(isinstance(s, str) for s in sc)):
        return no("malformed")
    if not isinstance(payload["sub"], str) or not payload["sub"]:
        return no("malformed")
    if not isinstance(payload["jti"], str) or not payload["jti"]:
        return no("malformed")
    if "ws" in payload:
        ws = payload["ws"]
        if not (isinstance(ws, list) and all(isinstance(w, str) for w in ws)):
            return no("malformed")
    if payload["jti"] in revoked_jti:
        return no("revoked")
    nb = (not_before_by_sub or {}).get(payload["sub"])
    if nb is not None and iat < nb:
        return no("revoked")
    if required_scope is not None and required_scope not in sc:
        return no("missing_scope")
    return Result(True, "ok", payload)


def check_inference_allowed(payload: dict) -> int:
    """HTTP status for a verified token on a metered route (CONTRACT section 5).

    200 allowed, 402 payment required, 403 suspended. The 429 spend-cap rule is
    separate (blocked-sub set from the billing topic).
    """
    if payload.get("billing_state") == "suspended":
        return 403
    if payload.get("tier") == "free" or payload.get("billing_state") != "active":
        return 402
    ref = payload.get("allowance_ref")
    if not isinstance(ref, str) or not ref:
        return 402
    return 200


# ---- JWKS cache (CONTRACT section 3) ---------------------------------------------

def check_jwks_url(url: str) -> None:
    """https only; plain http is accepted only for a loopback host (local dev)."""
    parts = urlsplit(url or "")
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"):
        return
    raise ValueError("CAPABILITY_JWKS_URL must be an https URL (http only for a loopback host)")


def sanitize_jwks(doc) -> dict:
    """Keep only OKP / Ed25519 keys that have a string kid and a string x."""
    keys = doc.get("keys") if isinstance(doc, dict) else None
    out = []
    for k in keys if isinstance(keys, list) else []:
        if (isinstance(k, dict) and k.get("kty") == "OKP" and k.get("crv") == "Ed25519"
                and isinstance(k.get("kid"), str) and isinstance(k.get("x"), str)):
            out.append({"kty": "OKP", "crv": "Ed25519", "kid": k["kid"], "x": k["x"]})
    return {"keys": out}


async def http_fetch_jwks(url: str) -> dict:
    """Default fetcher: one GET, no redirects, bounded size."""
    import httpx

    async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
        resp = await client.get(url, headers={"Accept": "application/json"})
    if resp.status_code != 200:
        raise RuntimeError(f"JWKS fetch returned HTTP {resp.status_code}")
    if len(resp.content) > JWKS_MAX_BYTES:
        raise RuntimeError("JWKS document too large")
    return resp.json()


class JwksCache:
    """The issuer's public keys, cached for JWKS_TTL_S.

    get(kid): returns the cached set; refetches when the set is older than the
    TTL, or once (rate-limited) when `kid` is not in the set. A failed fetch keeps
    the last good set (an empty set before the first success, which makes every
    token `unknown_kid`: fail closed).
    """

    def __init__(self, url: str, fetch: Callable[[str], Awaitable[dict]] = http_fetch_jwks,
                 clock: Callable[[], float] = time.monotonic):
        self.url = url
        self._fetch = fetch
        self._clock = clock
        self._jwks: dict = {"keys": []}
        self._fetched_at: float | None = None
        self._unknown_refetch_at: float | None = None
        self._lock = asyncio.Lock()

    def _kids(self) -> set[str]:
        return {k["kid"] for k in self._jwks["keys"]}

    async def _refresh(self) -> None:
        try:
            doc = await self._fetch(self.url)
            self._jwks = sanitize_jwks(doc)
        except Exception as exc:  # noqa: BLE001 - keep serving on the last set
            logger.warning("capability JWKS fetch failed (%s); keeping %d cached key(s)",
                           type(exc).__name__, len(self._jwks["keys"]))
        finally:
            # A failed fetch also restarts the TTL, so a broken issuer is asked
            # again after the TTL, not on every request.
            self._fetched_at = self._clock()

    async def get(self, kid: str | None) -> dict:
        async with self._lock:
            now = self._clock()
            if self._fetched_at is None or now - self._fetched_at >= JWKS_TTL_S:
                await self._refresh()
            elif kid is not None and kid not in self._kids():
                last = self._unknown_refetch_at
                if last is None or now - last >= JWKS_UNKNOWN_KID_MIN_INTERVAL_S:
                    self._unknown_refetch_at = now
                    await self._refresh()
            return self._jwks


# ---- revocation and billing state (CONTRACT section 6, interface doc 2.4 / 3) ----

def _parse_ts(value) -> float | None:
    """ISO-8601 (or integer seconds) -> epoch seconds; None if unusable."""
    if type(value) is int:
        return float(value)
    if isinstance(value, str) and value:
        try:
            dt = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    return None


class AuthState:
    """Revoked jti set, per-sub not-before map, and blocked subs. Thread-safe.

    Fed by the MQ subscriber thread (revocation and billing topics) and read by
    request handlers. Revocation entries expire after REVOCATION_HOLD_S. The
    blocked set has no expiry: an "ok" event removes a sub.
    """

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._jti: dict[str, float] = {}               # jti -> drop time
        self._nb: dict[str, tuple[int, float]] = {}    # sub -> (not_before, drop time)
        self._blocked: dict[str, float] = {}           # sub -> ts of the deciding event
        self._billing_ts: dict[str, float] = {}
        # Snapshot bookkeeping (backstop A).
        self._snapshot_ts: float | None = None         # ts of the newest snapshot applied
        self._snapshot_at: float | None = None         # our clock when it was applied
        self._pending: dict[str, dict] = {}            # event_id -> parts of a split snapshot

    def _prune(self, now: float) -> None:
        for j in [j for j, t in self._jti.items() if t <= now]:
            del self._jti[j]
        for s in [s for s, (_, t) in self._nb.items() if t <= now]:
            del self._nb[s]

    def apply_revocation(self, ev) -> bool:
        if not isinstance(ev, dict):
            return False
        jti, sub, nb = ev.get("jti"), ev.get("sub"), ev.get("not_before")
        if jti is not None and (not isinstance(jti, str) or not jti):
            return False
        if sub is not None and (not isinstance(sub, str) or not sub):
            return False
        if nb is not None and type(nb) is not int:
            return False
        if jti is None and sub is None:
            return False
        now = self._clock()
        with self._lock:
            if jti is not None:
                self._jti[jti] = now + REVOCATION_HOLD_S
            if sub is not None:
                t = nb if nb is not None else int(now)
                prev = self._nb.get(sub)
                if prev is None or t > prev[0]:
                    self._nb[sub] = (t, max(t, now) + REVOCATION_HOLD_S)
        return True

    def apply_billing(self, ev) -> bool:
        if not isinstance(ev, dict):
            return False
        sub, state = ev.get("sub"), ev.get("state")
        if not isinstance(sub, str) or not sub or state not in ("blocked", "ok"):
            return False
        ts = _parse_ts(ev.get("ts"))
        with self._lock:
            last = self._billing_ts.get(sub)
            if ts is not None and last is not None and ts < last:
                return True  # older than what we already applied: ignore
            if ts is not None:
                self._billing_ts[sub] = ts
            if state == "blocked":
                self._blocked[sub] = ts or self._clock()
            else:
                self._blocked.pop(sub, None)
        return True

    # ---- full snapshot (oasis/system/auth/billing/snapshot) ----

    @staticmethod
    def _parse_snapshot(ev) -> dict | None:
        """Validate one snapshot message. None = refuse the whole message, so a
        malformed snapshot never changes any state."""
        if not isinstance(ev, dict):
            return None
        event_id = ev.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            return None
        ts = _parse_ts(ev.get("ts"))
        if ts is None:
            return None
        part, of = ev.get("part"), ev.get("of")
        if part is None and of is None:
            part = of = 1
        elif type(part) is not int or type(of) is not int or not 1 <= part <= of <= SNAPSHOT_MAX_PARTS:
            return None
        lists = []
        for key in ("blocked", "revoked_subs", "revoked_jti"):
            items = ev.get(key, [])
            if not isinstance(items, list) or len(items) > SNAPSHOT_MAX_LIST:
                return None
            lists.append(items)
        blocked: dict[str, float] = {}
        for e in lists[0]:
            if not isinstance(e, dict) or not isinstance(e.get("sub"), str) or not e["sub"]:
                return None
            ets = ts
            if e.get("ts") is not None:
                ets = _parse_ts(e["ts"])
                if ets is None or ets > ts:
                    return None
            blocked[e["sub"]] = max(ets, blocked.get(e["sub"], ets))
        rsubs: dict[str, int] = {}
        for e in lists[1]:
            if (not isinstance(e, dict) or not isinstance(e.get("sub"), str) or not e["sub"]
                    or type(e.get("not_before")) is not int):
                return None  # not_before is required: a receive-time default would move on every snapshot
            rsubs[e["sub"]] = max(e["not_before"], rsubs.get(e["sub"], e["not_before"]))
        rjti: set[str] = set()
        for e in lists[2]:
            if not isinstance(e, dict) or not isinstance(e.get("jti"), str) or not e["jti"]:
                return None
            rjti.add(e["jti"])
        return {"event_id": event_id, "ts": ts, "part": part, "of": of,
                "blocked": blocked, "revoked_subs": rsubs, "revoked_jti": rjti}

    def apply_snapshot(self, ev) -> bool:
        """Apply the full billing and revocation state from auth.

        Returns False for a malformed message (state untouched), True otherwise,
        including when the snapshot is old, a replay, or a part of a split
        snapshot that is not complete yet. State only becomes "ready" when a
        complete, fresh snapshot has been applied.

        - Listed blocked subs are added with the same per-sub ordering rule as
          apply_billing (an entry older than the event we hold is ignored).
        - A locally blocked sub that is NOT listed is dropped when its deciding
          event is older than the snapshot (heals a missed "ok").
        - Revocations are merged, never removed (they expire by hold time).
        """
        snap = self._parse_snapshot(ev)
        if snap is None:
            return False
        now = self._clock()
        if snap["ts"] > now + SNAPSHOT_FUTURE_S:
            return False
        if snap["ts"] < now - SNAPSHOT_MAX_AGE_S:
            logger.warning("ignoring an old state snapshot (%.0f s old)", now - snap["ts"])
            return True
        with self._lock:
            if self._snapshot_ts is not None and snap["ts"] < self._snapshot_ts:
                return True  # older than the snapshot we already applied
            if snap["of"] > 1:
                for k in [k for k, p in self._pending.items() if now - p["t"] > SNAPSHOT_PENDING_TTL_S]:
                    del self._pending[k]
                pend = self._pending.get(snap["event_id"])
                if pend is not None and (pend["of"] != snap["of"] or pend["ts"] != snap["ts"]):
                    del self._pending[snap["event_id"]]
                    return False
                if pend is None:
                    while len(self._pending) >= SNAPSHOT_PENDING_MAX:
                        del self._pending[min(self._pending, key=lambda k: self._pending[k]["t"])]
                    pend = self._pending[snap["event_id"]] = {
                        "of": snap["of"], "ts": snap["ts"], "t": now, "parts": {}}
                pend["parts"][snap["part"]] = snap
                if len(pend["parts"]) < pend["of"]:
                    return True
                del self._pending[snap["event_id"]]
                merged = {"ts": snap["ts"], "blocked": {}, "revoked_subs": {}, "revoked_jti": set()}
                for p in pend["parts"].values():
                    for sub, t in p["blocked"].items():
                        merged["blocked"][sub] = max(t, merged["blocked"].get(sub, t))
                    for sub, nb in p["revoked_subs"].items():
                        merged["revoked_subs"][sub] = max(nb, merged["revoked_subs"].get(sub, nb))
                    merged["revoked_jti"] |= p["revoked_jti"]
                snap = merged
            self._apply_full(snap, now)
        return True

    def _apply_full(self, snap: dict, now: float) -> None:
        ts = snap["ts"]
        for sub, ets in snap["blocked"].items():
            last = self._billing_ts.get(sub)
            if last is not None and ets < last:
                continue  # we hold a newer decision for this sub
            self._billing_ts[sub] = ets
            self._blocked[sub] = ets
        for sub in [s for s in self._blocked if s not in snap["blocked"]]:
            decided = self._billing_ts.get(sub, self._blocked[sub])
            if decided < ts - SNAPSHOT_HEAL_GRACE_S:
                del self._blocked[sub]
                self._billing_ts[sub] = ts
                logger.warning("state snapshot healed a missed 'ok' for a blocked sub")
        for sub, nb in snap["revoked_subs"].items():
            prev = self._nb.get(sub)
            if prev is None or nb > prev[0]:
                self._nb[sub] = (nb, max(nb, now) + REVOCATION_HOLD_S)
        for jti in snap["revoked_jti"]:
            self._jti.setdefault(jti, now + REVOCATION_HOLD_S)
        self._snapshot_ts = max(ts, self._snapshot_ts or ts)
        self._snapshot_at = now

    @property
    def state_ready(self) -> bool:
        with self._lock:
            return self._snapshot_at is not None

    def state_age_s(self) -> float | None:
        """Seconds since the last applied snapshot; None before the first."""
        with self._lock:
            return None if self._snapshot_at is None else max(0.0, self._clock() - self._snapshot_at)

    def blocked_count(self) -> int:
        with self._lock:
            return len(self._blocked)

    def revoked_jti(self) -> frozenset[str]:
        with self._lock:
            self._prune(self._clock())
            return frozenset(self._jti)

    def not_before_by_sub(self) -> dict[str, int]:
        with self._lock:
            self._prune(self._clock())
            return {s: nb for s, (nb, _) in self._nb.items()}

    def is_blocked(self, sub: str) -> bool:
        with self._lock:
            return sub in self._blocked
