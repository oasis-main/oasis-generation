"""Per-client route switch: serve a caller here ("local") or forward its calls
to another oasis-generation gateway ("remote") — GEN-008, GENERATIVE_PLAN §11.6.

The Mac gateway is the fleet's only model endpoint. Each bot is a client (the
name its service token maps to, ADM-050). Switching a bot to "remote" makes
this gateway forward that bot's calls, unchanged, to the cloud gateway (the
Hetzner VM). Nothing changes inside the bot: no new egress, no new secret, no
recreate. Switching it back is the same operation.

Who can switch:
  - The admin token (the observatory on the Mac host) may switch any known
    client. The token is read from a file, never from the environment, so
    `docker compose config` and `docker inspect` cannot print it.
  - A client may switch ONLY ITSELF with its own service token (the bot's
    Telegram /genroute command), and only when its name is in
    OASIS_GENERATION_SELF_SWITCH_CLIENTS. Both targets are our own gateways,
    so the worst a misused self-switch does is move that bot's own traffic
    between them; every change is audited.

There is no silent fallback. A bot switched to "remote" gets a 502 when the
remote gateway is down — the switch is a decision about where its data goes,
and quietly serving it from the other place would undo that decision.
"""

from __future__ import annotations

import datetime as _dt
import hmac
import json
import os
import threading

TARGETS = ("local", "remote")
DEFAULT_TARGET = "local"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def read_secret_file(path: str) -> str:
    """Return the file's content stripped, or "" when absent/unreadable.
    Never logs or echoes the value."""
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def token_matches(presented: str, expected: str) -> bool:
    """Constant-time compare; an empty expected token never matches."""
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode(), expected.encode())


class RouteStore:
    """client name -> target, persisted as JSON with an append-only audit log.

    Written atomically (tmp file + os.replace) so a crash mid-write leaves the
    previous state, never a truncated file. A missing or corrupt state file
    means every client is "local" — the safe, pre-GEN-008 behaviour.
    """

    def __init__(self, path: str, audit_path: str | None = None) -> None:
        self.path = path
        self.audit_path = audit_path or os.path.join(
            os.path.dirname(path) or ".", "routing-audit.jsonl"
        )
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return {}
        clients = raw.get("clients") if isinstance(raw, dict) else None
        if not isinstance(clients, dict):
            return {}
        return {str(k): v for k, v in clients.items() if v in TARGETS}

    def get(self, client: str) -> str:
        return self._load().get(client, DEFAULT_TARGET)

    def snapshot(self) -> dict[str, str]:
        return self._load()

    def set(self, client: str, target: str, actor: str) -> tuple[str, str]:
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        with self._lock:
            clients = self._load()
            prev = clients.get(client, DEFAULT_TARGET)
            if target == DEFAULT_TARGET:
                clients.pop(client, None)  # absent == local; keeps the file small
            else:
                clients[client] = target
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"clients": clients, "updated": _now()}, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.path)
            try:
                with open(self.audit_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "ts": _now(), "actor": actor, "client": client,
                        "from": prev, "to": target,
                    }) + "\n")
            except OSError:
                pass  # audit is best-effort; the state change already landed
        return prev, target
