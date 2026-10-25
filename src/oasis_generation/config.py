"""Runtime configuration for the oasis-generation gateway.

Everything comes from the environment so the same image runs locally
(against Docker Model Runner) and in the EU cloud (against a vLLM runner).
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings


def _env(name: str, default):
    """A setting read from the unprefixed name in the ORG-171 interface doc
    (section 5.4) or, as a fallback, from OASIS_GENERATION_<name>."""
    return Field(default, validation_alias=AliasChoices(name, f"OASIS_GENERATION_{name}"))


class Settings(BaseSettings):
    # Upstream OpenAI-compatible engine. Local default is Docker Model
    # Runner's in-VM hostname; in the cloud this points at the vLLM runner.
    upstream_base_url: str = "http://model-runner.docker.internal/engines/llama.cpp/v1"

    # Comma-separated bearer tokens. Empty string = auth disabled (local dev
    # only — never deploy without tokens).
    service_tokens: str = ""

    # Fail closed (GEN-008): any internet-facing deployment sets this. With it
    # set and no service tokens, the gateway refuses to start and answers 503
    # to every authenticated route, instead of serving everyone.
    require_auth: bool = False

    # Request timeout toward the upstream engine, in seconds. Generation on
    # a cold or busy runner can be slow; keep this generous.
    upstream_timeout_s: float = 300.0

    # AWS region for the Bedrock backend (GEN-003). Bedrock credentials come
    # from the AWS SDK default chain (env keys / profile / role) — the gateway
    # holds them, the bots never do.
    bedrock_region: str = "us-east-1"

    # GEN-008. All optional; with none set the gateway behaves exactly as before.
    # Route table (routes.yaml, hot reloaded). Empty = legacy catalog routing.
    routes_file: str = ""
    # BYOK providers loaded (as "pending") at startup; activated only by a passing
    # conformance run (POST /v1/byok/{name}/conformance).
    byok_file: str = ""
    # Extra outbound ports for BYOK endpoints, comma list. Default: 443 only.
    # Operator-set; a tenant cannot add a port (review S3).
    byok_extra_ports: str = ""
    # Extra X label sets (yaml list of LabelSet); gate.v1 is built in.
    labelsets_file: str = ""
    # The decider sidecar behind /v1/x/decide and /v1/gate.
    decider_url: str = "http://oasis-decider:8790/v1/decide"
    x_timeout_s: float = 10.0

    # ── Per-client route switch (GEN-008, remote.py) ──
    # The other gateway a client can be switched to (the Hetzner VM). Must be
    # https; empty = no remote, every client stays local. Includes /v1.
    remote_base_url: str = ""
    # Secrets come from FILES (mounted read-only), never env vars, so neither
    # `docker compose config` nor `docker inspect` can print them.
    remote_token_file: str = "/secrets/generation/remote-token"
    admin_token_file: str = "/secrets/generation/admin-token"
    # Route state + audit log. Host-side, so a recreate keeps the switches.
    route_state_file: str = "/state/routing.json"
    # Clients allowed to switch THEMSELVES with their own token (Telegram
    # /genroute). Comma list of client names, "*" = every token client, "" = none.
    self_switch_clients: str = ""

    # ── ORG-171 capability tokens (browser and paid users) ──
    # service    = today's service tokens only (default; existing deployments
    #              do not change).
    # capability = capability tokens only (contract v1): /v1/chat/stream and
    #              /v1/models. Every service-token route answers 401.
    # both       = a token that is a configured service token takes the service
    #              path; any other bearer is checked as a capability token.
    auth_mode: Literal["service", "capability", "both"] = "service"
    # The oasis-auth JWKS (https; http only for a loopback host) and the exact
    # `iss` the tokens must carry, e.g. https://auth.dev.cloud.oasis-x.io.
    capability_jwks_url: str = _env("CAPABILITY_JWKS_URL", "")
    capability_issuer: str = _env("CAPABILITY_ISSUER", "")
    # Extra exact origins (comma list) allowed to call /v1/chat/stream besides
    # the request's own host, e.g. http://localhost:5173 for a local UI.
    stream_allowed_origins: str = ""
    # Largest /v1/chat/stream body (and /v1/chat/completions body in capability
    # mode) the gateway reads; larger bodies get 413 request_too_large.
    max_chat_body_bytes: int = _env("MAX_CHAT_BODY_BYTES", 1_048_576)

    # ── ORG-171 usage events (interface doc 5.3) ──
    # Empty MQTT_HOST = no broker: usage events go to the spool, and the 503
    # metering_unavailable rule stops metered requests after 10 minutes.
    mqtt_host: str = _env("MQTT_HOST", "")
    mqtt_port: int = _env("MQTT_PORT", 1883)
    mqtt_username: str = _env("MQTT_USERNAME", "mq-service-gen")
    mqtt_service_secret: SecretStr = _env("MQTT_SERVICE_SECRET", SecretStr(""))
    # A random suffix is always added, so two gateway tasks never share an id.
    mqtt_client_id: str = _env("MQTT_CLIENT_ID", "oasis-gen-gateway")
    # Must be writable by uid 10001. /tmp works in any container (ephemeral:
    # a spool there is lost when the task stops); point it at a volume to keep it.
    usage_spool_dir: str = _env("USAGE_SPOOL_DIR", "/tmp/oasis-generation/usage-spool")
    usage_spool_max: int = _env("USAGE_SPOOL_MAX", 1000)

    model_config = {"env_prefix": "OASIS_GENERATION_", "populate_by_name": True}

    @property
    def capability_enabled(self) -> bool:
        return self.auth_mode in ("capability", "both")

    @property
    def remote_configured(self) -> bool:
        return self.remote_base_url.startswith("https://") and bool(
            self.remote_token_file and os.path.exists(self.remote_token_file)
        )

    def may_self_switch(self, client: str) -> bool:
        allowed = {c.strip() for c in self.self_switch_clients.split(",") if c.strip()}
        if "*" in allowed:
            return client in set(self.token_clients.values())
        return client in allowed

    @property
    def tokens(self) -> set[str]:
        return set(self.token_clients)

    @property
    def token_clients(self) -> dict[str, str]:
        """Bearer token -> caller name.

        Accepts two entry shapes in OASIS_GENERATION_SERVICE_TOKENS, comma
        separated, so an existing deployment keeps working unchanged:

          name:token   -> attributed to `name`
          token        -> attributed to "unattributed"

        WHY THIS EXISTS (ADM-050): until 2026-08-26 every bot in the fleet
        presented the SAME token, so nothing anywhere could say which bot
        caused which cost — not this gateway, and not Bedrock either, since
        the gateway holds one AWS credential for all of them. A month in which
        87.5% of spend came from a single model could not be pinned to a bot
        without reading container logs by hand.

        A token containing no colon is still accepted. Rejecting the old
        single-token form would take the whole fleet offline on deploy, which
        is a far worse outcome than degraded attribution.
        """
        out: dict[str, str] = {}
        for raw in self.service_tokens.split(","):
            item = raw.strip()
            if not item:
                continue
            if ":" in item:
                name, _, tok = item.partition(":")
                name, tok = name.strip(), tok.strip()
                if tok:
                    out[tok] = name or "unattributed"
            else:
                out[item] = "unattributed"
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
