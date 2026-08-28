"""Runtime configuration for the oasis-generation gateway.

Everything comes from the environment so the same image runs locally
(against Docker Model Runner) and in the EU cloud (against a vLLM runner).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Upstream OpenAI-compatible engine. Local default is Docker Model
    # Runner's in-VM hostname; in the cloud this points at the vLLM runner.
    upstream_base_url: str = "http://model-runner.docker.internal/engines/llama.cpp/v1"

    # Comma-separated bearer tokens. Empty string = auth disabled (local dev
    # only — never deploy without tokens).
    service_tokens: str = ""

    # Request timeout toward the upstream engine, in seconds. Generation on
    # a cold or busy runner can be slow; keep this generous.
    upstream_timeout_s: float = 300.0

    # AWS region for the Bedrock backend (GEN-003). Bedrock credentials come
    # from the AWS SDK default chain (env keys / profile / role) — the gateway
    # holds them, the bots never do.
    bedrock_region: str = "us-east-1"

    model_config = {"env_prefix": "OASIS_GENERATION_"}

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
