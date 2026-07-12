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

    model_config = {"env_prefix": "OASIS_GENERATION_"}

    @property
    def tokens(self) -> set[str]:
        return {t.strip() for t in self.service_tokens.split(",") if t.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
