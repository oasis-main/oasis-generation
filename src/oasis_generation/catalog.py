"""Model catalog: public model ids -> upstream engine ids, by tier and backend.

Tier sizing and provider economics live in oasis-x/.swarm/GENERATIVE_PLAN.md.
Only `enabled` models are served; community fine-tunes stay disabled until
they clear the GEN-002 security/license review.

Backends (GEN-003 provider fanout):
  - "runner"       — self-hosted OpenAI-compatible engine (Docker Model Runner
                     locally, vLLM in the cloud). `upstream_id` is the engine's
                     model id; requests forward to `settings.upstream_base_url`.
  - "openai_compat"— any other OpenAI-compatible HTTP endpoint (OpenAI, direct
                     DeepSeek/Zhipu, Together, …). `upstream_id` is that
                     provider's model id; `base_url` + `api_key_env` name where
                     to send it and which env var holds the key.
  - "bedrock"      — Amazon Bedrock via the Converse API. `upstream_id` is the
                     Bedrock model id or inference-profile id (e.g.
                     `us.anthropic.claude-sonnet-4-6`). Auth is the AWS SDK
                     default credential chain (the gateway holds the key; bots
                     never do). See bedrock.py.

The frontier models below all resolve through Bedrock on a single AWS key, so
the catalog's aspirational self-hosted tiers (glm-5.2, deepseek-v4-*) can be
served NOW via proxying while their owned-runner path is still on the roadmap —
same product, cheaper backend until the economics justify self-hosting.
"""

from typing import Literal

from pydantic import BaseModel

Backend = Literal["runner", "openai_compat", "bedrock"]


class CatalogEntry(BaseModel):
    public_id: str
    upstream_id: str
    tier: str
    enabled: bool
    backend: Backend = "runner"
    # openai_compat only: where to forward and which env var holds the bearer key.
    base_url: str | None = None
    api_key_env: str | None = None
    notes: str = ""


CATALOG: list[CatalogEntry] = [
    # ---- Self-hosted tiers (runner backend) --------------------------------
    CatalogEntry(
        public_id="gemma-4-12b-it",
        upstream_id="huggingface.co/unsloth/gemma-4-12b-it-gguf:Q4_K_M",
        tier="S",
        enabled=False,
        notes="Baseline tier-S. Local copy removed 2026-07-13 (disk); re-pull or re-enable "
        "on the cloud runner (HF bf16/FP8 checkpoint).",
    ),
    CatalogEntry(
        public_id="gemma-4-12b-coder",
        upstream_id="huggingface.co/yuxinlu1/gemma-4-12b-coder-fable5-composer2.5-v1-gguf:Q4_K_M",
        tier="S",
        enabled=True,
        notes="Community distill (Composer 2.5 + Fable 5 traces). GEN-002 pass 2026-07-12: "
        "cleared for personal/fleet use; customer tier still blocked. "
        "Pinned HF rev 1380be1796e559fca96b4107599285cab3ddbb92.",
    ),
    CatalogEntry(
        public_id="gemma-4-12b-agentic",
        upstream_id="huggingface.co/yuxinlu1/gemma-4-12b-agentic-fable5-composer2.5-v2-3.5x-tau2-gguf:Q4_K_M",
        tier="S",
        enabled=True,
        notes="Community distill, tau2-telecom ~55% vs ~15% base (author-run). GEN-002 pass "
        "2026-07-12 incl. exfil-via-tool-injection probe: cleared for personal/fleet use; "
        "customer tier still blocked. Pinned HF rev 190a31365a6b80a692349be34ccdac730cad4fe4.",
    ),
    CatalogEntry(
        public_id="gemma-4-31b-it",
        upstream_id="google/gemma-4-31B-it",
        tier="M",
        enabled=False,
        notes="Awaits first cloud runner (1x H100 FP8).",
    ),
    CatalogEntry(
        public_id="deepseek-v4-flash",
        upstream_id="deepseek-ai/DeepSeek-V4-Flash",
        tier="L",
        enabled=False,
        notes="284B/13B-active MoE, MIT. Awaits 4-8x H100-SXM runner. "
        "Served now via `deepseek-v3.2` on Bedrock (see below).",
    ),
    CatalogEntry(
        public_id="glm-5.2",
        upstream_id="zai-org/GLM-5.2",
        tier="XL",
        enabled=False,
        notes="744B/40B-active MoE, native FP8. Awaits 8x B300 reserve-on-wake runner. "
        "Served now via `glm-5` on Bedrock (see below).",
    ),
    CatalogEntry(
        public_id="deepseek-v4-pro",
        upstream_id="deepseek-ai/DeepSeek-V4-Pro",
        tier="XXL",
        enabled=False,
        notes="1.6T/49B-active MoE, MIT. Reserve-on-wake only.",
    ),
    # ---- Bedrock passthrough (bedrock backend, GEN-003) --------------------
    # All resolve through the single oasis-dev AWS key (us-east-1 inference
    # profiles). IDs verified live 2026-07-15 via `aws bedrock
    # list-inference-profiles` / `list-foundation-models`.
    CatalogEntry(
        public_id="claude-sonnet-4-6",
        upstream_id="us.anthropic.claude-sonnet-4-6",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        notes="Anthropic Claude Sonnet 4.6 via Bedrock. Second Claude route (billed to the "
        "oasis-dev account) — a cross-provider failover for the direct Anthropic key.",
    ),
    CatalogEntry(
        public_id="claude-opus-4-8",
        upstream_id="us.anthropic.claude-opus-4-8",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        notes="Anthropic Claude Opus 4.8 via Bedrock.",
    ),
    CatalogEntry(
        public_id="claude-haiku-4-5",
        upstream_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        notes="Anthropic Claude Haiku 4.5 via Bedrock — cheap/fast tier.",
    ),
    CatalogEntry(
        public_id="gpt-oss-120b",
        upstream_id="openai.gpt-oss-120b-1:0",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        notes="OpenAI GPT-OSS 120B (open weights) via Bedrock. Real GPT-5 would need the "
        "OpenAI API via an openai_compat entry; gpt-oss keeps everything on oasis-dev.",
    ),
    CatalogEntry(
        public_id="glm-5",
        upstream_id="zai.glm-5",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        notes="Z.AI GLM-5 via Bedrock. Immediate stand-in for the self-hosted glm-5.2 tier.",
    ),
    CatalogEntry(
        public_id="deepseek-v3.2",
        upstream_id="deepseek.v3.2",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        notes="DeepSeek V3.2 via Bedrock. Immediate stand-in for the self-hosted deepseek-v4 tiers.",
    ),
    CatalogEntry(
        public_id="llama-4-maverick",
        upstream_id="us.meta.llama4-maverick-17b-instruct-v1:0",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        notes="Meta Llama 4 Maverick 17B via Bedrock.",
    ),
    # ---- Direct OpenAI-compatible providers (openai_compat backend) --------
    # Template, disabled: enable + set the api_key_env var to route real GPT
    # (or any OpenAI-compatible provider) instead of / alongside Bedrock.
    CatalogEntry(
        public_id="gpt-5",
        upstream_id="gpt-5",
        tier="openai",
        enabled=False,
        backend="openai_compat",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        notes="Direct OpenAI GPT-5. Disabled by default (keeps the fleet on the single "
        "oasis-dev Bedrock key); enable + provide OPENAI_API_KEY to the gateway to use.",
    ),
    # Staged for the provider-key consolidation (Mike, 2026-07-18): the fleet's
    # OpenAI / Google / Anthropic keys move OFF the bots and onto the gateway, so
    # bot egress can collapse to gateway + Telegram (the GEN-003 interlock) and the
    # per-bot `origins.exclude` for provider hosts becomes unnecessary. Left
    # DISABLED on purpose — flipping `enabled=True` + putting the key in
    # docker/.env is then the whole migration, no code change.
    #
    # Google resolves through its OpenAI-compatibility surface, so the existing
    # openai_compat path handles it unmodified (same {base_url}/chat/completions +
    # bearer shape). Verify the endpoint against current Google docs before enabling.
    CatalogEntry(
        public_id="gemini-3.1-flash-lite",
        upstream_id="gemini-3.1-flash-lite",
        tier="google",
        enabled=False,
        backend="openai_compat",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        notes="Direct Google Gemini via its OpenAI-compatible endpoint. Staged disabled for "
        "the key-consolidation move; enable + set GEMINI_API_KEY on the gateway. Cheap/fast "
        "tier — also the natural exec-autoReview reviewer model once bots route through here.",
    ),
    # Anthropic: deliberately NOT added as an openai_compat entry. Its native API
    # is the Messages API, not OpenAI chat-completions, so this backend's
    # {base_url}/chat/completions + Bearer shape does not apply as-is. Claude is
    # ALREADY behind this gateway via the three `bedrock` entries above
    # (sonnet-4-6 / opus-4-8 / haiku-4-5), which is the recommended route for the
    # consolidation. If a direct (non-Bedrock-billed) Anthropic route is ever
    # wanted, it needs either Anthropic's OpenAI-compat surface (verify it against
    # current Anthropic docs first) or a small `anthropic` backend module mirroring
    # bedrock.py — do not assume this backend will just work.
]


def enabled_models() -> list[CatalogEntry]:
    return [m for m in CATALOG if m.enabled]


def resolve(public_id: str) -> CatalogEntry | None:
    for m in CATALOG:
        if m.public_id == public_id and m.enabled:
            return m
    return None
