"""Model catalog: public model ids -> upstream engine ids, by tier.

Tier sizing and provider economics live in oasis-x/.swarm/GENERATIVE_PLAN.md.
Only `enabled` models are served; community fine-tunes stay disabled until
they clear the GEN-002 security/license review.
"""

from pydantic import BaseModel


class CatalogEntry(BaseModel):
    public_id: str
    upstream_id: str
    tier: str
    enabled: bool
    notes: str = ""


CATALOG: list[CatalogEntry] = [
    CatalogEntry(
        public_id="gemma-4-12b-it",
        upstream_id="huggingface.co/unsloth/gemma-4-12b-it-gguf:Q4_K_M",
        tier="S",
        enabled=True,
        notes="Baseline tier-S. Local dev id shown; cloud runner uses the HF bf16/FP8 checkpoint.",
    ),
    CatalogEntry(
        public_id="gemma-4-12b-coder",
        upstream_id="huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF",
        tier="S",
        enabled=False,
        notes="Community distill (Composer 2.5 + Fable 5 traces). BLOCKED on GEN-002 vetting.",
    ),
    CatalogEntry(
        public_id="gemma-4-12b-agentic",
        upstream_id="huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2",
        tier="S",
        enabled=False,
        notes="Community distill, tau2-telecom ~55% vs ~15% base (author-run). BLOCKED on GEN-002 vetting.",
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
        notes="284B/13B-active MoE, MIT. Awaits 4-8x H100-SXM runner.",
    ),
    CatalogEntry(
        public_id="glm-5.2",
        upstream_id="zai-org/GLM-5.2",
        tier="XL",
        enabled=False,
        notes="744B/40B-active MoE, native FP8. Awaits 8x B300 reserve-on-wake runner.",
    ),
    CatalogEntry(
        public_id="deepseek-v4-pro",
        upstream_id="deepseek-ai/DeepSeek-V4-Pro",
        tier="XXL",
        enabled=False,
        notes="1.6T/49B-active MoE, MIT. Reserve-on-wake only.",
    ),
]


def enabled_models() -> list[CatalogEntry]:
    return [m for m in CATALOG if m.enabled]


def resolve(public_id: str) -> CatalogEntry | None:
    for m in CATALOG:
        if m.public_id == public_id and m.enabled:
            return m
    return None
