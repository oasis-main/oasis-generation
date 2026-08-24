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
                     `us.anthropic.claude-opus-4-8`). Auth is the AWS SDK
                     default credential chain (the gateway holds the key; bots
                     never do). See bedrock.py.
  - "bedrock_mantle"— Amazon Bedrock **mantle** endpoint via the OpenAI Responses
                     API (GPT-5.6-sol). NOT Converse. `base_url` is the mantle
                     endpoint. Backend module not built yet — entries stay
                     disabled (design doc §2.4).

The frontier models below all resolve through Bedrock on a single AWS key, so
the catalog's aspirational self-hosted tiers (glm-5.2, deepseek-v4-*) can be
served NOW via proxying while their owned-runner path is still on the roadmap —
same product, cheaper backend until the economics justify self-hosting.
"""

from typing import Literal

from pydantic import BaseModel, Field

Backend = Literal["runner", "openai_compat", "bedrock", "bedrock_mantle"]


class Profile(BaseModel):
    """Concrete inference params for one named profile (fast|balanced|deep).

    effort: reasoning-effort level passed to the model (Anthropic/Bedrock
      low|medium|high|xhigh|max; OpenAI Responses minimal|low|medium|high), or
      None when the model has no effort control (the gateway omits it).
    max_tokens: output-token ceiling for this profile (thinking shares it).
    """

    effort: str | None = None
    max_tokens: int = 4096


class InferenceDefaults(BaseModel):
    """Per-model inference policy the gateway enforces (param unification).

    Single source of truth for how each model is asked to think and sample, so
    the whole fleet inherits consistent, *valid* settings (every request funnels
    through the gateway). See oasis-x/.swarm/GENERATIVE_BEDROCK_INFERENCE_DESIGN.md.

    thinking:
      "adaptive"  — adaptive-thinking Claude (Opus 4.8, Sonnet 5): the gateway
                    sends thinking={"type":"adaptive"} + output_config.effort.
      "off"/"native" — no thinking field injected (native = intrinsic to the
                    engine, e.g. runner GGUFs or probe-pending Bedrock opens).
      "always_on" — thinking cannot be disabled (reserved; no roster model today).
    allow_temperature / allow_sampling:
      False strips temperature / top_p before the call. The adaptive-Claude
      removed-set (opus-4.8, sonnet-5) 400s on `temperature` (verified live
      2026-07-24); Anthropic also rejects sampling params while thinking is on.
    profiles / default_profile:
      fast|balanced|deep -> Profile. default_profile is applied when the caller
      sends neither a `profile` nor an explicit `reasoning_effort`.
    """

    thinking: Literal["off", "adaptive", "native", "always_on"] = "native"
    allow_temperature: bool = True
    allow_sampling: bool = True
    default_profile: str = "balanced"
    profiles: dict[str, Profile] = Field(default_factory=dict)


def _adaptive_claude(deep_effort: str = "xhigh") -> InferenceDefaults:
    """Inference policy for adaptive-thinking Claude on Bedrock (Opus 4.8,
    Sonnet 5). Temperature/sampling stripped — the removed-set 400s on them
    (verified live 2026-07-24)."""
    return InferenceDefaults(
        thinking="adaptive",
        allow_temperature=False,
        allow_sampling=False,
        default_profile="balanced",
        profiles={
            "fast": Profile(effort="low", max_tokens=4096),
            "balanced": Profile(effort="high", max_tokens=16000),
            "deep": Profile(effort=deep_effort, max_tokens=32000),
        },
    )


def _native_thinking(fast: int = 4096, balanced: int = 8192, deep: int = 16000) -> InferenceDefaults:
    """Inference policy for models whose thinking is intrinsic / not yet
    gateway-controllable (runner GGUFs, probe-pending Bedrock opens like GLM-5).
    No thinking/effort injected; only max_tokens is set per profile."""
    return InferenceDefaults(
        thinking="native",
        default_profile="balanced",
        profiles={
            "fast": Profile(max_tokens=fast),
            "balanced": Profile(max_tokens=balanced),
            "deep": Profile(max_tokens=deep),
        },
    )


class CatalogEntry(BaseModel):
    public_id: str
    upstream_id: str
    tier: str
    enabled: bool
    backend: Backend = "runner"
    # openai_compat / bedrock_mantle only: where to forward and which env var holds the bearer key.
    base_url: str | None = None
    api_key_env: str | None = None
    # Per-model inference policy (thinking/effort/temperature/profiles). None =
    # legacy passthrough (no gateway-side param injection).
    inference: InferenceDefaults | None = None
    notes: str = ""

    def capabilities(self) -> dict:
        """Per-model capability descriptor surfaced on /v1/models (design §7.7).

        Drives the Telegram /genconfig surface, which renders ONLY the controls a
        model actually supports — so a temperature slider never appears for a
        model that 400s on it, effort is hidden where it has no effect, etc.
        Derived entirely from the `inference` policy so one catalog edit
        propagates to the UI. `inference=None` -> a permissive legacy descriptor.
        """
        inf = self.inference
        if inf is None:
            return {
                "supports_thinking": False,
                "thinking_forced": False,
                "supports_effort": False,
                "effort_levels": [],
                "supports_temperature": True,
                "supports_sampling": True,
                "max_output_tokens_ceiling": None,
                "profiles": [],
                "default_profile": None,
            }
        thinking_on = inf.thinking in ("adaptive", "always_on")
        # Effort choices, in profile order (fast->balanced->deep), deduped.
        effort_levels: list[str] = []
        for prof in inf.profiles.values():
            if prof.effort and prof.effort not in effort_levels:
                effort_levels.append(prof.effort)
        ceilings = [prof.max_tokens for prof in inf.profiles.values()]
        return {
            # Gateway-controllable extended thinking (adaptive). Native thinking
            # (intrinsic to the engine) is not exposed as a user control.
            "supports_thinking": inf.thinking == "adaptive",
            "thinking_forced": inf.thinking == "always_on",
            "supports_effort": thinking_on and bool(effort_levels),
            "effort_levels": effort_levels,
            # Temperature only if the model allows it AND thinking isn't forced/adaptive
            # (the gateway strips it in those cases, so the control would be a no-op).
            "supports_temperature": inf.allow_temperature and not thinking_on,
            "supports_sampling": inf.allow_sampling and not thinking_on,
            "max_output_tokens_ceiling": max(ceilings) if ceilings else None,
            "profiles": list(inf.profiles.keys()),
            "default_profile": inf.default_profile,
        }


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
        # Local llama.cpp: thinking is intrinsic (not gateway-controllable); the
        # profiles advertise max_tokens for the UI (small — 36GB-Mac / OOM cap).
        inference=_native_thinking(2048, 4096, 8192),
        notes="Community distill (Composer 2.5 + Fable 5 traces). GEN-002 pass 2026-07-12: "
        "cleared for personal/fleet use; customer tier still blocked. "
        "Pinned HF rev 1380be1796e559fca96b4107599285cab3ddbb92.",
    ),
    CatalogEntry(
        public_id="gemma-4-12b-agentic",
        upstream_id="huggingface.co/yuxinlu1/gemma-4-12b-agentic-fable5-composer2.5-v2-3.5x-tau2-gguf:Q4_K_M",
        tier="S",
        enabled=True,
        inference=_native_thinking(2048, 4096, 8192),
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
        "(Bedrock deepseek-v3.2 stand-in removed 2026-07-24 in the roster trim.)",
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
    # Frontier models on the single oasis-dev AWS key (us-east-1 inference
    # profiles; IDs verified live 2026-07-24 via `aws bedrock
    # list-inference-profiles`). Roster trimmed 2026-07-24 (design doc §0.1):
    # removed Sonnet 4.6, Haiku 4.5, gpt-oss-120b, deepseek-v3.2, llama-4-maverick;
    # Fable 5 not added (data-retention refusal + cost). Per-model inference
    # policy (thinking/effort/temperature) lives in the `inference` blocks.
    CatalogEntry(
        public_id="claude-opus-4-8",
        upstream_id="us.anthropic.claude-opus-4-8",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        inference=_adaptive_claude("xhigh"),
        notes="Anthropic Claude Opus 4.8 via Bedrock. Adaptive thinking; effort "
        "fast=low / balanced=high / deep=xhigh; temperature removed (400s, verified 2026-07-24).",
    ),
    CatalogEntry(
        public_id="claude-opus-5",
        upstream_id="us.anthropic.claude-opus-5",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        inference=_adaptive_claude("xhigh"),
        notes="Anthropic Claude Opus 5 via Bedrock (added 2026-08-24, ADM-048). Until "
        "now Opus 5 was reachable ONLY by Nimbus, which carries a direct amazon-bedrock "
        "provider on Mike's personal IAM key; the other six bots had no route to it at "
        "all and `claude-opus-5` 404'd here. Serving it through the gateway puts every "
        "bot on one metered path and off per-bot AWS credentials. Inference policy "
        "mirrors Opus 4.8 (adaptive thinking, temperature removed). Upstream verified "
        "live against Converse before this entry was added: HTTP 200, real token usage.",
    ),
    CatalogEntry(
        public_id="claude-sonnet-5",
        upstream_id="us.anthropic.claude-sonnet-5",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        inference=_adaptive_claude("high"),
        notes="Anthropic Claude Sonnet 5 via Bedrock (added 2026-07-24). Adaptive thinking; "
        "deep effort capped at high; ~30% heavier tokenizer, hence the roomier max_tokens. "
        "Temperature removed (400s).",
    ),
    CatalogEntry(
        public_id="glm-5",
        upstream_id="zai.glm-5",
        tier="bedrock",
        enabled=True,
        backend="bedrock",
        inference=_native_thinking(),
        notes="Z.AI GLM-5 via Bedrock. Kept 2026-07-24 as the interim stand-in for the "
        "self-hosted glm-5.2 tier. Thinking left native (no adaptive/effort injection) "
        "pending a capability probe of its Bedrock reasoning support.",
    ),
    # ---- Bedrock-mantle / OpenAI Responses (bedrock_mantle backend) --------
    # GPT-5.6-sol is served ONLY on the bedrock-mantle endpoint via the OpenAI
    # Responses API (NOT Converse/Chat-Completions) — design doc §2.4. DISABLED
    # until the responses/mantle backend module lands (separate GEN slice);
    # resolve() skips it so it 404s rather than mis-routing to Converse.
    CatalogEntry(
        public_id="gpt-5.6-sol",
        upstream_id="openai.gpt-5.6-sol",
        tier="bedrock",
        enabled=True,
        backend="bedrock_mantle",
        base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        inference=InferenceDefaults(
            # Reasoning is intrinsic; effort dials it. Ladder verified live
            # 2026-07-24: none/low/medium/high/xhigh (NOT "minimal").
            thinking="always_on",
            allow_temperature=False,
            default_profile="balanced",
            profiles={
                "fast": Profile(effort="low", max_tokens=4096),
                "balanced": Profile(effort="medium", max_tokens=16000),
                "deep": Profile(effort="high", max_tokens=32000),
            },
        ),
        notes="OpenAI GPT-5.6-sol via the bedrock-mantle Responses API (backend: mantle.py, "
        "SigV4 auth verified live 2026-07-24). 272K ctx; text + images + tool-calling (all "
        "verified live); profile -> reasoning.effort (low/medium/high) + max_output_tokens + "
        "text.verbosity; no temperature. store=False. Streaming is a fake-stream (true token "
        "streaming is a follow-up).",
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
    # ALREADY behind this gateway via the `bedrock` entries above (opus-4-8 /
    # sonnet-5), which is the recommended route for the consolidation. If a direct
    # (non-Bedrock-billed) Anthropic route is ever
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
