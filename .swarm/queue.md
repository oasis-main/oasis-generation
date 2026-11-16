# Queue — oasis-generation (Division Level)

Items are listed in priority order within each section.
Item IDs: `GEN-<3-digit-number>` — assigned sequentially, never reused.

This queue tracks **oasis-generation** (Python, sister to oasis-voice).
Platform-level items (billing, marketing surface, control plane in oasis-ai)
stay in oasis-x `.swarm/queue.md` under ORG numbers; see
`oasis-x/.swarm/GENERATIVE_PLAN.md` §8 for the split.

---

## Active

- [ ] [GEN-001] [OPEN] Provider spike: Scaleway L40S stop/start lifecycle + vLLM tier-S bring-up
      priority: high | project: infra | division: GEN
      notes: Scaleway account under hello@oasis-x.io (admin decision 2026-07-12).
             Terraform module skeleton; 1x L40S-1-48G in fr-par-2; weights on
             block volume (scratch NVMe is ERASED on full stop); vLLM serving
             Gemma-4-12B-it. Measure: cold-boot time, weight-load time,
             TTFT-after-wake (target 1-3 min), and VERIFY billing pauses on
             power-off (per-minute, storage-only when stopped — confirmed in
             docs 2026-07-12, confirm on invoice). Budget: < EUR 20.

- [ ] [GEN-002] [PARTIAL — personal/fleet tier CLEARED 2026-07-13] Security + license vetting of community Gemma-4-12B fine-tunes
      priority: medium (was high) | project: catalog | division: GEN
      notes: FIRST PASS DONE — see docs/GEN-002-vetting.md. Both variants
             passed behavioral probes (incl. exfil-via-tool-injection on the
             agentic model); enabled=true + revision-pinned in catalog.py
             2026-07-13; Mike waived the license question for personal use
             (fallback = vanilla Gemma/GLM/DeepSeek + own fine-tunes).
             REMAINS OPEN for the customer/paid tier only: multi-turn /
             jailbreak-framed / multi-category red-team + license/provenance
             legal read. Original checklist below for that phase.
             Candidates (now enabled for personal/fleet):
               - yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF (Q4_K_M 7.38 GB)
               - yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2 (safetensors + GGUF repo)
             Both: Apache-2.0-labeled distills of google/gemma-4-12B-it from
             Composer 2.5 + Fable 5 traces; author-run tau2-telecom ~55% vs ~15% base.
             Vetting checklist:
               1. Integrity: pin exact revision hashes; verify file formats
                  (GGUF/safetensors = no code execution, but pin anyway);
                  check HF security-scanner status on the repos.
               2. Behavioral: sandboxed eval vs baseline — our own prompts +
                  a tau2 slice; red-team for backdoor-ish behaviors (trigger
                  phrases, exfil-shaped tool calls in agentic harness).
                  NOTE: cards admit "reduced safety alignment / fewer
                  refusals" — decide per-bot whether that's acceptable
                  (Nimbus fleet has its own guardrails; customer tier does not).
               3. License: cards say Apache 2.0, but Gemma derivatives must
                  flow down the Gemma Terms of Use — the relabel is likely
                  invalid. Also provenance: distilled from proprietary-model
                  outputs (Composer 2.5, Fable 5) — fine to HOST for personal
                  use, murkier for the paid tier. Legal note before customer
                  exposure.
               4. If cleared: flip enabled=true in catalog.py, pin
                  upstream_id to @<revision>.

- [/] [GEN-003] [IN PROGRESS] Gateway v0 hardening: streaming, provider fanout, per-bot service tokens
      priority: high | project: gateway | division: GEN
      notes: Current main.py is a single-upstream proxy. Add: SSE streaming
             pass-through test against live DMR; provider fanout (Anthropic/
             OpenAI/Google/Bedrock) so bot .env provider keys collapse into
             the gateway (see memory note interlocks — egress allowlist
             narrows to gateway + Telegram); per-bot tokens with per-token
             rate/spend caps; structured usage events for metering.
      DONE 2026-07-15 — provider fanout via Bedrock (Converse, direct boto3;
             chosen over LiteLLM for robustness with brand-new 2026 model ids).
             catalog.py grew a `backend` field (runner | openai_compat |
             bedrock); bedrock.py does OpenAI<->Converse (non-stream + SSE,
             reasoning_content + cache-token passthrough). Enabled: claude
             sonnet-4-6 / opus-4-8 / haiku-4-5, gpt-oss-120b, glm-5,
             deepseek-v3.2, llama-4-maverick — all on the oasis-dev key
             (us-east-1 inference profiles; IDs verified live). Creds via the
             gateway's env_file (docker/.env.example); bots hold nothing.
             Verified end-to-end (non-stream + stream + glm-5) 2026-07-15;
             12 tests green (6 bedrock translation + routing).
      DONE 2026-07-24 — inference-param unification (design doc
             oasis-x/.swarm/GENERATIVE_BEDROCK_INFERENCE_DESIGN.md). catalog.py
             gained `InferenceDefaults`/`Profile` per model (thinking/effort/
             temperature + fast|balanced|deep profiles); bedrock.py now injects
             thinking={"type":"adaptive"} + output_config.effort via
             additionalModelRequestFields (verified accepted live) and strips
             temperature/top_p on the removed-set (opus-4.8/sonnet-5 400 on them).
             Precedence: explicit request > profile > catalog default. Roster
             trimmed: added claude-sonnet-5; removed sonnet-4-6/haiku-4-5/
             gpt-oss-120b/deepseek-v3.2/llama-4-maverick; glm-5 kept (native,
             effort probe-pending). gpt-5.6-sol staged DISABLED as a new
             `bedrock_mantle` backend (OpenAI Responses endpoint — not built).
             23 tests green.
      DONE 2026-07-24 (later same day) — capability descriptors on /v1/models
             (design §7.7) + the `bedrock_mantle` backend for gpt-5.6-sol
             (mantle.py: OpenAI chat-completions <-> Responses API on the
             bedrock-mantle endpoint, SigV4 auth verified live, store=False,
             text-first). gpt-5.6-sol enabled. Verified end-to-end live
             (non-stream + fake-stream). 31 tests green.
      STILL OPEN: per-bot service tokens with rate/spend caps; structured usage
             events for metering (ties to oasis-gateway-billing); mantle
             tool-calls + true token streaming (v0 is text-first / fake-stream);
             GLM-5 reasoning capability probe. FOLLOW-ON (oasis-claw
             repo): the durable entrypoint still registers the OLD 9-model
             roster (incl. the 5 removed here) + a fallback chain referencing
             oasis-generation/claude-sonnet-4-6 — update it to the trimmed
             roster (opus-4-8/sonnet-5/glm-5 + gemma) with correct context
             windows, or those refs 404 against the gateway.

- [ ] [GEN-004] [OPEN] Wake/sleep controller + oasis-ai control-plane integration
      priority: medium | project: infra | division: GEN
      depends: GEN-001
      notes: Instance state machine (asleep/warming/ready) surfaced in API;
             wake-on-request returns 202 + Retry-After or streamed warming
             events; idle timer stops the instance. Mike spec 2026-07-13:
             activity = AUTHORIZED requests only (valid service token);
             idle_timeout default 3600 s, per-instance override; Telegram
             surface = small fleet plugin (/gen: states + wake/sleep
             buttons, model-switcher pattern) — design in GENERATIVE_PLAN
             §3.1. Control-plane routers
             (/instances /models /adapters) live in oasis-cloud/src/ai —
             this repo ships the runner-side agent + gateway hooks.
             Capacity-miss path: no GPU at power-on -> queue with ETA +
             Nebius fallback.

- [ ] [GEN-005] [OPEN] Modal LoRA pipeline (shared_modal) + NuSci fine-tune platform interface
      priority: medium | project: finetune | division: GEN
      notes: Train LoRA/QLoRA on Modal via ai_research/shared_modal
             (ResearchExperiment + manifest + volumes); HF-format adapter ->
             adapter registry -> vLLM --enable-lora hot-load, no restart.
             Gemma-4-12B first. Upstream consumer: NuSci fine-tune platform
             (dataset creation by interviewing users on topics they care
             about; bias/vocabulary mapping vs AI models AND humans) — keep
             the job interface generic: dataset-in (JSONL chat format),
             adapter+eval-report out. Modal is US-resident: disclose in ToS
             for v0, or use Modal EU region if plan allows.

- [ ] [GEN-006] [OPEN — PLAN ONLY, do not implement yet] RL + fine-tuning on the archived open-model collection (dedicated big-machine reservations)
      priority: low | project: finetune | division: GEN
      notes: Source = the sovereign model archive on Hetzner Storage Box
             u642948, built via ai_research/ops/model_archive (Terraform +
             models.txt, sha256-verified). Collection: GLM-5.2 (native FP8),
             Kimi-K3 (native 4-bit MXFP4, 1.56 TB) + Kimi-K3 2-bit
             (unsloth UD-Q2_K_XL, 861 GB), Inkling, Gemma-4-12B-agentic +
             Gemma4-31B-Fable-5-Distilled (full + GGUF), GLM-5.2-NVFP4,
             Inkling-NVFP4; trace datasets Fable-5 + GPT-5.6-Sol under
             datasets/ on the box.
             Compute: ON-DEMAND big-machine reservations, never always-on.
             Sizing established 2026-07-31: full Kimi-K3 (2.8T, all 896 experts
             must be resident, ~1.6 TB) needs ONE Scaleway B300-SXM-8-288G
             (8x B300, 2304 GB VRAM, ~EUR 60-70/hr — confirm exact price) OR a
             multi-node H100 cluster; the 2-bit Kimi (861 GB) fits a single
             8x H100 (640 GB, ~EUR 23/hr). Use per-minute billing +
             power-off-when-idle, same lifecycle as GEN-001. Copy weights
             Hetzner -> Scaleway block volume once, boot instance against it.
             Realistic targets: LoRA/QLoRA + RL (GRPO/DPO) on the mid-size
             models (Gemma) and adapters on the big MoE ones; full-FT of a
             frontier MoE is out of scope. Extends GEN-005 (Modal LoRA) to the
             archived collection and to reserved GPU (Modal caps out below this
             scale). Sovereignty: keep on EU/Scaleway; note AI-Act GPAI
             exposure. Do NOT provision until a concrete experiment is scoped.

- [ ] [GEN-008] [OPEN 2026-10-15] Serving platform v1: GEX131 essentials, Scaleway single GPU, gateway hosting + routing (own / Bedrock / BYOK), local-vs-cloud switch
      priority: high | project: serving-platform | division: GEN
      notes: Plan = oasis-x/.swarm/GENERATIVE_PLAN.md §11 (work items in §11.7).
             Builder: Kolmogorov (clone bot-work/kolmogorov/oasis-generation; his
             design docs/OX-MODELS-INTERFACE-TRAINING-DESIGN.md, 9d3980f). Infra: Yes Man.
             Mike: order the GEX131 (Robot panel) + Robot webservice credential for
             Yes Man; validate identity on the hello@ Scaleway org (unlocks quota 1 for
             L40S-1-48G, then GEN-001 under INFRA-0002); pick the gateway host (§11.4,
             recommendation: ECS Fargate eu-central-1 for customers, Mac kept for the fleet).
             Code first, no infra: route table + hot reload + X-Oasis-Route, BYOK registry
             + conformance run + SSRF guard, /v1/x/decide + /v1/gate.
             Pin vllm/vllm-openai by digest before any box runs it (Dockerfile.runner uses :latest).

- [/] [GEN-007] [PHASE 0 LIVE 2026-10-07 · audit pass 1 DONE] Decision models: audit queue, fine-tune, placement
      priority: medium | project: decision-models | division: GEN
      notes: Plan = oasis-x/.swarm/GENERATIVE_PLAN.md §10. Sidecar + reviewer
             shadow = oasis-claw CLAW-118 (live on all 7 bots).
             Audit: ai_research/ops/model_archive/decision_models.txt (queue) +
             vet_static.py. Pass 1 (43 repos incl. the whole archive): 20 PASS,
             19 REVIEW, 4 GONE, 0 FAIL -> docs/archive-vetting-static.md,
             docs/decision-models-vetting.md. GONE: autotrust gemma4-31B distill
             (+GGUF) and 2 Fable-5 trace sets; our archive copies are the only ones.
             Added to models.txt (not yet transferred, 6.2 GB): laya-typed-decisions,
             laya-multilingual, strands-decider-2B v21 (LoRA) + Qwen3.5-2B-Base.
             Next: (1) read laya base .py + Mapika engine, pplx inference.py;
             (2) archive transfer after the reaper (ADM-057) is deployed;
             (3) phase 1 fine-tune of Laya on judge labels, gate in §10.6;
             (4) GPU placement on the pooled GEX131 (§10.7).
             Owner after Mike grants gh tokens: Kolmogorov. Infra applies: Yes Man.

---

## Done

(none yet)
