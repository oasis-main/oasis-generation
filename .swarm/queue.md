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

- [ ] [GEN-003] [OPEN] Gateway v0 hardening: streaming, provider fanout, per-bot service tokens
      priority: high | project: gateway | division: GEN
      notes: Current main.py is a single-upstream proxy. Add: SSE streaming
             pass-through test against live DMR; provider fanout (Anthropic/
             OpenAI/Google/Bedrock) so bot .env provider keys collapse into
             the gateway (see memory note interlocks — egress allowlist
             narrows to gateway + Telegram); per-bot tokens with per-token
             rate/spend caps; structured usage events for metering.

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

---

## Done

(none yet)
