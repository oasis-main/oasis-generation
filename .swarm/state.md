# State — oasis-generation

**2026-07-12 (Claude — repo bootstrap):**
Skeleton created during the oasis-generative planning session (plan:
oasis-x/.swarm/GENERATIVE_PLAN.md). Decisions captured that day:

- Provider: Scaleway primary / Nebius fallback (Mike approved 2026-07-12).
- Admin identity: hello@oasis-x.io owns the provider account and this
  service (Mike, 2026-07-12 — oasis-cloud infra project that provisions
  instances for clients; mike@ stays product-side for scientific/home/etc.).
- Tier-S candidates include two community Gemma-4-12B distills
  (coder + agentic, yuxinlu1/*fable5-composer2.5*) — catalog-disabled until
  GEN-002 vetting passes.
- Local dev loop validated on Mike's M4 Max via Docker Model Runner:
  Q4_K_M at 39 tok/s gen / 135 tok/s prompt; OpenAI-compatible JSON with
  usage+timings; gemma-4 thinking mode emits `reasoning_content` and eats
  max_tokens (product decision pending on exposing/budgeting it).
- Gateway skeleton: FastAPI proxy (auth, catalog, stream pass-through),
  4 tests green under .venv. Dockerfiles: slim CPU gateway + vLLM runner
  with volume-mounted weights (VOICE-001 lesson: never bake weights).

Next: GEN-001 (Scaleway spike) and GEN-002 (fine-tune vetting) are the
front of the queue; GEN-003 gateway hardening unlocks the bot-fleet key
consolidation.
