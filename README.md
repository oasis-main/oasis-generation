# oasis-generation

Personal generative-inference sidecar for the Oasis-X platform: an
OpenAI-compatible gateway in front of per-customer model runners in an EU
cloud, with on-demand wakeup (instances stop when idle, GPU billing stops
with them) and LoRA fine-tuning via Modal.

Sister services: [oasis-voice](https://github.com/oasis-main/oasis-voice)
(TTS/STT, flat monthly) and oasis-semantics (embeddings, flat monthly).
oasis-generation bills by active-machine-time. Full plan, tier economics,
and provider comparison: `oasis-x/.swarm/GENERATIVE_PLAN.md`.

## Model tiers

| Tier | Anchor models | Footprint |
|------|---------------|-----------|
| S | Gemma-4-12B-it (multimodal, 256K ctx) + community coder/agentic fine-tunes (pending GEN-002 vetting) | 1× 48 GB (L40S) |
| M | Gemma-4-31B-it | 1× H100 (FP8) |
| L | DeepSeek-V4-Flash (284B/13B active, MIT) | 4–8× H100-SXM |
| XL | GLM-5.2 (744B/40B active, native FP8) | 8× B300 |
| XXL | DeepSeek-V4-Pro (1.6T/49B active, MIT) | 8× B300, reserve-on-wake |

Cloud: Scaleway primary (per-minute billing, power-off = storage-only,
Terraform-first-class stop/start), Nebius fallback (H200 tier). Weights live
on persistent block volumes, never in images and never on scratch NVMe
(erased on full stop).

## Local development

The gateway proxies to Docker Model Runner, so the full API shape runs on a
laptop with no GPU cloud involved:

```sh
docker model pull hf.co/unsloth/gemma-4-12b-it-gguf:Q4_K_M
docker compose -f docker/docker-compose.yml up gateway
curl -s localhost:8800/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-12b-it",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Measured on an M4 Max (Metal, Q4_K_M): 39 tok/s generation, 135 tok/s
prompt. Note Gemma-4's thinking mode returns reasoning in a separate
`reasoning_content` field and consumes `max_tokens` budget.

## Tests

```sh
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Layout

- `src/oasis_generation/` — FastAPI gateway (auth, catalog, proxy)
- `docker/Dockerfile.gateway` — slim CPU image for the always-on gateway
- `docker/Dockerfile.runner` — vLLM GPU runner; weights mounted, not baked
- `.swarm/` — division-level queue (`GEN-XXX`) and state
