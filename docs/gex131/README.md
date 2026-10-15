# GEX131 serving project (GEN-008)

Files only. Nothing here has been applied, ordered or started. Provisioning, the Hetzner
box, weight staging and image digests all go through Yes Man as an INFRA-REQUEST.

Project: `docker/gex131/docker-compose.yml` + `docker/gex131/routes.gex131.yaml`.

| Service | Role | GPU | Network |
|---|---|---|---|
| gateway | oasis-generation (this repo), route table, /v1/x/decide, /v1/gate | no | internal + egress, 127.0.0.1:8800 |
| vllm | Gemma-4-12B BF16, `--enable-lora` | yes | internal |
| decider | X service (Laya typed decisions, CPU sidecar) | no | internal |
| embed | bge-m3 | yes | internal |
| rerank | bge-reranker | yes | internal |
| postgres | pgvector | no | internal |

## Invariants (checked by tests/test_gex131_compose.py)
1. Every image is a required variable that must be `repo@sha256:<digest>`. No tags.
2. No public port. Gateway on loopback only.
3. Only the gateway has internet egress; everything else sits on an `internal: true` network.
4. Weights come from the pre-staged read-only `gen008_models` volume. Services run offline.
5. No secret in the file. Postgres password is a file-backed secret; gateway keys are in a gitignored `.env.gateway`.

## What the operator (Yes Man) must supply
- Digests: GATEWAY_IMAGE, VLLM_IMAGE, DECIDER_IMAGE, EMBED_IMAGE, RERANK_IMAGE, PGVECTOR_IMAGE.
  DECIDER_IMAGE is built by oasis-claw; I have not seen its source.
- Volume `gen008_models` populated from the Storage Box: gemma-4-12b-agentic, laya-typed-decisions, bge-m3, bge-reranker-v2-m3.
- `docker/gex131/secrets/pg_password` and `.env.gateway` (service tokens, AWS keys). Never committed.

## Unverified / placeholders
- `VLLM_GPU_FRACTION=0.55` is a guess. vLLM, embed and rerank share the GPU(s); it must be
  tuned against the real card. The GEX131 order is still open, so the GPU count and memory are unknown.
  Three services each reserve `count: 1`; on a single-GPU box that is fine, but sizing is not verified.
- Health endpoints for decider (`:8790/health`) and the embed/rerank images are assumed, not read from source.
- The 31B distill (220 GB class models) is not in this project; only the 12B agentic model is.
- Decider CPU latency measured by console is about 2 s per 5-question call. This project does not fix that; the GPU/ONNX fast path is separate work.
- `docker compose config` has not been run (no docker in my container); only static YAML checks.

## Cost
Not estimated here. The GEX131 price comes from the Hetzner order, which is Yes Man's.
