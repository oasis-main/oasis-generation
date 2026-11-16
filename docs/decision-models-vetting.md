# Decision models — vetting notes (GEN-007)

Plan: `oasis-x/.swarm/GENERATIVE_PLAN.md` §10. Static pass of 2026-10-07, run with
`ai_research/ops/model_archive/vet_static.py` over `models.txt` and
`decision_models.txt`. The generated table, for every repo in both lists, is
`archive-vetting-static.md` beside this file. This file records the judgement
calls the script cannot make.

Checklist per GEN-002: (1) integrity, (2) behavioural red-team, (3) licence and
provenance. The script automates (1) and (3). Part (2) needs a GPU session.

## Result of the first pass

43 repos: **20 PASS, 19 REVIEW, 4 GONE, 0 FAIL.** No repo has a file flagged by
the Hugging Face scanner.

## Decision-model candidates

| Repo | Static | Commit | Size | Notes and next step |
|---|---|---|---|---|
| `convaiinnovations/laya-typed-decisions` | PASS | e929ae5cf69b | 0.85 GB | **In use** (oasis-decider shadow). Safetensors only. Loads through the `laya` PyPI package, which was read before pinning (0.4.0, wheel sha256 `c004d3f7…`): no network, subprocess, eval/exec, or pickle in the inference path. Checkpoint ships an out-of-range temperature (`choice:11+` = 0.10, clamped to 0.5 by Laya); confidence is uncalibrated. Publisher is an HF **user** account, not a verified org. Archive: YES. |
| `convaiinnovations/laya` | REVIEW | 7b928d828b7b | 2.37 GB | Three `.py` files in the repo (`rl_agent_api.py`, `rl_common.py`, `email_utils.py`). `laya.load` does not execute them; they are sample code. Fine-tune start point for phase 1. Archive: YES, after reading the three files. |
| `convaiinnovations/laya-multilingual` | PASS | 1720e3e3357c | 0.68 GB | mmBERT-base 322M, 1024 context. Archive: YES. |
| `StrandsAgents/strands-decider-2B-hobson-v21` | PASS | 2b52a6235c1b | 0.09 GB | A **LoRA adapter (67 MB) + decision head (4 MB)** on `Qwen/Qwen3.5-2B-Base`, which must be archived with it. Ships `MANIFEST.sha256`, training data hashes, and its training config: the best provenance record in this set. JevBench 0.762 (unverified). The org is not HF-verified; the Amazon link rests on press reports. Archive: YES, with the base model. |
| `Mapika/decider-2b`, `-4b`, `-35b-a3b` | REVIEW | — | 3.8 / 8.4 / 69 GB | Each repo ships its own `decider/` engine code. Most-downloaded community deciders. Read the engine before any use; do not run it inside the fleet. Archive: 2b only, after the read. |
| `morriszjm/Tacit-4B`, `Tacit-9B` | REVIEW | — | 8.1 / 18.9 GB | `tacit_decision.py` in the repo (the AnyJev loader). 9B scan not complete. Archive: defer until AnyJev reaches 1.0. |
| `perplexity-ai/pplx-decider-v1-27b` | REVIEW | 5117a6c7fe73 | 52 GB | Verified org. Flag is the training and inference source shipped in the repo (`inference.py`, `source/src/autojev/…`), which is a plus for provenance. Session-tier model (does not fit beside the Gemma pair). Archive: YES. |
| `autotrust/JEV-9B`, `JEV-27B` | REVIEW | — | 18 / 55 GB | **Uses the Jev name; AutoTrust has no link to TypeSafe**, which publishes no weights. Same publisher as the Gemma-4-31B distill that is now GONE (below). Archive: NO for now. |

## GONE upstream (2026-10-07, HTTP 401)

| Repo | On our box? |
|---|---|
| `autotrust/gemma4-31B-Fable-5-Distilled` | Yes (archived 2026-08) |
| `autotrust/gemma4-31B-Fable-5-Distilled-GGUF` | Yes |
| `dataset:Glint-Research/Fable-5-traces` | Yes |
| `dataset:greghavens/fable-5-coding-and-debugging-traces` | Yes |

Our archive copies are now the only copies. Keep them. Their provenance can no
longer be re-checked against the source, so the GEN-002 licence question for
serving the 31B distill to anyone other than the fleet stays open, and the
takedown is itself evidence for that question.

## Other REVIEW items in the archive (no action needed to keep them)

- `moonshotai/Kimi-K3`: `auto_map` remote-code hook and model code in the repo.
  Serve through vLLM's native support; never `trust_remote_code=True` outside a
  sealed session.
- `deepseek-ai/DeepSeek-V4-Flash-0731`: reference inference code in the repo.
  Weights are safetensors.
- `AgentNativeResearchLab/arc-agi3-*` datasets: solver scripts ship with the
  traces. Data only to us; never execute them.
- `armand0e/Fable-5-Chat`: no licence tag. Training use is uncertain.

## Next steps

1. Read the three `.py` files in `convaiinnovations/laya` and the Mapika
   `decider/` engine.
2. Add the YES rows to `models.txt` and run one archive transfer (about
   60 GB in total, dominated by pplx-decider-27B). This needs `make up` in
   `ai_research/ops/model_archive/terraform` with a lease, and the reaper
   (ADM-057) deployed first.
3. Part 2 (behavioural red-team) for any decider that leaves shadow mode.
