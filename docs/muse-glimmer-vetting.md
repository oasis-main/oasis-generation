# Muse-Glimmer-30B — vetting log (GEN-002 method)

Candidate: `meta-models/Muse-Glimmer-30B` at revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, plus its DFlash drafter
`meta-models/Muse-Glimmer-30B-assistant` at `e8192f3a8f61`.

Scope decided 2026-09-28 (Mike): **archive on the Hetzner Storage Box only.
Keep it off the serving box.** The model is not in `catalog.py` and stays out
of it until part 2 below is done.

The checklist follows `GEN-002-vetting.md`: (1) integrity, (2) behavioral
red-team, (3) license and provenance.

Note on the name: the only published size is **30B** (29.6B parameters,
including a ~1.8B vision encoder). No 31B repository exists under
`meta-models`.

## 1. Integrity — PASS

| Check | Result |
|---|---|
| Weight format | `safetensors` only (2 shards: 49.95 GB + 9.60 GB). No `.bin`, `.pt`, or pickle files. |
| Executable code in the repo | None. No `.py` files. |
| Remote-code hooks | None. No `auto_map`, `trust_remote_code`, or `custom_pipelines` in `config.json`, `tokenizer_config.json`, `processor_config.json`, or `generation_config.json`. |
| Hugging Face security scanner | `scansDone: true`, `filesWithIssues: []` on the main repo, the drafter repo, and the official GGUF repo. |
| Chat template (`chat_template.jinja`, 9,992 bytes) | No URLs, no encoded blobs, no hidden instructions. Only a default system line ("You are a helpful AI assistant.") and a tool-calling format. |
| Revision pin | Record `a4e59da52a7b` in the archive manifest. The transfer script writes the resolved revision and per-file sha256 into `MANIFEST-*.json`. |

## 2. Behavioral red-team — NOT RUN (deferred, not skipped)

This part was not run on 2026-09-28, and archiving does not depend on it.

Reason: the only local path is Docker Model Runner, which loads the model
into **host** memory, not into the Docker VM. The official Q4_K_M build is
16.8 GB. The host had about 12.7 GiB free while the seven-bot fleet was
running, and the Mac disk was 97% full (29 GiB free). The July out-of-memory
incident happened with a 12 GB model under the same conditions.

Required before any serving or catalog use:

- Run the GEN-002 probe set (capability, exfil-in-code, artifact scan,
  refusal delta, prompt injection, exfil-via-tool-injection) at temperature 0.
- The exfil-via-tool-injection probe matters most. The card positions this
  model for autonomous agentic work, which is the tool-driving use case.
- Run on a rented GPU, not the Mac. One Scaleway L40S-1-48G session
  (about €1.46/h, 60-minute minimum) holds the BF16 weights with room for
  the drafter. Request it through Yes Man (CLAW-116).
- Test the archived BF16 weights, not a community quant.

## 3. License and provenance — PASS

- **Publisher:** `meta-models`, a Hugging Face **verified** organization named
  "Meta Inc." (428 members, 4 models). It is not the historical `meta-llama`
  or `facebook` organization, so the verification badge is the evidence that
  matters here. Checked 2026-09-28.
- **License:** the `LICENSE` file is the full Apache License 2.0 text. The card
  states that all released artifacts are Apache 2.0.
- **Usage policy:** `USAGE_POLICY.md` carries a Meta prohibited-use list
  (illegal activity, malware, unlicensed professional practice, circumventing
  safety measures, and others) and states the model is not for users under 18.
  Apache 2.0 does not itself bind us to this policy, but our terms of service
  should carry the same flow-down if the model is ever served to customers.
- **Provenance:** the card says it is distilled from Meta's Muse Spark, and
  trained on public data, third-party data, and data from Meta's products,
  with a January 4, 2026 knowledge cutoff. This is first-party distillation,
  which is cleaner than the GEN-002 community distills trained on another
  vendor's outputs.
- **Safety work declared by the publisher:** safety SFT and safety RL, with
  agentic risk (irreversible-action confirmation, indirect prompt-injection
  resistance) as one of four evaluated axes. This is a publisher claim, not
  our result. Part 2 exists to check it.

## Verdict (2026-09-28)

- **Archive: CLEARED.** Integrity and provenance pass. Archive the BF16 weights
  and the DFlash drafter.
- **Fleet or customer serving: BLOCKED** on part 2. Black-box probes cannot find
  weight-level backdoors; the verified publisher and the hash pin are the main
  defense for that risk.
