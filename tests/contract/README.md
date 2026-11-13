# X backend contract fixtures (T12)

`tests/test_decider_contract.py` uses these files. Do not edit them by hand.

| File | What it is | How to refresh it |
|---|---|---|
| `decider_server.py` | A byte copy of `oasis-claw/sandbox/decider/decider_server.py` (oasis-claw commit 3e4a6b1, 2026-10-07). The SHA-256 is pinned in the test. | Copy the file again from oasis-claw. Then update `VENDORED_SHA256` and re-record. |
| `oasis_decider_recorded.json` | Requests from the gateway adapter, and the answers of the real `oasis-decider:local` image (laya 0.4.0, Laya revision e929ae5). It includes the pre-fix request, which the real image refuses with HTTP 400, and two `reviewer.toolcall.v1` question-set calls (real `choice`, `noul` and `score` answers in one call). | `. .venv/bin/activate && arch -arm64 python scripts/record_decider_contract.py` (needs Docker and the image; the container has no network). |

When the live oasis-claw checkout is on the machine, the contract tests also run
against the live `decider_server.py`. Set `OASIS_DECIDER_SERVER=/path/to/decider_server.py`
to name another file. Without a checkout, that half of the tests is skipped.

The same applies to the reviewer's questions: a test checks that `reviewer.toolcall.v1`
is verbatim `TOOL_CALL_QUESTIONS` in `oasis-claw/extensions/oasis-reviewer/src/decider.ts`.
Set `OASIS_REVIEWER_DECIDER_TS` to name another file. When that test fails, the reviewer's
questions changed: register `reviewer.toolcall.v2`; do not edit v1.

The recorded Laya answers are zero-shot and near chance (GENERATIVE_PLAN.md
section 10.5). The tests check only their shape, never their values.
