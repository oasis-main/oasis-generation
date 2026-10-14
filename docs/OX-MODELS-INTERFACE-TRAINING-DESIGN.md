# O/X models, a two-mode AI interface, and the training + data scheme

Status: DESIGN DRAFT v0, 2026-10-14. Author: Kolmogorov. Reviewer: Mike.
Scope: `oasis-generation` (gateway, catalog, adapters) plus the interface contract the
oasis-ai front end will call. Nothing here is built. Nothing here touches infra.

## 0. What I know and what I do not

Known from the repo and the board: the gateway (`/v1/chat/completions`, `/v1/models`),
the catalog with Bedrock and runner backends, the per-caller attribution ledger, GEN-001
(Scaleway vLLM spike), GEN-004 (wake/sleep), GEN-005 (Modal LoRA pipeline, JSONL in,
adapter + eval report out), `shared_modal` manifests, and the oasis-claw auto-reviewer
(Layer 1 deterministic, Layer 2 model judge that may only tighten Layer 1).

NOT known to me. The design treats these as inputs to fill in, not as facts:
- The model archive roster (which base models, licences, sizes). I could not list the
  Storage Box from this container (no `hcloud`, no Storage Box variables set).
- The state of the shadow fine-tunes for the auto-reviewer, and what Laya and AnyJev are.
  I found no spec for them anywhere in my reach. Section 7 names what I need.
- `oasis-x/.swarm/GENERATIVE_PLAN.md`, which is outside my reach.

Where a number appears below it is a proposed default, marked (P). None is a measurement.

## 1. Vocabulary

- **O model** ("System 2"): an LLM. Open output space (text), slow, expensive per call,
  strong at planning, explaining and enriching context. Its output is unconstrained, so it
  can be wrong in ways nobody enumerated.
- **X model** ("System 1"): a decision model. Closed, declared output space (a label set or
  a typed record), one forward pass or a few, low latency, cheap, with a calibrated
  probability over its labels and an explicit **abstain** option. The auto-reviewer's
  allow / escalate / deny verdict is the prototype. Laya and AnyJev are first applications.
- **Base roster**: the model archive. Every O and X model is a *performance fine-tune of an
  archive model*, never a bare third-party checkpoint. GEN-002 vetting applies to the base
  once; fine-tunes inherit its licence and provenance record.
- **Hosted frontier** (Bedrock Claude, GPT, GLM) stays available as O and as teacher and
  judge. It is not part of the owned roster and is never the only path (cost, licence).

The invariant that matters, in one line: **X holds authority, O holds ideas.** An O output
can propose. Only an X verdict (or a human) can authorise a side effect.

## 2. The two modes as control flow

Both modes run the same four primitives. They differ in who runs first and who can wake whom.

| Primitive | Meaning |
|---|---|
| `O.generate(ctx, prompt_id)` | open-ended generation under a named system prompt |
| `X.decide(ctx, labelset_id)` | distribution over a declared label set, plus abstain |
| `Gate(action)` | X verdict on a proposed side effect: allow / escalate / deny |
| `Surface(msg)` | emit text to the user |

### 2.1 Dialogue mode (System 2, O-led)

```
user turn -> O reasons in text -> O proposes tool call(s)
          -> Gate(X): allow | escalate(to human) | deny(+reason back to O)
          -> tool runs -> result appended to ctx -> O continues -> ... -> final reply
```
- Conclusions are reached in language, and actions follow from conclusions.
- X governs the action boundary. It can also act as structured responder wherever a tool or
  the user needs a fixed format (JSON that must validate).
- This is what the fleet already does. The design formalises the Gate so it can be an owned
  X model instead of a frontier-model judge, and so each verdict is a logged training
  example (section 5).

### 2.2 Submarine mode (System 1, X-led)

```
event / user turn -> X routes along a labelled process DAG (no text generated)
   node: X.decide(ctx, node.labelset) -> edge label -> next node
   SURFACE nodes: X decides whether and why to invoke O
        enrich:       O.generate(ctx, node.prompt_id) -> written to ctx, not shown
        communicate:  O.generate(ctx, node.prompt_id) -> Surface(text), shown
        assess/plan:  O.generate(...) -> typed plan -> new DAG edges or a goal
   every side effect still passes Gate(X)
   X abstains, or the chosen edge is low-confidence -> SURFACE with a "need help"
   prompt -> dialogue mode
```
- The user watches a labelled process, not a transcript: node names, edge labels, and the
  occasional spoken message when something is worth saying.
- O is called on purpose, with a preset system prompt chosen by the DAG node, over exactly
  the context that node needs. O cost becomes a function of the DAG, not of how chatty
  the model feels.
- Surfacing is an X decision with a price: each `O.generate` has a token budget, and each
  session has a cap.

### 2.3 Mode transitions

- submarine to dialogue: X abstains, confidence on the chosen edge falls below the node's
  floor, a Gate escalates, or the user speaks. User input always surfaces to dialogue.
- dialogue to submarine: the user (or O, with a Gate-approved proposal) promotes a worked
  procedure into a DAG. A transcript that succeeded three times (P) with the same tool
  sequence becomes a *candidate DAG*, reviewed before activation. This is how routines are
  learned.
- Both modes write the same trace format (section 5), so a session can switch mode without
  losing history and one dataset serves both.

### 2.4 Shared safety invariants

1. **Tighten-only.** Deterministic policy is Layer 1. An X Gate may turn allow into
   escalate or deny, never the reverse. This is the rule the auto-reviewer already obeys.
2. **O never self-authorises.** No code path lets an O output flip a Gate verdict.
3. **Fail closed on the Gate, fail open on the chat.** If X is down, side effects escalate
   to a human and plain conversation continues on O alone. The reviewer's fail-open setting
   for some bots is a fleet choice, not a recommendation for customer tenants.
4. **Abstain is a first-class output**, scored and rewarded, not a fallback.
5. **Do not train O on Gate approvals** (section 5.6).

## 3. AI interface contract (oasis-ai front end to oasis-generation)

The interface is one session object with a mode flag, not two products. The front end
renders what the trace says; it holds no logic.

### 3.1 Session and trace

```
Session { id, tenant, mode: "dialogue" | "submarine", dag_id?, budgets, policy_id }
TraceEvent (append-only, one JSON line each):
  { sid, seq, ts, kind, actor: "O"|"X"|"user"|"tool"|"human",
    model: "<public_id>@<adapter_version>", ctx_ref, out, meta }
  kind in: user_msg | o_text | o_proposal | x_decision | gate | tool_call | tool_result
           | surface | mode_switch | escalation | human_verdict
x_decision.out = { labelset_id, probs{label:p}, chosen, abstain:bool, abstain_floor }
gate.out       = { action_hash, verdict: allow|escalate|deny, layer1_rule?, reason }
```
Why one trace: it is the UI feed, the audit log, the billing meter (token + machine-time
per event), and the training data (section 5). Building four parallel records invites the
four to disagree.

### 3.2 Endpoints (proposed; the gateway keeps `/v1/chat/completions` unchanged)

| Route | Purpose |
|---|---|
| `POST /v1/sessions` | create session; body picks `mode`, `dag_id`, budgets |
| `POST /v1/sessions/{id}/events` | user input or tool result in; returns stream of TraceEvents (SSE) |
| `POST /v1/x/decide` | stateless X call: `{model, labelset_id, ctx}` returns the x_decision record |
| `POST /v1/gate` | stateless Gate call for external callers (e.g. oasis-claw reviewer) |
| `GET /v1/dags`, `PUT /v1/dags/{id}` | list, create, version process DAGs (draft/active) |
| `GET /v1/sessions/{id}/trace` | full trace, for replay and audit |
| `POST /v1/sessions/{id}/mode` | switch mode; always logged as a `mode_switch` event |

`/v1/x/decide` and `/v1/gate` are deliberately usable without a session. That lets the
oasis-claw auto-reviewer call an owned X model through the same gateway during shadow
(section 6) without adopting the session model.

### 3.3 DAG format (submarine mode)

```
DAG { id, version, status: draft|active|retired, entry, nodes[], edges[] }
Node { id, label, kind: "decide" | "tool" | "surface" | "end",
       labelset_id?,                      # decide nodes
       floor: 0..1,                       # min top-prob; below it -> escalate to dialogue
       surface?: { intent: enrich|communicate|assess|plan, prompt_id, ctx_slice, budget_tokens },
       tool?: { name, args_schema } }     # every tool node is Gate-checked at runtime
Edge { from, label, to }                  # label must be a member of from.labelset
```
Validation at `PUT` time (static, no model): every decide node's outgoing edge labels equal
its labelset exactly (+abstain); every path reaches `end` or an escalation; every tool node
has a Gate; surface budgets sum below the session cap. A DAG that fails cannot go `active`.

### 3.4 Rendering

Dialogue mode: transcript with Gate badges on proposed actions. Submarine mode: the DAG
with the current node lit, edge labels showing the winning probability, and surfaced
messages as bubbles anchored to the node that spoke. Same trace, two renderers.

### 3.5 Open product decision

`fleet/LANES.md` lane 4 is still waiting on Mike: chat, dashboard + Q&A, or email bot.
Dialogue mode is a chat; submarine mode is a dashboard. An email bot is a submarine DAG
whose surface nodes write email. So the contract above does not force the choice, but the
first build should pick ONE renderer. My default (P): dialogue-mode chat first, because it
needs no DAG authoring, and with the Gate logging on from day one so data accrues.

## 4. Serving

### 4.1 Roster shape

| Role | Size class | Where it runs | Why |
|---|---|---|---|
| X models (decision) | small, 0.5B to 4B (P), encoder or small decoder with a classification head or constrained decoding | CPU or shared small GPU, always warm | latency budget in tens of ms; called many times per session |
| O models (LLM) | tier S/M from the catalog (Gemma-4-12B class and up) | GEN-001 L40S, stop/start (GEN-004) | seconds per call; can tolerate wake latency |
| Frontier O | Bedrock via the gateway | hosted | teacher, judge, fallback, and the quality ceiling |

X is cheap enough to stay warm, which matters: submarine mode calls X on every node, and
a wake-on-request delay on every edge would destroy the mode.

### 4.2 Gateway changes

1. Catalog gains `kind: "o" | "x"` and, for X, a `labelsets` map (id to label list,
   abstain floor default). `/v1/models` exposes both.
2. X backend: a runner that returns the full label distribution (logprobs over the
   constrained label tokens, or head softmax), never a free-text label. The gateway
   computes `chosen` and `abstain` itself from the declared floor, so a model cannot argue
   its way past its own threshold.
3. Adapter routing: `model: "gemma-4-12b-agentic@lora-2026-11-02"`; vLLM `--enable-lora`
   hot-load (GEN-005). Unknown adapter version returns 404, never silently the base.
4. Per-token caps and usage events (open in GEN-003) become a prerequisite, not a
   follow-up, once X is called per DAG node: spend per session must be bounded.
5. Cache: X decisions are deterministic given (model@version, labelset, ctx hash); cache
   them keyed on that triple. Do NOT cache Gate verdicts across tenants.

### 4.3 What this costs, roughly (P, to be measured in GEN-001)

An X call at 1 to 4B on CPU: order 10 to 100 ms. A 20-node submarine run: 0.2 to 2 s of
X time plus O only at surface nodes. That is the performance case for the mode, stated as
a hypothesis for the spike to test, not a result.

## 5. Training and data management

### 5.1 One trace, many datasets

Every TraceEvent (section 3.1) is a candidate training row. Nothing is trained on raw
traces. A trace becomes training data only through a **dataset build**: a versioned,
declarative query over traces plus a labelling rule, producing JSONL with a manifest.

```
DatasetManifest {
  dataset_id, version, built_at, git_sha_of_builder,
  source: { trace_query, tenant_scope, date_range, n_rows_in },
  filters: [ ...each with rows_dropped ],
  labelling: { kind: human|teacher_model|rule|outcome, who/which model@version, agreement_rate? },
  splits: { train, dev, test, shadow_holdout } by SESSION (never by row),
  provenance: { consent_class, licence_notes, pii_pass: {tool, version, residue_rate} },
  sha256 of each shard, storage_uri
}
```
Splitting by session is not optional. Rows in one session are strongly dependent, and a
row-level split leaks the future into the past and inflates every eval.

### 5.2 Label sources, ranked by how much I trust them

1. **Human verdicts** (`human_verdict` events: Mike or a tenant admin resolving an
   escalation). Rare, expensive, and the ground truth for Gate models.
2. **Outcomes**: the action ran and something measurable happened (test passed, email
   bounced, user undid it within N minutes). Delayed and noisy, but not self-referential.
3. **Teacher model** (frontier O or the current Sonnet-class judge). Cheap and broad, and it
   inherits the teacher's blind spots. Always carry `teacher@version` in the manifest, and
   always keep a human-labelled audit slice (P: 5%) to measure teacher-versus-human
   agreement per label.
4. **Rules** (Layer 1 outcomes). Exact, but they teach X only what the rule already says.
   Useful as a regression set: the X model must never allow what Layer 1 denies.

An X model trained only on rows labelled by its own earlier verdicts is a closed loop.
The build must refuse any dataset whose labels descend from the model being trained
without an independent source in the audit slice.

### 5.3 What each model class trains on

| Model | Objective | Data | Never |
|---|---|---|---|
| X Gate | allow/escalate/deny + calibrated probs + abstain | human verdicts, outcomes, teacher labels, Layer 1 regression set | its own approvals as sole labels |
| X router (DAG decide nodes) | edge label | successful dialogue traces promoted to DAGs, human-edited DAGs | traces from failed sessions as positives |
| O (LLM) | SFT / preference on assistant text and tool proposals | accepted dialogue turns, human-edited replies | Gate approvals as a reward (5.6) |

Method for the X models: LoRA/QLoRA first (GEN-005 already scopes it), with
temperature scaling on a held-out dev split so the reported probabilities are calibrated.
Calibration matters more than accuracy for X, because the abstain floor is a threshold on
the probability and the threshold is meaningless if the probability is not honest.

### 5.4 Evaluation gate: what a model must clear before it serves anything

Every candidate adapter gets an **eval report** (GEN-005 output) with these numbers, each
with a bootstrap 80% CI over sessions:

- Accuracy and macro-F1 against the human-verdict test split.
- Cost-weighted error: a false allow on a destructive action is not the same as a false
  escalate. The cost matrix is part of the labelset and is set by a human, not learned.
- Calibration: ECE and reliability curve; abstain-rate and coverage-versus-risk curve.
- Layer 1 regression: zero allowed actions that Layer 1 denies, on the full regression set.
  One violation fails the candidate outright.
- Slice metrics by tool, tenant class, and mode, so an average cannot hide a bad slice.
- Teacher agreement, and teacher-versus-human agreement, so label noise is visible.
- Latency p50/p95 on the target hardware (X: tens of ms budget, P).

Promotion rule (P): a candidate replaces the active model only if it beats the incumbent
on cost-weighted error with the CI excluding zero, passes every hard check above, and does
not regress any slice by more than a pre-registered margin. Pre-register the margin and
the comparison before looking at the test split. A model that cannot clear this runs in
shadow only (section 6).

### 5.5 Data management

- **Stores.** Manifests and small shards: `s3://oasis-kolmogorov-data-451873237418/oasis-generation/...`
  (versioned, 30-day recovery). Weights, adapters and large archives: Storage Box
  over WebDAV. Git holds code, manifests and eval reports only. Never weights, never rows.
- **Immutable versions.** A dataset version is never edited; a fix is a new version with
  a changelog. Every adapter's eval report names the dataset versions it trained and
  evaluated on, and the base archive model hash.
- **Lineage.** `adapter@ver -> {base_model_hash, dataset versions, trainer git sha,
  hyperparameters, Modal manifest id}`. The `shared_modal` run manifest is the receipt;
  the adapter registry links to it.
- **Tenant boundary.** Default is that a tenant's traces train only that tenant's adapters.
  Pooling across tenants needs an explicit, recorded consent class. NuSci interview data
  and anything from the product trees stays out of any public artefact (AGENTS rule 6).
- **PII.** A scrub pass runs at dataset build, not at trace write, so raw traces stay
  auditable for the retention window and training data does not. Report residue rate; do
  not claim zero.
- **Retention and deletion.** Raw traces: a retention window (P: 90 days), then
  aggregate-only. A deletion request removes the rows from future builds and flags the
  adapters trained on them for retrain at the next scheduled cycle.
- **Contamination.** Shadow-holdout sessions are frozen before any training and never
  re-enter a train split, including through teacher relabelling.

### 5.6 Why O must not be rewarded by the Gate

If O is optimised against Gate approvals, it learns to write proposals the Gate approves,
which is not the same as proposals that are good. That is the verifier-gap failure we
already measured on reward models, applied to our own Gate. Rule: the Gate is a
constraint on O's actions, never a training signal for O's text. O trains on human
acceptance and outcomes.

The reverse leak matters just as much: if X is trained on O's explanations of why an
action is fine, then O's persuasiveness becomes an attack surface on X. X's inputs are the
action, the context and the policy, not O's argument for it, unless a human has labelled
that argument's effect.

## 6. Shadow-first rollout (matches how the auto-reviewer is being evaluated now)

1. **Collect.** Gate logging on in dialogue mode from day one; no model change.
2. **Shadow.** The candidate X scores every action the incumbent judge scores. Its
   verdict is logged, never enforced. Compare on the shadow-holdout and on live traffic.
3. **Disagreement review.** Humans label a sample of the disagreements, oversampled where
   the candidate would have allowed what the incumbent denied. These labels return to 5.2.
4. **Tighten-only enforcement.** First enforcement step: the candidate may escalate or
   deny on top of the incumbent, never allow over it. This is safe by construction.
5. **Authority transfer.** Only after the 5.4 gate passes on live shadow data does the
   candidate become the primary Gate, with the incumbent kept as a sampled auditor.
6. **Drift watch.** Monthly (P): re-score the frozen holdout and a fresh labelled slice;
   alert on calibration drift before accuracy drift.

## 7. What I need to make this concrete

1. The archive roster: model ids, sizes, licences, and which are already fine-tuned.
2. What Laya and AnyJev are (task, label set, current data, current score). I found no
   spec in my reach. Mike mentioned them as first applications.
3. Where the reviewer's shadow decisions are stored today, and in what schema, so
   the TraceEvent mapping can be checked against real rows.
4. Mike's call on the lane-4 first surface (3.5).
5. The cost matrix for Gate errors per tool family. Only a human should set it.
6. A decision on tenant pooling (5.5): single-tenant only for v0, or an opt-in pool.

## 8. Work items proposed (to be numbered by the board owner)

| Item | Needs infra? | Depends on |
|---|---|---|
| Trace + manifest schemas as pydantic models, with tests | no | this doc |
| Catalog `kind`/`labelsets` + `/v1/x/decide` + `/v1/gate` against a stub X | no | schemas |
| DAG validator (static, section 3.3) with tests | no | schemas |
| Dataset builder + split-by-session + manifest, run on existing fleet traces | no | access to reviewer logs |
| Eval harness: cost-weighted error, ECE, Layer 1 regression, bootstrap CI | no | builder |
| GEN-001 spike: serve an X candidate and an O candidate, measure latency/cost | YES (Yes Man) | schemas |
| GEN-005: LoRA run on Modal for one X Gate candidate | YES (Yes Man, Modal) | builder, eval harness |

## 9. Hypotheses this design commits to (so they can fail)

- H1. An X Gate at 1 to 4B, trained on human verdicts plus teacher labels, matches the
  Sonnet-class judge on cost-weighted error at under 1/50 of the per-call cost (P).
- H2. Submarine mode cuts O tokens per completed routine by at least half versus dialogue
  mode on the same routines, with no drop in task success.
- H3. Calibrated abstain at a 0.7 floor (P) catches more than half of the X errors that
  matter, at under a 20% abstain rate.
- Any of these can be false. The spike and the shadow run exist to find out.
