# Browser stream contract v0 (draft, for the /interface UI; ORG-171)

Hello World builds the UI against a mock of this. Change it by PR, not by chat.

Endpoint: `POST https://ai.<env>.cloud.oasis-x.io/v1/chat/stream` (oasis-ai host in front of the gateway).
Auth: `Authorization: Bearer <capability token>` (contract v1).
Same origin: the UI is served at `ai.<env>.cloud.oasis-x.io/interface` and calls the API on the same host, so the browser needs no CORS. The server sends no `Access-Control-Allow-*` headers and refuses a request whose `Origin` is present and is not that host (403 `cross_origin`).
Token storage: the UI keeps the token in memory only. Never `localStorage`, `sessionStorage` or a cookie readable by script. A page reload fetches a fresh token from oasis-auth.
Request JSON: `{"model": "<public id>", "messages": [{"role","content"}], "tools": [..optional..], "client_request_id": "<uuid>"}`.
Response: `200 text/event-stream`. Each frame is `event: <type>` + `data: <json>` + blank line. Every data object has `seq` (int from 0) and `request_id`.

| event | data (besides seq, request_id) | notes |
|---|---|---|
| `start` | `model`, `served_by` (`own`/`bedrock`/`byok`) | first frame |
| `text` | `delta` (string) | append to the current text block |
| `thinking` | `delta` (string), `redacted` (bool) | UI shows collapsed; `redacted=true` has empty delta |
| `tool_call` | `id`, `name`, `arguments` (JSON string, may arrive in several frames with `partial=true`) | |
| `tool_result` | `id`, `ok` (bool), `output` (string, capped) | server-side tool loop only |
| `usage` | `tokens_in`, `tokens_out`, `cost_estimate` | last frame before `done`; informational, not the bill |
| `error` | `code`, `message`, `retryable` (bool) | terminal. `code` in: `rate_limited`, `upstream_error`, `spend_cap_reached`, `token_expired`, `internal` |
| `done` | `finish_reason` (`stop` `length` `tool_calls`) | terminal. Failures use the `error` event, never `done`. |

Exactly one of `done` or `error` ends a stream. A stream cut without either is a failure.

Retries and billing. Every upstream call is billed, including a retry after a cut stream; a retry is a new call and costs money. `client_request_id` has one job: while a request with the same id (same `sub`) is still running, a second one is refused with 409 `request_in_flight`. Once the first has ended, the same id may be sent again and is billed as a new call.

Stop. When the browser closes the stream (abort, tab close), the gateway cancels the upstream call and bills the tokens used up to that point. The final `usage` frame may be missing; the bill comes from the gateway ledger, not from the stream.

Tools. A tool named in the request `tools` array is run by the browser (client tool): the stream ends with `done.finish_reason=tool_calls`, and the UI sends the result as a `role:"tool"` message in the next request. A tool the server owns (server tool, from the model's tool registry) is run by the server inside the same stream and reported with `tool_result`. A browser tool never produces `tool_result`; a server tool never ends the stream with `tool_calls`.

Errors before the stream opens are plain JSON `{"error": {"code","message"}}` with HTTP status:

| status | code | UI |
|---|---|---|
| 401 | `token_expired` / `invalid_token` | refresh token via oasis-auth, retry once |
| 402 | `payment_required` | free tier or no card: show "add a card". Body also has `tier` and `billing_state`. No inference was done. |
| 403 | `account_suspended`, `missing_scope` or `cross_origin` | show the message |
| 429 | `spend_cap_reached` / `rate_limited` | show cap and reset date from `Retry-After` |
| 409 | `request_in_flight` | the same `client_request_id` is still running; wait or abort the first |
| 503 | `metering_unavailable` | the gateway refuses unmetered inference; retry later |

`GET /v1/models` (scope `gen:models`) works on the free tier and returns the public model list.
Open item (needs one decision with Hello World): the UI proposes the Vercel AI SDK UI message stream. Choose one: (a) this contract, as is; or (b) the AI SDK protocol, with a thin adapter in oasis-ai that maps these events to it. Until chosen, Hello World builds against this file. The choice is made by changing this file by PR.

Mock: serve the table above from a static file; the mock must emit one stream per event type and one pre-stream response per status.
