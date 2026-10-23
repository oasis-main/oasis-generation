# Browser stream contract v0 (draft, for the /interface UI; ORG-171)

Hello World builds the UI against a mock of this. Change it by PR, not by chat.

Endpoint: `POST https://ai.<env>.cloud.oasis-x.io/v1/chat/stream` (oasis-ai host in front of the gateway).
Auth: `Authorization: Bearer <capability token>` (contract v1). CORS: only the /interface origin.
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
| `done` | `finish_reason` (`stop` `length` `tool_calls` `error`) | terminal |

Exactly one of `done` or `error` ends a stream. A stream cut without either is a failure; the UI retries with the same `client_request_id` (the gateway does not bill the same id twice).

Errors before the stream opens are plain JSON `{"error": {"code","message"}}` with HTTP status:

| status | code | UI |
|---|---|---|
| 401 | `token_expired` / `invalid_token` | refresh token via oasis-auth, retry once |
| 402 | `payment_required` | free tier or no card: show "add a card". Body also has `tier` and `billing_state`. No inference was done. |
| 403 | `account_suspended` or `missing_scope` | show the message |
| 429 | `spend_cap_reached` / `rate_limited` | show cap and reset date from `Retry-After` |
| 503 | `metering_unavailable` | the gateway refuses unmetered inference; retry later |

`GET /v1/models` (scope `gen:models`) works on the free tier and returns the public model list.
Mock: serve the table above from a static file; the mock must emit one stream per event type and one pre-stream response per status.
