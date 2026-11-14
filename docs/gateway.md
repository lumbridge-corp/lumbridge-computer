# Gateway

One address for every model on the box. Clients talk OpenAI to
`http://<box>:8011` and pick a model with the `model` field; the gateway sends
each request to the running model that serves that name.

```sh
lumbridge-computer agent   --listen 127.0.0.1:8011 --upstream 127.0.0.1:8001   # inside the resident agent
lumbridge-computer gateway --listen 127.0.0.1:8011 --upstream 127.0.0.1:8001 # foreground, same code
```

| Flag | Default | Meaning |
|---|---|---|
| `--gateway-mode http\|tcp` | `http` | `http`: route by `model` (this page). `tcp`: a plain byte pump to `--upstream`, no parsing. |
| `--upstream host:port` | `127.0.0.1:8001` | `http`: where requests **without** a `model` go (see below). `tcp`: the only upstream. |
| `--gateway-max-body-mib N` | `64` | Largest request body buffered to read `model`. |
| `--wake-timeout-sec N` | `600` | How long a request for a **stopped** model is held while it cold-starts (a 35B brain takes 4–6 min). A **sleeping** model is held for its own `serve.sleep.wake_timeout_sec` (default 60 s). On-demand only. |
| `--max-held-per-model N` | `32` | Requests held per model while it wakes; the next gets 503 `wake_queue_full`. On-demand only. |

## Route table

Rebuilt every 2 s from `registry/models.yaml` and `.compute/state.yaml`:

- The Scene is `desired_scene` (else `active_scene`).
- A model in that Scene is **running** when its `serve.port` accepts a TCP
  connection. (Models in one Scene never share a port, so a connect is enough,
  and unlike the `health` probe it adds nothing to the engines' access logs.)
- A running model is routable under its **registry id** and every
  **`served_name`**. Ids win over served names; if two running models advertise
  the same served name, the first in Scene order wins and the clash is logged.
- A Scene model that isn't answering is **not running** (503, retry). A
  registry model outside the Scene is **not in scene** (404 saying which Scene
  is active). Anything else is an unknown model (404).
- If the registry or state file fails to parse, the last good table is kept and
  the error is logged once.
- **On-demand (only when the gateway has a lifecycle):** a Scene model whose
  mode is `sleepable` or `on_demand` is routable even while its port is closed,
  and is listed in `/v1/models`. Resident models (the default) keep the rules
  above. See [Sleep and on-demand](#sleep-and-on-demand).

Table changes are logged as `gateway routes: brain->127.0.0.1:8001 embed->127.0.0.1:8012`.

## Finding the model

| Request | Where `model` is read |
|---|---|
| POST/PUT/PATCH with a `*json` content type | top-level `"model"` string. The body is buffered (up to the cap) and scanned with a two-field struct: other values are skipped, never built into a tree. |
| POST/PUT/PATCH `multipart/form-data` (e.g. `/v1/audio/transcriptions`) | the `model` form field (minimal RFC 7578 scan; file parts are skipped). |
| anything, as a fallback | `?model=` in the query string (percent-decoded). |

Other bodies (raw audio, form-urlencoded, ...) are not buffered; they stream to
the upstream picked from `?model=` or the default upstream.

The original request bytes are forwarded **untouched**. Hop-by-hop headers are
dropped and `Host` is set to the upstream, as any HTTP/1.1 proxy must.

## Routing rules

0. A vLLM dev-mode route (`/collective_rpc`, `/sleep`, `/wake_up`,
   `/is_sleeping`, `/pause`, `/resume`, `/reset_*_cache`, the weight-update
   routes; full list in `gateway::extract::ENGINE_CONTROL_PATHS`) → **403**
   `engine_control_forbidden`, with or without a `model`, however the path is
   spelled (percent-encoding, case, extra slashes). A sleepable engine runs in
   dev mode bound to 127.0.0.1 so the gateway is its only door; this keeps the
   door shut. (`--gateway-mode tcp` is a byte pump and cannot filter.)
1. `GET /health` → `{"status":"ok"}` (gateway liveness only).
   `GET /lumbridge-computer/v1/inflight` → `{"in_flight":{"brain":1,…},"last_finished_ms":{"brain":1762…}}` for loopback
   callers only (403 `loopback_only` otherwise); `lumbridge-computer status` reads it.
2. `GET /v1/models` → every routable name, OpenAI-shaped:
   `{"object":"list","data":[{"id":"qwen3.6-35b-a3b","object":"model","created":…,"owned_by":"lumbridge-computer","root":"brain"}]}`
   (`root` is the registry id).
3. `model` names a running model → forwarded to it.
4. `model` names a model in the Scene that isn't answering → **503** + `Retry-After`.
   With on-demand, a `sleepable`/`on_demand` model that isn't `Ready` is instead
   held and woken (below).
5. `model` names a registry model outside the Scene → **404** `model_not_in_scene`.
6. `model` names nothing in the registry → **404** `model_not_found`.
7. No `model` (any path, inference included, and bodies that aren't valid JSON)
   → `--upstream`, logged as `model=<default>`. This matches a lone vLLM serving
   its default model, so clients that omit `model` keep working behind the gateway.
   If that address is a running model's upstream, the request counts against
   that model's in-flight total and carries its `x-lumbridge-computer-model`.
8. No `model` and no default upstream (`--upstream ""`) → **400**
   (`invalid_json` if the body isn't a JSON object, else `missing_model`).

## Responses

- Never buffered. Each upstream frame goes to the client as it arrives (SSE and
  chunked included). `text/event-stream` responses get `X-Accel-Buffering: no`
  and, if absent, `Cache-Control: no-cache`, so a proxy in front doesn't buffer.
- `x-lumbridge-computer-model: <registry id>` on every routed response.
- Client disconnect (mid-stream or during a silent prefill) drops the upstream
  request; the engine sees its connection close and aborts the generation.
- An upstream that fails mid-stream aborts the client connection rather than
  ending the stream cleanly, so a truncated answer is never mistaken for a
  complete one.
- No request timeout: a long prefill is silent for minutes. Upstream connect
  times out after 5 s.

One log line per request, written when the response ends:

```
gateway POST /v1/chat/completions model=brain status=200 bytes=48213 ms=9120 streamed=true ok
```

Outcomes: `ok`, `client-abort` (client left mid-body), `client-gone` (left
before the response started), `upstream-error`.

## Error shapes

OpenAI's envelope, so SDKs print the message:

```json
{"error":{"message":"The model `gpt-9` does not exist on this Lumbridge Computer. Available: brain, embed",
          "type":"invalid_request_error","param":"model","code":"model_not_found",
          "available_models":["brain","embed"]}}
```

| Status | `code` | Extra |
|---|---|---|
| 400 | `invalid_json`, `missing_model` (only without a default upstream), `body_read_failed` | |
| 403 | `engine_control_forbidden` | a vLLM dev-mode route (rule 0) |
| 404 | `model_not_found` | `available_models` |
| 404 | `model_not_in_scene` | `model_id`, `scene`; no `Retry-After` |
| 413 | `body_too_large` | |
| 502 | `upstream_unavailable` | `type: server_error` |
| 503 | `model_not_running` | `model_id`, header `Retry-After: 10` |
| 503 | `model_unavailable` | wake/start failed (e.g. can't admit); `model_id`, `Retry-After: 10` |
| 503 | `wake_queue_full` | `--max-held-per-model` requests already waiting; `model_id`, `Retry-After: 10` |
| 504 | `wake_timeout` | not ready within the hold deadline (below); `model_id` |

## Registry fields

Optional, per model, registry only (host policy, never in a Scene):

```yaml
models:
  brain:
    gateway:
      upstream: 127.0.0.1:8041   # route via a local tracing proxy instead of serve.port
      rewrite_model: false       # true: replace `model` with this model's first served_name
    serve: { port: 8001, served_name: [brain, ...] }
```

- `upstream` must be `ip:port` (no DNS). Anything else is logged (once, with the route table) and ignored.
  Running-ness is still judged by `serve.port`.
- `rewrite_model` rewrites only the `model` value: the top-level JSON value, the
  multipart field, or the `?model=` parameter, and fixes `Content-Length`. Other
  JSON values are copied byte-for-byte as raw JSON, but the **top-level keys are
  re-emitted in sorted order** (JSON object order carries no meaning; no
  OpenAI-compatible engine depends on it). Bodies are untouched when the flag is off.

## Known limitations

- Running-ness is a TCP connect to `serve.port`; `health_contains` is not
  checked. During a Scene switch that reuses a port, a request can briefly reach
  the outgoing model's process (it answers or fails; the next refresh, within
  2 s, fixes the table).

## In-flight counters

`gateway::InFlight` (cheap clone, shared counters) with
`fn get(&self, model_id: &str) -> u32`, also exposed as
`trait InFlightSource`, plus `waiting_snapshot()` and `idle_for(id)`. A request counts from routing until its response body
ends or the client drops, streams included. Supervision reads it to tell a busy
engine from a wedged one.

## Sleep and on-demand

Active only when the gateway is given a `Lifecycle` (the model-states
implementation); without one, everything on this page above applies unchanged
and scene modes are ignored.

**Modes** come from the active Scene file, default `resident`. Either form:

```yaml
models: [brain, {id: brain-alt, mode: on_demand}]
modes: {embed: sleepable}
```

**When a request is held.** The named model is in the Scene, its mode is
`sleepable` or `on_demand`, and `Lifecycle::state` is not `Ready` (a sleeping
vLLM still accepts connections, so an open port proves nothing). The gateway
calls `ensure_ready(id, now + deadline)` on a blocking thread and forwards
the request once it returns.

**Two deadlines**, picked from the model's state when the request arrives:

| State | Deadline | Why |
|---|---|---|
| `sleeping`, `waking` | the model's `serve.sleep.wake_timeout_sec` (default 60 s) | a level-2 wake takes seconds (measured 1.4–2.0 s for an 8 GB model; 5–15 s expected for a 22 GB brain) |
| `stopped`, `starting`, `draining` | `--wake-timeout-sec` (default 600 s) | a cold start of a 35B brain takes 4–6 min |

The deadline covers admission too (making room by sleeping another model). A
wake that fails stops the model (never served half-woken, `docs/sleep.md`), so
the request gets 503 `model_unavailable` and the next one cold-starts it on the
long deadline. Concurrent requests each call it; the lifecycle
shares one wake between them. A client that leaves stops waiting; the wake
itself carries on.

**Counting.** A held request counts in `in_flight` from the start (so the idle
sleeper never sleeps a model someone is waiting for) and also in `waiting`
until the model is ready. Requests that don't name a model are never held.

**Non-streaming** requests wait silently: the upstream's response when ready,
**504** `wake_timeout` at the deadline, **503** `model_unavailable` if the wake
fails.

**Streaming** (`"stream": true`, POST to `/v1/chat/completions`,
`/v1/completions` or `/v1/responses`) get `200` + `text/event-stream` headers
immediately, then an SSE comment right away and every 5 s:

```
: lumbridge-computer waking brain-alt 0s

: lumbridge-computer waking brain-alt 5s

data: {"id":"chatcmpl-…","choices":[…]}        <- the upstream's own events, same body
```

Early headers are sent only on those paths, where an OpenAI-compatible engine
answers `stream: true` with SSE; any other streaming request waits like a
non-streaming one. Comments are ignored by every SSE client (the spec says so),
and OpenAI SDKs skip them. Since `200` is already sent, failures arrive inside
the stream as one OpenAI-style error event, after which the stream ends:

```
data: {"error":{"message":"The model `brain-alt` could not be woken: …","type":"server_error","param":"model","code":"model_unavailable","model_id":"brain-alt"}}
```

`code` is `model_unavailable` (wake failed), `wake_timeout` (deadline),
`upstream_unavailable` (connect failed after the wake), or the upstream's own
error when it answers with something other than an event stream (e.g. a 400
validation error; vLLM's flat `{"object":"error",…}` is wrapped in `error`).
An upstream that breaks after its events started aborts the connection, as
elsewhere.

`/lumbridge-computer/v1/inflight` adds `"waiting":{"brain-alt":1,…}` when on-demand is
on. A drain before sleep waits only for *active* requests (`in_flight −
waiting`), so a request held for a model never blocks that model's drain. Each wake logs `gateway wake <id>: ready after 12.3s` (or `failed …`).

The idle clock is `InFlight::last_finished_ms(id)` (Unix ms, also in the
inflight JSON); `InFlight::idle_for(id) -> Option<Duration>` is the same clock
as a duration: `Some(0)` while a request is in flight or held, time since the
last request finished otherwise, `None` if none has finished since the gateway
started. Modes come from the Scene's `modes:` map (`docs/sleep.md`); the agent
passes its one `states::ModelLifecycle` to the gateway.

## Rollback

`--gateway-mode tcp` turns the gateway into a byte pump: every connection is
copied to `--upstream`, with no parsing. Add it to the agent unit's `ExecStart`
and restart the agent.

If a model must keep going through a local proxy (for tracing, say), set
`gateway.upstream` on that model in the registry before switching to `http`.
