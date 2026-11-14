# Usage telemetry and concurrency history

Lumbridge Computer records operational model usage without recording model input or
output. The runtime remains the source of truth: Lumbridge Computer samples each running
model's local Prometheus endpoint and makes counter resets durable across model
and Scene restarts.

## Runtime support

Engine knowledge lives in one module, `src/engines/`. Telemetry, liveness and
(later) scheduling all go through it:

- `engines::detect(&Model) -> Engine` picks `Vllm`, `Sglang`, `LlamaCpp`,
  `NemoSpeech` or `Generic` from, in order: an optional `serve.engine` field in
  the registry (`vllm`, `sglang`, `llama-cpp`, `nemo-speech`, `generic`), then
  `serve.kind`, then the launch argv (`vllm serve`, `vllm.entrypoints`,
  `sglang serve`, `sglang.launch_server`, `llama-server`, `nemo-speech`, or a
  `docker run` image named after vLLM/SGLang). Registries without the field
  parse unchanged. Set `engine:` when a wrapper script hides the engine
  (for example `vision_launch.py` or `transcribe/serve.sh`).
- `engines::parse_load(engine, text) -> Option<EngineLoad>` normalises
  `/metrics` text to `running`, `waiting`, `kv_usage` (0..1),
  `prompt_tokens_total` and `generation_tokens_total`. `Generic` sniffs the
  metric names.
- `engines::scrape(&Model)` does a loopback-only GET of the model's own port
  (`supervision.liveness.metrics_path`, else `/metrics`) with a 2 s cap.

| Field | vLLM | SGLang | llama.cpp (`--metrics`) | NeMo speech |
|---|---|---|---|---|
| running | `vllm:num_requests_running` | `sglang:num_running_reqs` | `llamacpp:requests_processing` | `speech_requests_in_flight` + `speech_stream_active` |
| waiting | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | `llamacpp:requests_deferred` | `speech_requests_waiting` |
| kv_usage | `vllm:kv_cache_usage_perc`, older `vllm:gpu_cache_usage_perc` | `sglang:token_usage` | `llamacpp:kv_cache_usage_ratio` | none |
| prompt tokens | `vllm:prompt_tokens_total` | `sglang:prompt_tokens_total` | `llamacpp:prompt_tokens_total` | none |
| generation tokens | `vllm:generation_tokens_total` | `sglang:generation_tokens_total` | `llamacpp:tokens_predicted_total` | none |

Counts and token totals are summed across label sets (vLLM `engine="0"`,
`engine="1"`, several `model_name`s, vllm-omni `replica`/`stage`). SGLang
series that carry a non-empty `priority` label are skipped, because they
repeat the total. `kv_usage` takes the fullest cache, because summing ratios
means nothing. NaN and Inf samples count as absent.

**Unknown means `None`, never idle.** If the running gauge is absent, renamed,
NaN, or the endpoint cannot be read, there is no load reading, and nothing is
recorded as an idle sample. A missing waiting gauge next to a present running
gauge reads as 0 waiting. Liveness has always used that rule. A NeMo ASR
server that exports `speech_stream_active` but no HTTP in-flight gauge has an
unknown load until it adds `speech_requests_in_flight`. A server with no
`/metrics` endpoint has no load reading at all.

Counters are mapped onto the existing durable schema (SGLang names).
vLLM: completed requests (including finish reason), prompt, output and cached
tokens, and latency sums and counts. llama.cpp: prompt and output tokens.
A `runtime` label keeps counter series distinct across engines. Aborted
completions are also recorded as aborted requests. The first scrape sets a
baseline. Missing historical SQLite samples are not rebuilt from lifetime
counters.

Fields an engine does not export stay absent. vLLM has no equivalent of
SGLang's instantaneous generation throughput, cache-hit-rate gauge, bounded
caller labels, or vision encoder counters. Unlabelled requests are reported as
unknown, not credited to an invented client. KV utilisation is not a cache-hit
rate. Prometheus keeps its own history while the collector is down.

Test fixtures in `src/testdata/` say on their first line which engine and
version they stand for (vLLM generate, vLLM pooling embed/rerank, vllm-omni
TTS, NeMo ASR, SGLang, llama.cpp, pre-v1 vLLM).

## What gets scraped

By default, the collector scrapes every model that meets all of these:

- it is in the active Scene (else the desired Scene, else the whole registry);
- its engine is known (not `Generic`);
- it is serving on its port.

All brain variants share port 8001, so the collector keeps one model per port.
When several models could own a port, it picks the one whose health marker
answers.

`--metrics-url URL` (on `agent` and `usage collect`) still scrapes only that
endpoint, as before. The engine comes from the registry model on that port, or
is sniffed from the metric names. `--telemetry-timeout-ms` (agent) and
`--timeout-ms` (collect) set the scrape timeout. The default is 2000 ms, which
is also the cap.

Scrape failures are logged when the state changes:

- once when a target starts failing;
- once when it recovers;
- at most once an hour while it keeps failing;
- once when it stops serving.

The old collector logged every 60th failure: "scrape failed (1320
consecutive)".

## SGLang launch contract

The managed SGLang recipe enables:

```text
--enable-metrics
--enable-cache-report
--tokenizer-metrics-allowed-custom-labels client agent workload
```

Callers may attach a bounded label dictionary in SGLang's `x-custom-labels`
header, for example:

```text
x-custom-labels: {"client":"cloud-agent","agent":"research-1","workload":"interactive"}
```

Vision callers use `"workload":"vision"`. Lumbridge Computer prefers native encoder
counters when the serving build exports them and otherwise uses this label for
vision call share and vision prompt-token volume. The report includes
`vision_request_source` so this fallback is never ambiguous.

`client`, `agent`, and `workload` must be low-cardinality stable categories.
Never put a request id, conversation id, user id, file name, URL, or job id in a
Prometheus label. Job ids belong in the future Jobs ledger.

## What is stored

The database is `<root>/.compute/usage/telemetry.sqlite3` in WAL mode. It contains:

- time-sampled running and queued requests per model, configured concurrency,
  generation throughput, cache hit rate, and (schema v2) KV-cache usage, the
  engine that read the sample, and the serving port;
- reset-safe request, prompt-token, generation-token, cached-token, and abort
  counters;
- vision request share and vision prompt-token volume, plus multimodal image
  items, encoder tokens, and image-cache hits when the serving build exports
  those native counters;
- the active Scene, model id, and a fingerprint of the launch configuration; and
- the three bounded caller labels above.

Schema v2 only adds columns: it adds the nullable columns
`engine_samples.kv_usage`, `engine` and `port` with `ALTER TABLE` and leaves
existing rows alone (NULL in the new columns). A v1 binary can still read a v2
database.

The top-level `concurrency` in reports is the busiest serving port in the
window. On a typical box that is the brain on :8001, across brain swaps, so idle
embed/rerank samples do not bury the four-lane distribution in C0.
`concurrency_scope` says which port (`port 8001`, or `model <id>` for rows
written before v2). `models[]` has each model's own distribution and peak KV
usage. Totals still sum every model.

It never stores prompts, input token ids, model output, image/audio/video data,
image URLs, arbitrary headers, or sampling parameters. Counter snapshots are
written once per minute. Scheduler gauges are sampled every five seconds by
default, which makes the C0-C4 distribution a time-weighted approximation rather
than a count of request admissions.

## Commands

```bash
lumbridge-computer usage collect                      # every running model
lumbridge-computer usage collect --metrics-url http://127.0.0.1:8001/metrics
lumbridge-computer usage summary --since 24h
lumbridge-computer usage agents --since 7d
lumbridge-computer usage concurrency --since 30d

# Every report also has machine-readable output.
lumbridge-computer usage summary --since 7d --json
```

The resident agent collects automatically. Use `--no-telemetry` only when the
runtime cannot expose local metrics; failures degrade telemetry and do not stop
the gateway, watchdog, or model supervisor.

The loopback control API exposes the same read-only records:

```text
GET /v1/usage?since=24h
GET /v1/usage/agents?since=7d
GET /v1/usage/concurrency?since=30d
```

## Four-lane scheduling policy

Telemetry is the gate for Jobs scheduling, not a reason to keep all lanes busy
unconditionally. The measured DFlash2 curve on the reference GB10 node is
31.54, 57.09, 81.43, and 85.96 aggregate token/s at C1-C4. C4 adds only 5.6%
over C3.

The planned scheduler policy (not built yet) therefore is:

1. three fair-share background lanes;
2. one latency-reserve lane for interactive work;
3. background borrowing of lane four only while queueing and TTFT remain healthy;
4. weighted round-robin across agents, with job ids in a durable ledger rather
   than metric labels; and
5. one simultaneous vision-prefill job until measured encoder queueing supports
   relaxing the limit.

Lumbridge Computer should own admission and job lifecycle. Any outside benchmark
tool may read aggregate artifacts, but never raw operational traffic or request content.
