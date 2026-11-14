# Evaluations (`lumbridge/v1`)

Lumbridge Computer evaluates the model configuration that is actually serving:
weights, quantization, context, runtime, parsers, and speculative decoder. A
model name without its serving configuration is not a reproducible benchmark
target.

```bash
lumbridge-computer eval ls
lumbridge-computer eval run smoke
lumbridge-computer eval run performance --model brain --repeat 5
lumbridge-computer eval run finance-core --base-url http://box1.example.com:8011/v1
```

`--base-url` defaults to `http://127.0.0.1:8001/v1` and `--model` to `brain`.
Point `--base-url` at the gateway (`:8011`) to evaluate whichever model serves
that name in the active Scene.

Suites live in `evals/*.eval.yaml` under the root. Each case has a stable id,
prompt, category, generation limit, and deterministic assertions. Runs produce
append-only JSON in `eval-results/` with raw outputs and per-sample metrics.

The shipped suites:

| Suite | What it checks |
|---|---|
| `smoke` | The endpoint answers and follows short instructions. |
| `performance` | TTFT and decode speed on fixed prompts. |
| `finance-core` | Basic arithmetic and market-mechanics questions. |
| `voice-agent` | Answers that read well when spoken (no markdown, spelled-out numbers). |

## Metrics

- **TTFT**: wall time until the first streamed content or reasoning token.
- **Prefill tok/s (approximate)**: API-reported prompt tokens divided by TTFT.
  This is client-observed and includes queueing and scheduling; server-native
  prefill metrics are a separate source (see [telemetry.md](telemetry.md)).
- **Decode tok/s**: completion tokens divided by time after the first token.
- **Score**: share of samples satisfying every declared assertion.

Performance runs should include warmups and record the hardware, the Scene,
the runtime version, the model revision, and cold or warm cache state. The
result files are local and portable JSON.

## Scope

The native suites are post-activation health and performance checks for the
exact configuration serving on a box. They are not a training or research
harness. Eval suites are shareable data: like Scenes, they never contain shell
commands (see [SECURITY.md](../SECURITY.md)).
