# Lumbridge Computer Scene Spec (`lumbridge/v1`)

This is the public contract. Everything else — the Governor, the CLI, the
scheduler — can be refactored freely. This format cannot, once people publish
scenes against it. So it is deliberately small.

## The core idea: scenes reference ids, not weights

A **scene** is a manifest listing **model ids**. A **model registry** resolves each
id to actual weights + a launch command. The registry churns as models evolve (new
quants, new backends, bigger context); the scene stays stable.

```
  scene (stable, shareable)          registry (local, evolves)
  ─────────────────────────          ──────────────────────────────
  models: [ears, brain, voice]  ──▶  brain → ~/models/Qwen3.6-35B-A3B-NVFP4-Fast
                                            → vllm serve, gmu 0.55, flashinfer...
```

Keep the id `brain` and a stable `served_name` while the weights underneath swap
(say a dense 27B for a 35B-A3B MoE), and downstream agents never notice. The spec formalizes that as the mechanism that
lets scenes "update over time as models evolve" without breaking anyone.

### Why this also solves scene-sharing security

A published scene contains **only ids and parameters — never shell commands.** The
launch commands live in your *local, vetted* registry. So activating a downloaded
scene can only ever start models your own registry already trusts. If a scene
references an id you don't have, Lumbridge Computer asks you to add it to your registry, showing
the launch command for review — an explicit opt-in, not silent remote code
execution. Declarative-by-construction; there is no field in which a scene can smuggle
a command.

---

## Scene manifest

`scenes/everything.scene.yaml` is the reference scene.

```yaml
apiVersion: lumbridge/v1
kind: Scene
metadata:
  name: everything
  version: 1
  description: "Reference stack: a chat brain, speech in and out, and retrieval."
  author: example
  tags: [brain, speech, rag]
models:
- brain
- asr
- tts
- embed
- rerank
budget_gb: 100
activation:
  order: listed
  wait_healthy: true
```

### Fields

| Field | Required | Meaning |
|---|---|---|
| `apiVersion` | ✔ | `lumbridge/v1`. |
| `kind` | ✔ | `Scene`. |
| `metadata.name` | ✔ | Unique scene name; the CLI handle. |
| `metadata.version` | ✔ | Integer, bumped on any change. Reproducibility. |
| `metadata.description` | ✔ | One line, shown in `lumbridge-computer scene ls`. |
| `metadata.tags` | – | For the (future) registry search. |
| `models` | ✔ | Ordered list of model ids resolved via the registry. |
| `budget_gb` | – | Per-scene budget override; defaults to the global Governor budget. |
| `activation.order` | – | `footprint-asc` (default) \| `listed`. |
| `activation.wait_healthy` | – | Default `true`. Block until health checks pass. |

A scene **never** contains: weight paths, shell commands, or GPU flags. Those live in
the registry. This is load-bearing for both stability and security.

---

## Model registry

`registry/models.yaml` — local to each box, evolves freely. Ids are the stable
contract; everything under `serve` can change.

```yaml
apiVersion: lumbridge/v1
kind: Registry
version: 1

models:
  brain:
    name: "Qwen3.6 35B-A3B MoE (NVFP4)"
    footprint_gb: 66              # worst-case unified memory once serving (weights + KV + encoder)
    channel: stable               # stable | latest — how aggressively to track new weights
    health: "http://localhost:8001/v1/models"
    serve:
      kind: vllm
      port: 8001
      weights: "~/models/Qwen3.6-35B-A3B-NVFP4-Fast"
      served_name:                # stable aliases so downstream clients survive a weight swap
        - brain
        - "qwen3.6-35b-a3b"
      args:
        max-model-len: 65536
        kv-cache-dtype: fp8
        gpu-memory-utilization: 0.55
        enforce-eager: true
        moe-backend: flashinfer_b12x      # Unsloth DGX Spark recipe; critical for speed
        limit-mm-per-prompt: '{"image": 0, "video": 0}'
      env:
        CUTE_DSL_ARCH: sm_121a

  image:
    name: "FLUX.2-dev (FP8)"
    footprint_gb: 32
    channel: stable
    health: "http://localhost:8007/health"
    serve:
      kind: diffusers            # not vllm — a separate runtime (ComfyUI/diffusers)
      port: 8007
      weights: "~/models/FLUX.2-dev"
      args: { dtype: fp8 }

  # ears / voice / music elaborated the same way (ASR, Chatterbox, ACE-Step).
```

### `extends:` and macros: writing a variant as a delta

Running the same recipe twice (two sets of weights, two ports) used
to mean copying ~60 lines. A registry model may instead inherit from another:

```yaml
  brain-alt:
    extends: brain
    name: "Alternative brain weights (NVFP4)"
    health_contains: "\"root\":\"/home/sam/models/brain-alt-nvfp4\""
    serve:
      port: 8002
      weights: ~/models/brain-alt-nvfp4
      served_name: [brain-alt]
```

The full working pair is `examples/two-brains.registry.yaml` with
`examples/two-brains.scene.yaml`.

Rules:

- **Deep merge, child wins.** Mappings (`serve`, `serve.args`, `serve.env`,
  `supervision`, `supervision.liveness`, …) merge key by key; any other value
  in the child replaces the inherited one.
- **Sequences are replaced, never concatenated.** `served_name`, `command`
  and any list in `args` are taken whole from the child if it sets them. Above,
  brain-alt does *not* answer to `brain`.
- **`null` deletes.** `key: null` in the child removes the inherited key, e.g.
  `args: {enforce-eager: null}` drops a flag, `supervision: {liveness: null}`
  turns liveness off. Deleting a required field (`serve: null`) is a load error.
- **Chains** (`a` extends `b` extends `c`) are allowed. A cycle or an unknown
  base fails the whole registry load, naming the models involved.
- Resolution happens in `Registry::load`. Every other part of Lumbridge Computer only
  ever sees fully resolved models; `extends` is not a field of the result.
  A base is an ordinary model (it must itself be complete and is runnable).

After the merge, **macros** expand in every string value of each model:

| Macro | Value |
|---|---|
| `${MODEL_ID}` | the model's registry id (`brain-alt`) |
| `${PORT}` | the model's own `serve.port` (error if it has none) |

So `health: http://localhost:${PORT}/v1/models` on the base gives the child
:8002 with no override. Only `${NAME}` with `NAME` matching `[A-Z_][A-Z0-9_]*`
is a macro: `$HOME`, `${lower}`, `$$` and JSON such as
`'{"method":"mtp","num_speculative_tokens":3}'` are left alone. An unknown
macro (`${PROT}`) is a load error, not left in place; write `$${NAME}` for a
literal `${NAME}` (for example in a shell `command`). `~/` paths keep expanding
at launch exactly as before. Inside a YAML flow list, quote a macro:
`[brain, "${MODEL_ID}"]` (in flow syntax `{` starts a mapping).

A model that uses neither `extends` nor a macro is parsed exactly as before.
Resolved models are re-parsed from YAML text, so text coercions still apply
(`MAX_JOBS: 4` in `env` reads as `"4"`).

To check what an entry resolves to, and the argv it would launch with:

```sh
lumbridge-computer model show brain-alt          # YAML
lumbridge-computer model show brain-alt --json
```

Two models in one scene may not share a `serve.port`; `scene activate` (and
`--dry-run`) refuses such a scene. Models that share a port across scenes,
like `brain` and `brain-small` on :8001, still swap normally.

### Registry-only supervision

Long-lived runtimes may opt into resident-agent recovery in the local registry:

```yaml
  voice:
    health: "http://localhost:8095/health"
    health_contains: '"model_loaded":true'
    supervision:
      restart: always
      check_interval_sec: 30
      failure_threshold: 3
      backoff_sec: 60
      max_backoff_sec: 900
      startup_timeout_sec: 180    # no-progress window while starting
      probe_timeout_ms: 5000      # readiness probe budget (default 5000)
```

The block is absent by default, preserving the original one-shot lifecycle for
every existing model. It belongs to the vetted local registry—not a shareable
Scene—because a downloaded Scene may select known model ids but may not create a
new process-restart policy. Recovery applies only while that model is in the
persisted desired Scene. The agent requires consecutive exact-health failures,
takes the Scene transition lock, signals only an identity-owned process group,
re-runs declared and observed-memory admission, and backs off failed attempts.
The full rule set (process exit, busy, stall, chat probe, startup progress) is
in [operations.md](operations.md#supervision-rules).

#### Liveness: catching a frozen engine

Readiness (`health` + `health_contains`) proves the HTTP server answers, not that
the engine behind it is moving. A vLLM engine can freeze with requests running
while `/v1/models` still passes, and readiness alone never restarts it. A supervised vLLM model can opt into a progress check:

```yaml
    supervision:
      restart: always
      check_interval_sec: 15
      failure_threshold: 2
      liveness:
        kind: vllm-metrics       # the only kind today
        stall_sec: 300           # default 300, never below 60
        metrics_path: /metrics   # default; always on the model's own port
```

A due check then also fails when `vllm:num_requests_running +
vllm:num_requests_waiting > 0` and `vllm:prompt_tokens_total +
vllm:generation_tokens_total` has not changed for `stall_sec`. Idle models
(nothing in flight) are never stalled, and an unreadable `/metrics` is
"unknown", not a failure. vLLM counts a prompt's tokens only when the request
emits its first token, so a long prefill is silent on both counters; `stall_sec`
must exceed the slowest prefill the model can be asked for (a cold 128K-token
prompt on a GB10 box takes roughly one to two minutes). A stall still has to repeat
`failure_threshold` times, so the policy above restarts about
`stall_sec + check_interval_sec` after the last counted token. The restart
re-reads the counters under the transition lock and stands down if they moved.
Requests the gateway holds for the model count as in flight too, so an engine
that stops reporting while clients wait is still a stall.

An optional `chat_probe` catches the opposite case, an engine that looks idle
but hangs every request:

```yaml
      liveness:
        kind: vllm-metrics
        stall_sec: 300
        chat_probe:
          interval_sec: 120    # least time between probes (never below 30)
          timeout_sec: 30      # per probe, clamped to 5..120
```

The probe is a 1-token chat sent only when every source says idle and the
counters have not moved since the last check. One failure is only suspicious.
A second consecutive failure confirms the hang and restarts the model
(`failure_threshold` does not stack on top of it). Load is re-read right after a
failed probe, and work that arrived meanwhile voids that failure.

`health_contains` is strongly recommended for supervised HTTP runtimes. Without
it, any process accepting TCP on the port satisfies health, including an app that
is listening while its model failed to load.

### `channel`: how scenes track evolving weights

- `stable` — pin the exact `weights` path. Reproducible; you update deliberately.
- `latest` — the registry may resolve to a newer quant of the same model family
  (e.g. a fresh NVFP4 build) on activation. Bleeding edge; use for your own box, not
  for scenes you publish for others.

The scene picks the *id*; the registry's `channel` decides how much the weights are
allowed to drift underneath it. That's the whole "scenes evolve as models evolve"
story, made explicit and controllable.

---

## Governor interaction

On `activate`, Lumbridge Computer computes the diff between the running set and the target
scene's `models`, then:

1. **Stops** running models not in the scene (frees their footprint first).
2. **Starts** the scene's models in `activation.order`, each passing **admission
   control** against `budget_gb` before launch.
3. Waits for health if `wait_healthy`.

The watchdog runs throughout, unchanged — the safety net if any `footprint_gb` is
wrong. A scene can never talk the Governor into over-committing; admission control is
not bypassable by a scene.
