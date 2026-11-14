# Sleep and on-demand models

A model that isn't needed right now should cost (almost) no memory and come
back in seconds instead of a 4–6 minute cold start. Lumbridge Computer does that with
per-model **states**, per-Scene **modes**, vLLM **sleep**, and admission that
counts a sleeping model at what it really holds.

Code: `src/states/` (states, `Lifecycle`, admission, idle sleeper),
`src/engines/sleep.rs` (vLLM adapter), `assets/vllm/lumbridge_computer_sleep_ext.py`
(vLLM worker extension).

## States

```text
stopped → starting → ready ⇄ draining → sleeping → waking → ready
                       └──── draining → stopped
```

| State | Meaning |
|---|---|
| `stopped` | No live process Lumbridge Computer owns. |
| `starting` | Cold start in progress (spawned, waiting for exact health). |
| `ready` | Serving. |
| `draining` | About to sleep or stop: waiting for in-flight requests to reach 0. |
| `sleeping` | Engine asleep (vLLM level 2); the process is kept. |
| `waking` | Wake calls sent, waiting for readiness. A failed wake ends in `stopped`, never `ready`. |

The state is stored on each process record in `.compute/state.yaml`
(`state: {kind: sleeping, level: 2}`, `state_since_ms`). A record without a
`state` key, which is every process a Scene transition starts, means `ready`.
A record without changes is written back byte-for-byte, so an older binary can
still read the file.

## Scene modes

```yaml
models: [brain, brain-alt, embed]
modes:                       # optional; a model not listed is resident
  brain: {mode: sleepable, idle_sec: 0}
  brain-alt: {mode: on_demand, idle_sec: 600, stop_after_sec: 1800}
```

| Mode | Started with the Scene | Idle | Supervised |
|---|---|---|---|
| `resident` (default) | yes | never slept | yes |
| `sleepable` | yes | slept after `idle_sec` without requests | while ready |
| `on_demand` | no: first request starts it | slept after `idle_sec`, stopped after `stop_after_sec` asleep | while ready |

- `idle_sec` defaults to 600 and is at least 30. `idle_sec: 0` means the model
  is never slept for being idle, only to make room for another model.
- `stop_after_sec` (on_demand only) defaults to 1800.
- Only an engine that can sleep may be `sleepable`: a vLLM model with
  `serve.sleep` in the registry. Any engine may be `on_demand`. An `on_demand`
  model that can't sleep is stopped when idle and cold-started on demand.
- Scenes without `modes:` are unchanged; every model is resident.
- Modes live in the Scene, not the registry: they say *when* a model runs, never
  *how*, so a shared Scene can't use them to run anything new.

## Registry: `serve.sleep`

```yaml
brain:
  serve:
    kind: vllm                 # the builder is required (no `command`)
    sleep:
      level: 2                 # the only accepted value (see below)
      # Optional; defaults shown.
      # footprint_gb: max(6, 0.3 x footprint_gb)   # sleeping footprint for admission
      # wake_timeout_sec: 60     # all wake calls plus readiness; also the gateway's hold
      # sleep_timeout_sec: 120   # POST /sleep
      # drain_timeout_sec: 30    # in-flight -> 0 before sleeping/stopping
      # extension_dir: <root>/assets/vllm, else next to the binary
      # allow_public_dev_mode: false
      # wake:   [{path: "/wake_up?tags=weights"}]
      # reload: [{path: /collective_rpc, body: '{"method":"reload_all_fast"}'},
      #          {path: "/wake_up?tags=kv_cache"}]
```

Every default comes from measurements on a GB10 box with vLLM 0.27.1. `serve.sleep` is the only switch. It changes the launch
(`lumbridge-computer model show <id>` prints the argv):

| Added | Why |
|---|---|
| `--enable-sleep-mode` (unless `args` has it) | vLLM's cumem allocator, so weights and KV can be released |
| `--worker-extension-cls lumbridge_computer_sleep_ext.LumbridgeComputerSleepExt` | the reload step of the wake (below) |
| `VLLM_SERVER_DEV_MODE=1` (unless `serve.env` sets it) | exposes `/sleep`, `/wake_up`, `/collective_rpc`, `/is_sleeping` |
| `PYTHONPATH=<extension_dir>:<serve.env's or the inherited PYTHONPATH>` | so the EngineCore can import the extension |
| `--host 127.0.0.1`, replacing any `host` in `args` | dev mode serves `/collective_rpc` (any worker method) to whoever reaches the port |

Load errors: `level` other than 2; `serve.sleep` on a non-vLLM engine; with a
hand-written `command` (Lumbridge Computer could neither add the extension nor force the
bind); `worker-extension-cls` in `args`. A missing extension file is a launch
error naming the directories searched.

**Why only level 2.** On GB10, level 1 backs the weights up into pinned "CPU"
memory, which is the same unified pool: it frees only the KV cache, keeps the
weights' size in host RSS forever after, and on a hybrid Qwen3.5 model it
failed with HTTP 500 (`AssertionError: Memory usage increased after sleeping`)
and **killed the engine**. Level 2 drops weights and KV (7.6 of 8.9 GiB on the
test model) in about 1.3 s.

**The worker extension** is `assets/vllm/lumbridge_computer_sleep_ext.py`. After level 2
the weights' memory comes back with undefined contents, so a bare `/wake_up`
serves garbage (`"!!!!!!…"`). vLLM's stock `reload_weights` RPC restores the
target in 42–52 s on GB10 (a device copy from a file-backed mmap runs at
0.18 GB/s) and **silently skips the MTP drafter**: acceptance falls from 0.85 to
0, decode is 3–4× slower, nothing errors. `reload_all_fast` clones each tensor
before the copy (6.7 GB/s) and reloads the drafter's `mtp.*` tensors too:
1.05–1.45 s, byte-identical output, identical acceptance.

### Sleep

Drain (in-flight → 0), then `POST /sleep?level=2&mode=wait`, then confirm with
`GET /is_sleeping`. `mode=wait` lets a request that slipped in after the drain
finish; vLLM's default `abort` would cut it off with an HTTP 200 and a
truncated body. vLLM 0.27.1 reads `mode` (`entrypoints/serve/dev/sleep/api_router.py`);
an engine that answers 400/422 to it is asked again with plain
`/sleep?level=2`, since the drain already did the job. Measured: 1.3–1.9 s for
an 8 GB model.

### Wake

One deadline, `wake_timeout_sec` (default 60 s), for the calls and the
readiness wait together:

```
POST /wake_up?tags=weights                          # ~0.3 s; scheduler stays paused
POST /collective_rpc {"method":"reload_all_fast"}   # ~1.2 s for 8 GB; results[0] checked
POST /wake_up?tags=kv_cache                         # ~0.07 s; scheduler resumes
then: health check passes and GET /is_sleeping says false
```

vLLM resumes scheduling only once every tag is awake, so requests queued while
asleep never run on torn weights in this order. The RPC's reply is
`{"results":[{"target_secs":…,"has_draft":…,"draft_secs":…,"draft_tensors":…,"draft_loaded":…}]}`.
If the model has a speculative config (`speculative-config` in `serve.args`), the
wake **fails unless `draft_loaded` is `true`**.

**A failed wake stops the model.** Any failed call, a missing drafter, an
exit, or the deadline: Lumbridge Computer stops the process and records the model
`stopped`; the next request cold-starts it (on the gateway's long deadline).
A half-woken engine is never served, even one that still reports sleeping: a
failed reload can leave the weights torn, and the next request then kills the
EngineCore.

Measured on Qwen3.5-4B FP8 + MTP: whole wake 1.4–2.0 s, first request after it
at warm speed, five cycles without drift. Expected for a 22 GB NVFP4 brain:
5–15 s (copy ≈ 3.3 s at 6.7 GB/s plus unmeasured NVFP4/MoE post-processing),
against a ~6 min cold start.

### Security: loopback only

Dev mode makes `/collective_rpc` an arbitrary worker RPC, plus `/sleep`,
`/wake_up`, `/reset_prefix_cache` and the RLHF weight-update routes. So a model
with `serve.sleep` binds **127.0.0.1**, and the gateway (:8011) is the only
door: it answers every dev-mode route with 403 `engine_control_forbidden`
(`docs/gateway.md`, rule 0), and Lumbridge Computer's own engine-control calls refuse
non-loopback hosts. `allow_public_dev_mode: true`
keeps the recipe's bind; it exists for a box with its own firewall, and
`lumbridge-computer doctor` reports it as **WARN** (`models.dev_mode`). `doctor` **FAILs**
when any running model's process has `VLLM_SERVER_DEV_MODE=1` in its
environment and listens on its port anywhere but loopback (read from
`/proc/<pid>/environ` and `/proc/net/tcp{,6}`).

A loopback bind cuts off clients that call the engine's port directly from
another machine. Move them to the gateway (`http://<box>:8011`, with `model`)
before giving a model `serve.sleep`.

**Any other proxy in front of the engine is a second door.** If a local proxy
forwards every path to the engine and listens beyond loopback, dev-mode routes
such as `POST /collective_rpc` reach the engine through it. Such a proxy must
refuse the dev-mode routes (the list in `gateway::extract::ENGINE_CONTROL_PATHS`)
or listen on 127.0.0.1 only. `lumbridge-computer doctor` cannot see this,
because the engine itself is on loopback.

## Admission

- A sleeping model counts at its **sleeping footprint**: the peak sampled by the
  footprint sampler since its current sleep began, else
  `serve.sleep.footprint_gb`, else `max(6, 0.3 × footprint_gb)` (a 40 GB brain
  12 GB, small models 6), never above its full footprint. Not 3 GB: a sleeping
  vLLM keeps its whole host side plus ~1.3–3 GB of GPU (CUDA context, graph
  pool); a 4B model measured 6.4 GB asleep.
- A model that is starting, waking, draining or ready counts at its admission
  footprint (`footprint_gb`, or the measured one with `admission: measured`).
- Waking or starting re-admits the full footprint through the pool: the
  declared budget *and* observed `MemAvailable` (with the safety margin).
- If it doesn't fit, room is made from **other non-resident models of the
  desired Scene in the same pool**: first sleep ready ones (stop those whose
  engine can't sleep), then stop sleeping ones. Within each phase: highest
  `shed_priority` first (the watchdog's "shed me first" order), then least
  recently used (last finished request, else since it became ready). Never a
  resident model, never one with requests in flight (gateway or engine), never
  one mid-transition. Lumbridge Computer plans first and evicts only the shortest prefix
  that makes it fit; if nothing could, it evicts nothing and refuses.
- `status` and Scene activation use the same counts.

**Scene budget check.** With `modes:`, a Scene needs: residents in full + each
non-resident at its sleeping footprint (0 for an `on_demand` engine that can't
sleep) + room for the largest non-resident to be awake. It validates when the
residents fit with one non-resident awake; waking a second makes room by
sleeping the first. Without `modes:` this is the plain sum, as before.

## Supervision

A model that is sleeping, draining, waking or starting under the lifecycle is
**never probed as failed and never restarted**; neither is a non-resident model
that is simply stopped. Its supervision track is dropped, so liveness starts
fresh once it is ready again. The restart path re-checks this under the
transition lock. A ready non-resident model is supervised exactly like a
resident one.

## Idle sleeper

The agent runs it every 5 s on its own thread. For each `sleepable`/`on_demand`
model of the desired Scene that is ready, has nothing in flight (gateway count
and the engine's own running + waiting) and has been idle for `idle_sec`, it
calls `sleep`. It stops `on_demand` models that have been asleep for
`stop_after_sec`. The idle clock starts at the latest of: the gateway's last
finished request for the model (`InFlight::last_finished_ms`), the model
becoming ready, and the last time the sleeper saw it busy. A model started by a
Scene transition gets a full `idle_sec` from the first time the sleeper sees it.
A resident model found asleep (its Scene changed its mode) is woken.

## The `Lifecycle` contract

`states::Lifecycle` (implemented by `states::ModelLifecycle`):

- `state(id)`: an in-memory read (the gateway calls it on its async request
  path). The cache is updated on every transition this lifecycle makes and
  re-read from `state.yaml` at most once a second, which picks up the CLI's
  transitions and dead processes. A model without a live owned record reads
  `stopped`; `ensure_ready` on one that serves anyway (not started by Lumbridge Computer) returns
  at once without touching it.
- `ensure_ready(id, deadline)`: idempotent, single-flight per model (concurrent
  callers share one worker; it carries on past an impatient caller's deadline).
  Wakes a sleeping model or cold-starts a stopped non-resident one, making room
  as above. Never starts a stopped *resident* model: activation and supervision
  own those. The model must be in the desired Scene.
- `sleep(id, reason)`: drain (bounded by `drain_timeout_sec`; on timeout the
  model is left `ready`), then engine sleep, or stop for engines without sleep.
  Refuses resident models. The drain waits for *active* requests only
  (gateway `in_flight − waiting`, or the engine's own count): a request held
  for a wake never blocks a drain. The idle sleeper, by contrast, treats any
  in-flight request, held ones included, as busy, so it never starts sleeping a
  model somebody is waiting for.
- `ModelLifecycle::stop(id, reason)` (inherent, not in the trait): drain then stop.

In the agent one `Arc<ModelLifecycle>` is shared by the gateway (which holds
and wakes requests through it), the idle sleeper and supervision (which also
skips any model the lifecycle reports mid-transition); all read the same
`gateway::InFlight`.

All state changes happen under the Scene transition lock, which the lifecycle
waits for by polling; activation, supervised restarts and the watchdog only try
it and back off, so there's no deadlock. The lock is not held across a cold
start or a wake's readiness wait: the persisted `starting`/`waking` state keeps
everyone else away meanwhile.

The agent's gateway reports `last_finished_ms` next to `in_flight` on
`/lumbridge-computer/v1/inflight`, so `status` (a separate process) can show idle time.

## CLI

```sh
lumbridge-computer model sleep <id>    # drain, then sleep (or stop if the engine can't sleep)
lumbridge-computer model wake <id>     # wake a sleeping model, wait until ready
lumbridge-computer model start <id>    # start or wake, wait until ready (on_demand/sleepable)
lumbridge-computer model stop <id>     # drain, then stop
#   --gateway 127.0.0.1:8011  where the agent's gateway is (in-flight counts)
#   --timeout-sec 900         how long wake/start wait
lumbridge-computer status [--json]     # `states:` block / `models` array: mode, state, idle s, asleep GB
lumbridge-computer scene show <name>   # mode per model, and the sleep-aware budget check
```

The CLI uses the same `Lifecycle` as the agent; the transition lock serialises
them.

## Example: `two-brains-lite`

`examples/two-brains-lite.scene.yaml` runs the reference stack plus a second
brain, `brain-alt`, on demand; `examples/two-brains-lite.registry.yaml` is the
registry delta (`serve.sleep` on `brain`, inherited by `brain-alt`).

Both brains resident would not fit in 128 GB. So the stock brain is
`sleepable` with `idle_sec: 0`: it is never slept for being idle, only when a
`brain-alt` request needs its room and it has nothing in flight. The budget is
residents + both brains at their sleeping footprint + room for one brain awake,
plus the 8 GB margin. Once you have measured your brain asleep, set
`serve.sleep.footprint_gb` and lower the Scene budget.

Timings to expect, scaled from small-model measurements: sleep ~2 s after the
drain; wake 5–15 s (`wake_timeout_sec: 60`); cold start of a 35B brain ~4–6 min
(held on the gateway's 600 s deadline). Trade-off: while the stock brain is
busy, `brain-alt` can't wake (the gateway answers 503 + Retry-After), and a
`brain` request after a `brain-alt` session waits for the stock brain to wake.

## Example: one brain with sleep

`examples/brain-sleep.registry.yaml` is a single brain with `serve.sleep`, for a
first sleep/wake cycle on your own box. It binds the brain to 127.0.0.1:8001;
clients go through the gateway on :8011. Before you use it: move any client
that calls :8001 directly to the gateway, close other proxies' paths to the
dev-mode routes, install the extension, and mark the brain `sleepable` in the
Scene.

Check on your own model: the layerwise reload on its checkpoint format, that a
speculative drafter really reloads (the `draft_loaded` check catches a drafter
that loaded nothing, not one that loaded wrong: watch acceptance on the first
requests), the residual GPU memory while asleep, and sleep while another model
starts (Lumbridge Computer serialises its own sleeps and wakes under the
transition lock).
