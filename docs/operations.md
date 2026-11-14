# Operations

How to run Lumbridge Computer on a box: the root, Scene state, switching,
the resident agent and its supervision rules, installing and upgrading the
agent, and the status snapshot. For the box-level config (`lumbridge-computer.toml`,
`doctor`, `apply`) see [box.md](box.md).

## The root

Every command works on one **root**: a directory holding `registry/models.yaml`,
`scenes/*.scene.yaml` and, once something has run, the state directory `.compute/`.
The root is `--root <dir>`, else `$LUMBRIDGE_COMPUTER_ROOT`, else the current
directory. Keep your box's registry and scenes in their own repository; this
repo's `registry/` and `scenes/` are a reference stack to copy from.

## Persistent Scene state

Lumbridge Computer stores state at `<root>/.compute/state.yaml`. State writes use
an atomic temporary-file replacement. The persisted lifecycle fields are:

- `desired_scene`: the Scene the node should resume after an agent or node restart;
- `active_scene`: the Scene whose exact model health was last verified;
- `last_known_good_scene`: the previous proven Scene used for recovery;
- `transition_scene`: an in-progress transaction, cleared on success or failure;
- `last_error`: the most recent activation or watchdog failure.

Process ownership is bound to `(boot_id, pid, process_start_ticks)`. Lumbridge Computer will
never signal a PID-only record (one without that identity) or a PID that Linux has reused.

## Adopting models that are already running

If the models of a Scene are already running but the state has no process
identities for them (a fresh state file, or one written by hand), bind them
first. Adoption neither starts nor stops a model.

```bash
lumbridge-computer scene adopt everything
lumbridge-computer status
lumbridge-computer scene activate everything --dry-run
```

Adoption fails if a Scene model is unhealthy, its health marker identifies the
wrong model, another registered model is serving, a process record is missing,
or its process group does not equal its PID.

## Transactional switching and recovery

```bash
lumbridge-computer scene activate everything --dry-run
lumbridge-computer scene activate everything
lumbridge-computer scene resume
```

Activation validates the complete Scene and all process ownership before any
stop. A shared port occupied by the wrong model is a hard failure. Every started
model must report its exact health marker. If a transition fails after mutation,
Lumbridge Computer stops the partial target and restores the previously active Scene.

`scene resume` tries the persisted desired Scene. If it cannot become exactly
healthy, it tries the previous last-known-good Scene.

## Resident agent and stable gateway

The resident agent (`lumbridge-computer agent`) resumes desired state, serves the model gateway,
enforces the unified-memory floor, and reconciles only registry models that
explicitly opt into `supervision`. Models without that block are never restarted
by the steady-state agent.

```bash
lumbridge-computer agent --listen 127.0.0.1:8011 --upstream 127.0.0.1:8001 --floor 3
```

Apps use `http://<box>:8011` and pick a model with the OpenAI `model`
field; the gateway routes each request to the running model that serves that
name and streams responses (SSE included) through untouched. `--upstream` is
where requests without a `model` go. `--gateway-mode tcp` turns the gateway into a
protocol-transparent byte pump to `--upstream`. See [gateway.md](gateway.md).

The same agent samples the local runtime's Prometheus endpoint into a durable,
privacy-safe SQLite history. SGLang models must opt in with `--enable-metrics`;
telemetry failure never takes down the gateway, watchdog, or model supervisor.
See [telemetry.md](telemetry.md) for the stored fields and label contract.
It also samples each model's measured memory (cgroup scope or smaps, plus
per-process NVML) every 30 s; `lumbridge-computer footprints` compares that with the
declared `footprint_gb`. See [memory.md](memory.md).

An opted-in model is restarted according to the rules below. Recovery takes the
same transition lock as Scene activation, refuses to signal an unowned or stale
process identity, repeats both memory admission checks, and uses exponential
backoff after a failed attempt. This is intentionally per-model: one failed voice
runtime does not stop or roll back the healthy brain, ASR, or embedding processes
in the same Scene.

### Supervision rules

The governing principle: **a busy engine is never restarted.** The rules live in
`src/health.rs`.

Each due check (every `check_interval_sec`) applies, in order:

1. **Process exit is fatal.** If the identity-owned process is gone (no `/proc`
   entry, a reused pid, or a zombie), the model restarts at once, without
   waiting for `failure_threshold`. Backoff still applies. A record
   without process identity is never treated as exited.
2. **Stall.** Work in flight, meaning engine `running + waiting` or the
   gateway's in-flight count (whichever is larger), while the token counters
   stay flat for `liveness.stall_sec` counts as a failed check. This applies
   even while readiness is slow.
3. **Readiness.** `health` + `health_contains` within
   `supervision.probe_timeout_ms`. A refused connection or a missing marker
   counts as a failed check. A **timeout while work is in flight is busy**: it
   neither counts nor resets the failure streak.
4. **Chat probe** (`liveness.chat_probe`). Sent only when idle by every source
   with counters unchanged since the last check. The first failure is
   "suspect". Load and in-flight are re-read right after a failed probe, and
   work that arrived during the probe voids it. A second consecutive idle
   failure confirms the hang and restarts the model.

Rules 2 and 3 restart after `failure_threshold` consecutive failed checks. Before
signalling anything, the restart re-checks its trigger under the lock: readiness
for a readiness failure (and stands down if busy), progress for a stall, and
idle plus one more probe for a probe failure.

**Startup.** A (re)started model is given up on only when its process exits, or
when nothing moved for `startup_timeout_sec`. Progress means the port opening,
the model log growing, or the metrics endpoint answering, and each sign of
progress restarts the window. A hard ceiling of 4× the window stops a runtime
that logs forever without ever becoming ready from holding the agent. Activation
uses the same wait, with the model's `startup_timeout_sec` if it is supervised,
else 900 s.

**Logs.** One line per non-healthy due check with its evidence, for example
`busy: running=2 waiting=1 gateway=0; readiness took over 5000 ms; not counting it`.
The same kind of line repeats at most every 5 minutes, with a count of the
lines it suppressed. When the model recovers, one "healthy again" line is
printed. On respawn the previous model log is kept as `.compute/logs/<id>.log.1`.

Supervision fields (all optional except `restart`):

| field | default | meaning |
|---|---|---|
| `restart` | — | `always` |
| `check_interval_sec` | 30 | seconds between due checks (min 1) |
| `failure_threshold` | 3 | consecutive failed checks before a restart (rules 2–3) |
| `backoff_sec` / `max_backoff_sec` | 60 / 900 | delay after a failed restart, doubling |
| `startup_timeout_sec` | 180 | startup **no-progress** window (ceiling 4×) |
| `probe_timeout_ms` | 5000 | readiness probe and metrics budget, clamped to 500–60000 |
| `liveness.stall_sec` | 300 | stall window (min 60) |
| `liveness.chat_probe.interval_sec` | 120 | least time between probes (min 30) |
| `liveness.chat_probe.timeout_sec` | 30 | one probe's budget, clamped to 5–120 |

The gateway provides the per-model in-flight counts (`gateway::InFlight`). Run
without the agent's gateway, supervision relies on engine metrics alone.

## Installing the agent

Install the unit only after a Scene is active (or adopted) and `scene resume`
is a no-op on that box. The resident agent is the only unit.

For an unprivileged install, use the user unit. It expects the checkout and
root at `~/lumbridge-computer`; edit both paths if yours differ.

```bash
loginctl enable-linger "$USER"
install -Dm0644 systemd/lumbridge-computer-agent-user.service \
  ~/.config/systemd/user/lumbridge-computer-agent.service
systemctl --user daemon-reload
systemctl --user enable --now lumbridge-computer-agent.service
```

A system unit is in `systemd/lumbridge-computer-agent.service`; set `User=` and
the paths first:

```bash
sudo install -m 0644 systemd/lumbridge-computer-agent.service \
  /etc/systemd/system/lumbridge-computer-agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now lumbridge-computer-agent.service
```

The unit sets `LUMBRIDGE_COMPUTER_ROOT` and the agent takes its root from
there, so the root is declared in exactly one place. The units listen on
`127.0.0.1:8011`, because the gateway has no authentication. To let other
machines on your private network use it, change that to `0.0.0.0:8011` (or the
box's private address) and keep it off the internet.

Verify model identity through both the backend and the gateway before pointing
any client at the box:

```bash
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8011/v1/models
lumbridge-computer usage collect
lumbridge-computer usage summary --since 24h
systemctl --user is-active lumbridge-computer-agent.service
```

## Upgrading the agent binary

`KillMode=process` means model process groups outlive an agent restart, and the
agent revalidates every `(boot_id, pid, start_time_ticks)` identity on resume.
A binary upgrade therefore does not disturb a healthy Scene:

```bash
git pull && cargo build --locked --release
systemctl --user restart lumbridge-computer-agent.service
```

Confirm the model PIDs are unchanged afterwards. If the agent cannot revalidate
an identity it refuses to signal that process rather than guessing:
investigate before forcing anything. To roll back, check out the previous
commit, rebuild, and restart the same unit; the model processes are untouched
either way.

Building on the box itself competes with the models for CPU and memory, and
can make health probes time out. Build with fewer jobs (`cargo build -j 2`) or
on another machine of the same architecture.

After a registry or Scene change, restart the agent once while you watch it:
two registry models on the same port need distinct `health_contains` markers,
or the agent cannot tell which one is serving.

## Status snapshot

`lumbridge-computer snapshot` prints one compact JSON line (keys sorted, `schema_version: 1`)
that is safe to publish to a status page or database: a status publisher can run it every
30 s and post the line as is.

```bash
lumbridge-computer --root ~/lumbridge-computer snapshot
```

- `models`: the active Scene's models in Scene order, else the default Scene's
  (`everything`). `desired_scene` is reported but never chooses the models.
- `health`: `ready` if the readiness probe (port and `health_contains` marker) passes,
  otherwise `unavailable`. Probes run in parallel, capped at 3 s each, so the command
  finishes in seconds. No metrics scrape, no `nvidia-smi`.
- `declared_allocation_gb`: the sum of those models' `footprint_gb`; `scene_budget_gb`:
  the Scene's `budget_gb`, or `null`; `global_budget_gb`: the compiled default (100).
- `error_code`: `null`, or `runtime_error` when the state records a `last_error` or the
  snapshot could not load something. It never carries the error text.

It is sanitized: nothing from a model's `serve` or `health` sections, no paths, PIDs,
environment, hostnames or error text. Model names are registry display labels and pass
through verbatim, trimmed to 200 characters.

It always exits 0. When the registry does not parse, the active Scene is missing, a Scene
names an unknown model, or the state file does not parse, it prints
`error_code: runtime_error` with whatever models could be resolved (none if the registry
failed). `runtime_error` is the only non-null code.
