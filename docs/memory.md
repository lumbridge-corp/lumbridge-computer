# Memory: pools, scopes, limits and measured footprints

The Governor admits models against their declared `footprint_gb`. On top of
that, Lumbridge Computer runs each model in its own cgroup, measures what it
really uses, and can learn admission figures from those measurements. This page
covers the memory pool, per-model scopes and limits, how memory is measured on
GB10, Docker models, learned footprints, the Supabase memory floor and swap.

"GB" means GiB throughout Lumbridge Computer.

## Pools

A `Pool` (`src/pool.rs`) is a named piece of memory with a **budget**, a
**safety margin** and a **watchdog floor**. Its total and available memory are
sensed on demand (`Pool::sense`), never stored. Admission (`can_admit`),
headroom, reservations (`reserved_gb`, `reserved_unused_gb`,
`model_available_gb`), the Scene budget cap (`effective_budget_gb`) and the
watchdog (`below_floor`) are all methods on the pool.

- A DGX Spark-class box has exactly one pool, `unified`: the memory GB10's CPU
  and GPU share, sensed from `/proc/meminfo` (NVML reports no memory on GB10).
  Its numbers: budget 100 GB when a Scene names none, margin 8 GB, floor 3 GB.
- Models and reservations take an optional registry field `pool:`, default
  `unified`. It is an enum (`PoolId`), so an unknown pool is a load error, on
  plain models, `extends:` children and reservations alike.
- `lumbridge-computer.toml [memory] pool` is the same `PoolId`. Doctor builds
  its memory checks from `Pool::from_config` and warns (`memory.governor`) when
  the file's margin or floor differ from the compiled pool. The running agent
  keeps the compiled numbers; `--floor` overrides the floor.
- Activation keeps committed memory per pool, so a model competes only with
  models in its own pool. A Scene's `budget_gb` is the unified pool's budget.

Adding a second pool means a new `PoolId` variant, its `sense()` and its
default numbers. Nothing else in the governor assumes there is only one.

### Two admission checks

A model starts only if it passes both:

1. **Declared.** Committed models plus the new one, plus the safety margin, fit
   the Scene budget (capped by MemTotal minus reservations).
2. **Observed.** The new model's expected size plus the safety margin fits in
   `MemAvailable` (minus the unused part of any reservation). For a model with
   enough samples of its current definition, the expected size is the largest
   resident size it reached in the last 7 days, which can be below its
   declaration: sensed memory already counts every running model at its real
   size, so asking for a padded number on top would charge the padding twice.

A start refused for memory is retried every 30 s, not on the crash backoff.

## Scopes

With scopes on (the default) every model Lumbridge Computer starts runs in its
own transient systemd user scope:

```
user@1000.service/lumbridge-computer.slice/lumbridge-computer-models.slice/lumbridge-computer-model-<id>-<nonce>.scope
```

It is launched as
`systemd-run --user --scope --quiet --collect --unit=… --slice=lumbridge-computer-models.slice -p MemoryHigh=… -p MemorySwapMax=… -- <argv>`.

**Why the pid does not change.** `systemd-run --scope` asks the user manager
to create the scope around its *own* pid, waits for the job, then `execvp`s
the model. Lumbridge Computer has already put that pid in its own process
group, and exec keeps the pid, the process group and the `/proc/<pid>/stat`
start time. So the recorded pid is the model and its group leader: process
identity, `stop_owned` (SIGTERM, then SIGKILL to `-pid`) and adoption all work
unchanged. The `#[ignore]` integration test
`scoped_spawn_keeps_the_model_pid_group_and_identity_and_stops_cleanly`
checks this on a machine with a systemd user manager. For under 50 ms the pid
still reads `systemd-run`, though it is already in the scope. Then it execs. A
group stop empties the scope, and `--collect` removes it.

**Why systemd-run and not cgroupfs.** The agent runs in a service with
`Delegate=no`, so it may not create cgroups or move processes itself. The
user manager already controls `cpu memory pids` on its own subtree. Scope mode
also keeps stdio, environment and working directory, which service mode does
not, so the per-model log and the registry's `env:` keep working.

**Side benefit.** Without scopes every model lives inside the agent's service
cgroup: its `memory.current` is one number for everything, and an agent
restart with any `KillMode` other than `process` would kill every model.
Models in their own scopes are outside the agent's cgroup.

**Unit names** carry a millisecond nonce. If a daemonised grandchild outlives
its process group and keeps an old scope alive, the next start still gets a
fresh name instead of failing with "unit already exists". The first line of
`.compute/logs/<id>.log` names the scope, or says `unscoped`.

**When no scope is used:**
- `scopes: off` in the registry, or `LUMBRIDGE_COMPUTER_SCOPES=off` in the
  agent's environment (turns scopes off without editing the registry);
- no user manager is reachable (`$XDG_RUNTIME_DIR/systemd/private` is
  missing, or `systemd-run` is not on `PATH`);
- the registry command already starts with `systemd-run`. Wrapping it again
  would only move the process from our scope into theirs.

Processes adopted from before scopes were turned on are not moved. `status`
and `footprints` report them as `unscoped`.

## Limits

Each model can set an optional `limits:` block in the registry:

```yaml
  brain:
    footprint_gb: 40
    limits:
      memory_high_gb: 46   # default: footprint × 1.15
      memory_max_gb: 60    # default: unset
      swap_max_gb: 0       # default: 0 if footprint_gb > 0, unset for 0 GB shims
```

- **`MemoryHigh`** throttles the scope and reclaims its memory, but never
  kills it. 0 GB shims get no `MemoryHigh`: a limit of 0 bytes would stall
  them forever.
- **`MemoryMax`** is unset by default. It is an OOM kill inside the scope,
  and an OOM-killed brain is worse than the watchdog shedding a lower-priority
  model.
- **`MemorySwapMax=0`** applies to models that hold memory. When a model's
  host side pages out to swap, its next request hits a latency cliff.

### What a limit means on GB10 unified memory

**CUDA allocations are not charged to the memory cgroup.** On GB10 the driver
allocates them, and they appear in no cgroup `memory.*` file, in no smaps
category and in no `/proc/meminfo` category. So `MemoryHigh`, `MemoryMax`
and `MemorySwapMax` only govern the *host* side of a model: the Python heap,
tokenizer, pinned buffers, page cache and swap. A footprint × 1.15
`MemoryHigh` is a loose guard against a runaway host process. It is not a
bound on the model's real footprint.

The tradeoff with `MemorySwapMax=0`: idle models' host memory often sits in
swap, and that is part of what `MemAvailable` reports as free. Under
`MemorySwapMax=0` those pages stay resident. The box is more predictable (no
swap-in stalls, demand equals residency), but less memory is available. A
model that should be allowed to swap needs an explicit `swap_max_gb`.

## Measurement

The agent samples every 30 s. `status` and `footprints` also measure live.

| Source | Scoped model | Unscoped or adopted model |
|---|---|---|
| host | `memory.current − file + shmem` from the scope (anon + shmem + kernel) | Σ `Pss_Anon + Pss_Shmem` of `smaps_rollup` over the process group |
| page cache (reported, not counted) | `file − shmem` | Σ `Pss_File` |
| swap | `memory.swap.current` | Σ `SwapPss` |
| peak (informational) | `memory.peak` (includes cache) | — |
| GPU | `nvidia-smi --query-compute-apps=pid,used_memory`, summed over the pids in the scope or process group | same |

A model counts as scoped only if it is in a `lumbridge-computer-model-*.scope`
or a `run-*.scope` (a systemd-run in the registry command). The agent's service
cgroup and terminal session scopes are shared, so they say nothing about any
one model. Pss is used rather than Rss because forked vLLM workers share
their parent's pages.

**resident = host + GPU; demand = resident + swap.** Demand is the number to
compare with `footprint_gb`: swapped pages still belong to the model, and
under `MemorySwapMax=0` they become resident.

**GPU memory comes from NVML per process.** On GB10 the aggregate
`nvidia-smi --query-gpu=memory.used` is `[N/A]`, but `--query-compute-apps`
reports `used_memory` per process, and it accounts for the gap. Without it,
every measurement would be a small fraction of the truth: a vLLM engine core
can show 5 GB of RSS and 29 GiB in NVML. The call has a 3 s timeout, so a
wedged nvidia-smi cannot stall the watchdog. When nvidia-smi is unavailable
the GPU reading is `null` (unknown), not 0.

### What GB10 shows

On a GB10 box running a nine-model Scene, `/proc/meminfo` showed 121.7 GiB
total and 22.8 GiB available. AnonPages, Cached, Slab and MemFree added up to
about 40 GiB. The other **~80 GiB appeared in no meminfo category**. They
matched the per-process NVML total of 79.3 GiB. In the brain's engine-core
smaps, the CUDA reservations show up as anonymous VMAs (52.6 GiB of address
space, 1 GiB resident). The memory is real but invisible to smaps and cgroups.

Per model on that box (GiB):

| model | declared | host | GPU | swap | **demand** | vs declared |
|---|---:|---:|---:|---:|---:|---:|
| brain (vLLM, 35B NVFP4) | 40 | 7.6 | 29.2 | 0.0 | **36.8** | −3.2 |
| asr (large speech model) | 26 | 1.1 | 26.4 | 3.9 | **31.5** | **+5.5** |
| tts | 12 | 3.9 | 8.7 | 0.0 | **12.5** | +0.5 |
| embed (vLLM, 1B) | 3 | 0.4 | 8.0 | 2.1 | **10.4** | **+7.4** |
| rerank (vLLM, 1B) | 5 | 0.3 | 4.1 | 2.3 | **6.8** | **+1.8** |

What this shows:
- **Small vLLM models are the usual offenders.** An embedder declared at 3 GB
  with `gpu-memory-utilization 0.02` held 8 GiB of CUDA memory: CUDA graphs,
  the torch allocator and activation workspace sit outside the KV fraction.
- **Swap belongs to the models.** Most of the swap in use was the host side of
  idle small models.
- **The brain's declaration was about right.** Total resident memory roughly
  matched the declared total; the overrun was spread across models and hidden
  in swap.
- Wait for a week of `lumbridge-computer footprints` p99 before changing a
  footprint: a single reading misses KV-cache growth under load.

### Sizing vLLM's KV cache in bytes

vLLM sizes itself with `--gpu-memory-utilization`, a fraction of the *whole*
pool, per instance, covering weights and KV cache together. On a unified-memory
box that means the number you reason about (GB) is not the number you type,
and a quantisation change silently changes context capacity. vLLM also accepts
`--kv-cache-memory-bytes` (for example `12G`), which ignores the fraction. Then
a model's memory is `weights + KV bytes + runtime overhead`, three numbers that
do not depend on what else runs. `serve.args` passes it through as is. Measure
the result with `lumbridge-computer footprints` and set `footprint_gb` from it.

## Docker models

A registry entry whose command is `docker run …` (usually `serve.kind: docker`)
is launched like any other model, but the process Lumbridge Computer spawns
and records is the **docker CLI client**. The model runs under containerd in
its own cgroup, `system.slice/docker-<cid>.scope`, with its own pids.
`src/docker.rs` handles that:

- **Start.** `--cidfile <root>/.compute/cid/<id>.cid` is inserted right after
  `run` (a registry `--cidfile` among the run options is replaced; anything
  after the image is left alone). Before spawning, the container a stale
  cidfile names and any container (running or stopped) already holding the
  same `--name` are `docker rm -f`'d, each logged in `.compute/logs/<id>.log`.
  After spawning, the start polls the cidfile for up to 30 s while the CLI
  lives and stores the id as `container_id` in `state.yaml`. A first start
  that pulls an image may outlast the 30 s; that record is then tracked by name.
- **Measure** (`[container]`). The container's cgroup comes from its main
  pid (`docker inspect -f '{{.State.Pid}}'` → `/proc/<pid>/cgroup`), else the
  systemd path `system.slice/docker-<cid>.scope`. Host, swap and peak are read
  as for a scope. GPU = the nvidia-smi compute pids whose `/proc/<pid>/cgroup`
  is that cgroup or below it. The CLI's own Pss (a few tens of MB) is added.
  If docker cannot be asked the record is measured as the CLI only (`[unscoped]`).
- **Liveness.** Supervision's "the process is gone" is true when the CLI is
  gone **or** docker reports the container `exited`, `dead`, `removing`, or
  (by stored id) no longer present. `created` counts as alive, since the
  cidfile is written before the container starts. A docker error is never
  evidence: it is logged once per distinct message and the container is
  treated as alive. Ownership, which gates every signal, still means the exact
  CLI process Lumbridge Computer started.
- **Stop.** `docker stop -t 30 <cid>` first (the memory watchdog uses 6 s),
  then the CLI's process group, then `docker rm -f` if the container is still
  present. With a stored id the container is stopped even if its CLI already
  died.
- **Records without `container_id`** are resolved by the `--name` in the CLI's
  own `/proc/<pid>/cmdline` (`docker ps -aq --no-trunc -f name=^<name>$`).
  This only happens while the CLI is still the exact process Lumbridge
  Computer recorded, so a reused pid can never lead to someone else's
  container. A name that matches nothing is not taken as an exit (the CLI may
  be pulling).

Every docker call is bounded (10 s; `stop` gets its grace plus 15 s) and goes
through the `docker::Docker` trait, so unit tests script docker and never need
a daemon.

## Learned footprints

Each sample goes into `footprint_samples` in the telemetry SQLite database
(`.compute/usage/telemetry.sqlite3`). Samples are kept for 30 days. Each
sample records which definition of the model it belongs to (a hash of the
model's `serve` section), so a different model under the same id does not
inherit old numbers.

```
lumbridge-computer footprints [--since 7d] [--json]
```

For each running or recently sampled model it shows: declared, current
demand, GPU, swap, peak and p99 demand over the window, and a **suggested
`footprint_gb` = p99 + max(1 GB, 10 %), rounded up**. If any sample in the
window has a GPU reading, samples without one are ignored, so a missed
nvidia-smi call does not make a model look like it shrank by its whole KV
cache. `status` shows `measured X GB (Y swap) [scope|container|unscoped]`
next to each model's declared footprint.

### Admission

Admission uses the declared numbers by default. The opt-in

```yaml
admission: measured
```

budgets each model at `max(footprint_gb, learned p99 resident size over 7
days)`. This applies both to the model being admitted and to everything
already committed. It can only raise a number, so no model is ever admitted on
less than its author declared. Resident, not demand: the budget is RAM, and
pages a model has had swapped out are not in it. A model's current definition
needs 20 samples (10 minutes) with a GPU reading before they count; until
then it uses the declared number.

### Watchdog

When the floor trips, the watchdog measures the owned models once. Among
equal `shed_priority`, it stops the model furthest over its declared
footprint first, since that model's declaration let the box get here. It
compares in 0.1 GB steps, so noise never reorders equals. When nothing is
measured, the order is: priority, then models that hold memory before 0 GB
shims, then newest.

### Memory pressure (PSI)

`MemAvailable` says how much could be handed out, not whether the kernel is
already struggling to hand it out. On unified memory, reclaim and compaction
stall tasks well before the floor is reached. The agent reads
`/proc/pressure/memory` and logs a warning when `some avg10` reaches 10 % or
`full avg10` reaches 2 %, repeated every 5 minutes while it stands. It only
warns: nothing is stopped or refused on pressure.

## The Supabase memory floor

The Supabase containers run in a **system** slice, `lumbridge-server.slice`
under `lumbridge.slice` (Docker's `cgroup_parent` creates it as an implicit
slice; there is no unit file). By default nothing protects it: under memory
pressure the kernel may swap Postgres out to make room for a model.

`lumbridge-computer.toml [supabase] memory_min_gb` (default 4, the db
container's `mem_limit`; must lie within `reservation_gb`) sets cgroup v2
`memory.min` on the slice. A cgroup's `memory.min` is capped by its parent's,
so the floor goes on **every** slice on the path:

```
/etc/systemd/system/lumbridge.slice.d/50-lumbridge-computer-memory-min.conf
/etc/systemd/system/lumbridge-server.slice.d/50-lumbridge-computer-memory-min.conf
```

each containing:

```ini
# Written by `lumbridge-computer apply --only supabase-floor` from lumbridge-computer.toml
# ([supabase] memory_min_gb = 4). cgroup v2 memory.min: under memory pressure
# the kernel reclaims from unprotected cgroups (the models) before it touches
# the first 4G of lumbridge-server.slice. A slice's protection is capped by its
# parent's, so every slice above it carries the same floor.
[Slice]
MemoryMin=4G
```

followed by `systemctl daemon-reload`.

- `doctor` → `supabase.memory_floor` reads `memory.min`/`memory.low` of the
  slice and its ancestors from `/sys/fs/cgroup` and warns when the effective
  protection (the smallest `memory.min` on the path) is below the floor.
- `apply --dry-run` lists `update memory floor` with both file paths and
  contents (`--json` too).
- `apply --only supabase-floor` writes the drop-ins (atomically, mode 0644),
  reloads systemd, and reads `memory.min` back. Because the slice is a system
  slice, as a normal user it changes nothing and prints the exact `sudo`
  commands, exiting 3. Run it again once the kernel shows the floor and it
  says `nothing to do`. See [box.md](box.md).

A floor below the slice's current use reclaims nothing at apply time. It only
changes what the kernel reclaims first the next time memory runs short.

## Swap

`lumbridge-computer doctor` prints `memory.swap.top`: the top swap holders
(`VmSwap` from `/proc/*/status`) grouped by cgroup, each tagged `[models]`,
`[supabase]`, `[desktop]` or `[other]`. When more than half of swap is in use
it warns, with a hint chosen by the largest holder.

What to expect on a GB10 box: the small services come up first, their
CPU-side pages (Python heaps, host-side copies made at init) go cold, and the
brain's load spike pushes them to swap, where they stay. That is usually
harmless. Check `/proc/pressure/memory` and swap-in traffic (`pswpin` in
`/proc/vmstat`) before acting. Advice:

1. **Leave `vm.swappiness` and zswap alone** unless swap-in traffic or PSI
   rises. Lowering swappiness box-wide makes the kernel drop page cache
   (mostly weights read once) instead, and affects everything else too.
2. **Apply the Supabase floor** (above). Postgres being swapped out is the
   swap event that hurts.
3. **`swap_max_gb: 0` only for latency-critical models** (an ASR on a
   dictation path, say), whose first request after idle would stall on
   page-ins. Pinning a model in RAM costs what it had in swap, and that memory
   then has to be in its footprint.
