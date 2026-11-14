# The box: `lumbridge-computer.toml`, `doctor`, `apply`

`lumbridge-computer.toml` describes the whole box as it should be: the hardware, its
unified memory, the model root and desired Scene, the Supabase stack and what the
box serves on the tailnet. Keep it with your box's own configuration (a private
repo of your registry, scenes and this file), not in this repo.
`src/box_/fixtures/lumbridge-computer.toml` is a complete example.

```sh
lumbridge-computer doctor [--config lumbridge-computer.toml] [--json]
lumbridge-computer apply --dry-run [--only supabase-floor] [--config lumbridge-computer.toml] [--json]
lumbridge-computer apply --only supabase-floor [--config lumbridge-computer.toml]
```

The config path is `--config`, then `$LUMBRIDGE_COMPUTER_CONFIG`, then `./lumbridge-computer.toml`.

**Doctor and the dry run are read-only.** The one real apply is
`apply --only supabase-floor` (below). Every external command goes
through one runner (`src/box_/cmd.rs`) that refuses anything outside a read-only
allowlist: `tailscale status|serve status --json`, `tailscale version`,
`docker ps`, `docker inspect --format`, `docker compose ls`,
`docker compose -p <project> ps`, `docker exec <db> pg_isready`, `nvidia-smi`
queries, `df` and `uname -m`. `docker inspect` uses a template naming only the
fields the checks use, so container environments (which hold Supabase secrets)
are never read. The Supabase `.env` is read for key names only; values are
dropped while parsing. Nothing is written: the Scene plan uses
`lifecycle::plan_read_only`, which skips the transition lock.

## `lumbridge-computer.toml` reference

All tables reject unknown keys, so a typo is a load error. Paths may start
with `~/`. "GB" means GiB throughout Lumbridge Computer.

```toml
[box]
name = "box1"
hardware = "dgx-spark"    # the only accepted value today
arch = "aarch64"          # default; what `uname -m` must say
os = "ubuntu-24.04"       # optional; <ID>-<VERSION_ID> from /etc/os-release

[memory]
pool = "unified"          # GB10: one pool for CPU and GPU (see docs/memory.md)
budget_gb = 113           # ceiling for model memory; no Scene budget should exceed it
safety_margin_gb = 8      # must match the governor in this build (not configurable yet)
watchdog_floor_gb = 3     # ditto; doctor warns if they differ

[models]
root = "~/lumbridge-computer"   # holds registry/, scenes/ and .compute/ state
desired_scene = "everything"
# registry = "registry/models.yaml" # relative to root; only these defaults are accepted today
# scenes = "scenes"

[supabase]
project = "lumbridge-computer-supabase-home"   # compose project name
dir = "~/lumbridge-computer/supabase-home"
upstream_ref = "self-hosted/v0.8.4"
upstream_commit = "189adda99d1b56da55fc5a6ee3f849354aaea92a"  # the tag peeled to a commit
overlay = "lumbridge-computer.override.yml"   # file name inside dir
slice = "lumbridge-server.slice"         # default lumbridge-computer.slice
reservation_gb = 8.5                     # must equal the registry reservation for the slice
memory_min_gb = 4                        # default 4 (the db container's limit); 0..=reservation_gb
public_url = "https://box1.tail0000.ts.net:8443"
tailnet_port = 8443                      # must equal public_url's port

[tailscale]
tailnet = "tail0000.ts.net"              # MagicDNS suffix
hostname = "box1"

[[tailscale.routes]]                     # one per `tailscale serve` handler; unlisted ones are drift
https_port = 8443
path = "/"                               # default "/"
proxy = "http://127.0.0.1:8000"          # also "127.0.0.1:8000" or "8000"; loopback only
owner = "supabase"                       # optional label for humans
```

Load-time validation also checks that `public_url` is
`https://<hostname>.<tailnet>:<tailnet_port>`, that routes are unique per
`(https_port, path)`, and that `upstream_commit` is a full 40-character hash.

The slice maps to its cgroup path the way systemd nests slices:
`lumbridge-server.slice` lives at `lumbridge.slice/lumbridge-server.slice`.

## `lumbridge-computer doctor`

Each check prints `ok`, `WARN` or `FAIL`, a one-line reason and, when not ok, a
fix hint. Exit code: 1 if any check failed, otherwise 0 (warnings allowed).
`--json` prints `[{id, status, reason, hint?}]`.

| id | ok when | otherwise |
|---|---|---|
| `hardware.arch` | `uname -m` equals `[box] arch` | fail |
| `hardware.os` | os-release equals `[box] os` (if set) | warn |
| `hardware.gpu` | nvidia-smi lists a GB10 | fail |
| `hardware.driver` | driver and CUDA version are reported | warn |
| `hardware.nvml_memory` | NVML memory is `[N/A]`/Not Supported — **expected on GB10** | warn if NVML reports a number |
| `hardware.cgroup_v2` | `/sys/fs/cgroup/cgroup.controllers` exists | fail |
| `hardware.linger` | `/var/lib/systemd/linger/$USER` exists | fail |
| `disk.models` | ≥ 100 GB free on the models root filesystem | warn < 100, fail < 20 |
| `memory.available` | MemAvailable ≥ floor + margin | warn above floor, fail below |
| `memory.swap` | ≤ 50% of swap in use | warn |
| `memory.swap.top` | lists the top swap holders by cgroup (`VmSwap` from `/proc/*/status`), tagged models/supabase/desktop/other; ok while ≤ 50% of swap is used | warn, with a hint for the largest holder (see docs/memory.md) |
| `memory.budget` | `budget_gb` ≤ MemTotal − `reservation_gb` | warn |
| `memory.governor` | margin/floor equal this build's governor constants | warn |
| `memory.scene_budget` | (only shown when) the desired Scene asks for more than `budget_gb` | warn |
| `docker.daemon` | `docker ps` answers | fail (Supabase checks skipped) |
| `supabase.project` | compose project exists | fail |
| `supabase.health` | every container running, healthy where it has a healthcheck | fail |
| `supabase.overlay` | overlay file exists and the project was started with it | fail |
| `supabase.slice` | every container's cgroup parent is `slice` | fail |
| `supabase.db_oom` | db `oom_score_adj` < 0 | fail |
| `supabase.loopback` | every published port is on 127.0.0.1; **any 0.0.0.0/`::`/empty host IP fails** | fail |
| `supabase.pin` | the deployment's version stamp equals `upstream_ref` (and commit, if stamped) | fail; warn if no stamp |
| `supabase.stamps` | (only shown when) another stamp file disagrees | warn |
| `supabase.images` | each running image equals its `image:` in `docker-compose.yml`, and no default service is missing | fail |
| `supabase.limits` | every container has a memory limit and they sum to ≤ `reservation_gb` | warn |
| `supabase.memory` | (only shown when) the slice uses > 90% of its reservation | warn |
| `supabase.memory_floor` | `memory.min` ≥ `memory_min_gb` on the slice **and every ancestor slice** (read from `/sys/fs/cgroup`); `memory.low` alone is reported ok as best effort | warn when unprotected, a user slice, or the slice has no cgroup |
| `supabase.pg_isready` | `pg_isready` in the db container succeeds | fail |
| `supabase.env` | `.env` has the required keys and is mode 600 | fail |
| `supabase.reservation` | the registry reserves `reservation_gb` for the slice's cgroup | fail |
| `supabase.public_url` | the route on `tailnet_port` proxies to the API gateway's host port | fail |
| `tailscale.login` | backend state `Running` | fail |
| `tailscale.identity` | hostname and MagicDNS suffix match | fail |
| `tailscale.health` | (only shown when) tailscaled reports health warnings | warn |
| `tailscale.routes` | served routes equal `[[tailscale.routes]]` | fail |
| `tailscale.funnel` | no Funnel entry; **any Funnel route fails** | fail |
| `models.registry` | the registry loads | fail |
| `models.scenes` | scenes load and the desired Scene exists | fail |
| `models.desired_scene` | the planner validates the desired Scene (`plan_read_only`) | fail |
| `models.active` | the active Scene is the desired one, no transition in flight | warn |

Version stamps are read in order: `.supabase-version` (written by upstream's
`update.sh`), then `.lumbridge-computer-upstream` (written by `install.sh`). The
first one found is the deployed version.

## `lumbridge-computer apply --dry-run`

Computes what applying `lumbridge-computer.toml` would change, per area:

- **tailscale**: hostname/tailnet drift; serve routes to `add`, `remove`
  (served but unlisted) or `update` (different backend, plain HTTP, or served
  under another host name); any Funnel entry as `update funnel … on -> off`.
- **supabase**: `install` when the project is missing (nothing else is
  listed); project dir or overlay not loaded; `upgrade` when the deployed
  stamp differs from `upstream_ref`/`upstream_commit`; per service, `start`
  when not running and `recreate` with the reasons (image differs from
  `docker-compose.yml`, wrong slice, db `oom_score_adj` ≥ 0, a port beyond
  loopback); the registry reservation for the slice; the **memory floor**
  as `update memory floor` followed by every drop-in it would write, path and
  full contents (see below).
- **models**: `activate` when the active Scene isn't the desired one, plus the
  planner's `stop` and `start` lists for the desired Scene.

When everything matches it prints exactly `no changes`. An area whose state
can't be read (tailscaled down, Docker down, the planner refusing) is printed
as `! cannot diff <area>: <reason>`; silence always means "verified equal".

Exit code: 0 no changes, 2 changes pending, 1 when an area couldn't be read.
`--json` prints `{changes: [{area, action, target, detail}], blockers: [..]}`.

Without `--dry-run`, `lumbridge-computer apply` prints that apply is not available yet
and exits 1. The only mutating apply is `--only supabase-floor`.

## Supabase assets

`assets/supabase/` holds the installer (`install.sh`), the overlay
(`lumbridge-computer.override.yml`), the migration runner and the upstream pin
(`UPSTREAM`, self-hosted v0.8.4). New installs are compose project
`lumbridge-computer-supabase-<name>` in `$LUMBRIDGE_COMPUTER_HOME/supabase-<name>`
(`$LUMBRIDGE_COMPUTER_HOME` defaults to `~/lumbridge-computer`), with containers in
`$LUMBRIDGE_COMPUTER_SLICE` (default `lumbridge-computer.slice`, written to `.env`).
Whatever slice you use, state it in `lumbridge-computer.toml` (`[supabase] slice`).

```sh
sh assets/supabase/install.sh --name home --public-url https://box1.tail0000.ts.net:8443 [--serve]
sh assets/supabase/migrate.sh --name home <migrations-dir>
```

`install.sh` generates the secrets once into `.env` (mode 600, never printed),
binds every port to loopback, and refuses to leave a port published beyond
loopback. `--serve` publishes the API gateway on the tailnet with
`tailscale serve`.

## `lumbridge-computer apply --only supabase-floor`

Guarantees the Supabase slice `memory_min_gb` of cgroup v2 `memory.min`, so
under memory pressure the kernel reclaims from models before Postgres. The
overlay's `mem_limit`s cap the stack and `oom_score_adj` picks the OOM victim,
but neither stops reclaim (swapping the database out) when a model overshoots.

- A cgroup's protection is capped by its parent's, so a drop-in goes on every
  slice from the root down: for `lumbridge-server.slice` that is
  `lumbridge.slice` and `lumbridge-server.slice`, as
  `/etc/systemd/system/<slice>.d/50-lumbridge-computer-memory-min.conf`:

  ```ini
  [Slice]
  MemoryMin=4G
  ```

  then `systemctl daemon-reload`, after which the result is read back from
  `/sys/fs/cgroup/.../memory.min` and printed.
- The slice is a **system** slice (dockerd is a system service; it lives at
  `/sys/fs/cgroup/lumbridge.slice/lumbridge-server.slice`), so this needs
  root. Run as a normal user it changes nothing, prints the exact `sudo`
  commands (the same files and the reload) and exits 3.
- Idempotent: when the kernel already guarantees the floor it prints
  `nothing to do` and exits 0; drop-ins already in place but not yet effective
  only trigger the reload.
- A slice found under `user@<uid>.service` is refused: `user.slice` caps its
  protection, so a floor there would protect nothing.
- `--host-root <dir>` (hidden, for testing) treats `<dir>` as `/` and skips
  the reload.
