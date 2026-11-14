# Setting up a box

Lumbridge Computer is one Rust binary. It does not install model runtimes,
weights or voices: you provide those, and the registry says how to start each
model. This page is the order of work for a new box.

Target: an NVIDIA DGX Spark-class machine (GB10, 128 GB unified memory,
aarch64, Ubuntu 24.04 based OS), with cgroup v2, a systemd user manager with
lingering on, and Docker if you run Docker models or Supabase. `doctor` checks
each of these.

## 1. Build

```sh
cargo build --locked --release
install -m 755 target/release/lumbridge-computer ~/.local/bin/
```

## 2. Make a root

Copy the reference stack into a directory of your own (a private Git repo is a
good home for it):

```sh
mkdir -p ~/lumbridge-computer
cp -r registry scenes evals ~/lumbridge-computer/
export LUMBRIDGE_COMPUTER_ROOT=~/lumbridge-computer
```

`registry/models.yaml` is an example, not a working recipe for your box. For
each model you keep: install its runtime (a vLLM virtualenv, a container
image, a llama.cpp build), download its weights under `~/models/`, and fix
`serve` (executable, weights, port, args) to match. Remove the models you do
not run.

## 3. Check before you start anything

```sh
lumbridge-computer model show brain          # the resolved entry and the argv it launches
lumbridge-computer scene show everything     # footprints and the admission verdict
lumbridge-computer scene activate everything --dry-run
```

Then start models one at a time if you can, and measure them: run
`lumbridge-computer footprints` after a day of normal use and set each
`footprint_gb` from its suggestion. Declared footprints that are too small are
the usual reason a box runs out of memory.

## 4. Activate and install the agent

```sh
lumbridge-computer scene activate everything
lumbridge-computer status
```

Then install the agent unit ([operations.md](operations.md#installing-the-agent)).

## 5. Optional: Supabase and Tailscale

`assets/supabase/install.sh` installs a self-hosted Supabase pinned to the
upstream release in `assets/supabase/UPSTREAM`, with every port on loopback
and its memory reserved in a cgroup slice. Describe the box in
`lumbridge-computer.toml` and run `lumbridge-computer doctor`. See
[box.md](box.md).

## Network exposure

The registry's model endpoints and the agent's gateway have no
authentication. Bind models to `127.0.0.1` where you can and reach them through
the gateway; keep the gateway on a private network (a tailnet, a LAN), never
on the internet. The read-only HTTP API (`lumbridge-computer api`) listens on
`127.0.0.1:8012` by default; that port clashes with an embedder on :8012, so
pass `--listen` with another port if you run both.

## Data you keep out of Git

Weights, voices, recordings, speaker enrollments, `.env` files and the
`.compute/` state directory stay out of any repository. Shared Scenes and eval
suites name model ids only; the commands stay in your own registry.
