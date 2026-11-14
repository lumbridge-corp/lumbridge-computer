# lumbridge-computer

Lumbridge Computer turns a unified-memory GPU box (an NVIDIA DGX Spark-class
machine: GB10, 128 GB, aarch64) into a self-hosted AI computer. One Rust
binary, `lumbridge-computer`, owns the models (Scenes, memory admission,
routing, health, telemetry), the box's Supabase, and what it publishes on
Tailscale.

Read the code. This file holds only what reading won't tell you: the rules
and the decisions that look arbitrary until you know why.

## Rules that aren't in the code

- Never run anything that changes state on a box serving real users while you
  work: no `scene activate|deactivate|adopt|resume`, no `model start|stop|sleep|wake`,
  no `apply` without `--dry-run`, no `systemctl`, `kill`,
  `docker compose up|down|restart` or `tailscale serve|up|set`. Read-only
  commands (`status`, `doctor`, `--dry-run`, `curl 127.0.0.1:<port>/metrics`,
  `docker ps`, `tailscale status --json`, `journalctl`) are fine. Use
  `examples/demo` to watch real transitions.
- The hardware target is one kind of box. Don't add code paths for other
  hardware; keep memory accounting behind the `Pool` idea so other boxes are
  config later.
- On GB10, NVML/`nvidia-smi` aggregate memory is "Not Supported": memory truth
  is `/proc/meminfo`, cgroup v2 files and per-process NVML. Never trust CUDA
  free memory.
- Model commands live in the trusted registry, never in Scene or eval
  manifests. No field of a Scene may start a process or run a command.
- A busy engine is never restarted. Read `src/health.rs` before changing
  supervision.
- `--json` and `snapshot` output are interfaces: change them only on purpose,
  with the tests and docs in the same change.
- Never commit state (`.compute/`), weights, recordings, voice samples,
  `.env`, keys or tokens. A box's own registry and `lumbridge-computer.toml`
  belong in that box's own repository.
- Fixtures are made up. Use the example host `box1`, tailnet `tail0000.ts.net`,
  home `/home/sam`, and `example.com` addresses. `scripts/scrub-check` fails on
  private addresses, home folders and token formats.

## Gates (all must pass before a change lands on main)

```sh
scripts/ci.sh        # cargo fmt --check, clippy -D warnings, cargo test --locked
scripts/scrub-check
```

Tests must not need a GPU, Docker, Tailscale or the network: use fixtures,
scripted fakes and servers bound to `127.0.0.1:0`.

## Style

- Small dependency set; justify each new crate in the pull request.
- No `unwrap()`/`expect()` outside tests; errors carry context (`anyhow`).
- Comments explain *why*. Tests are named as sentences describing behaviour.
- Commit messages: imperative subject, body says what changed and why.
