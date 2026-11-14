# Changelog

## [Unreleased]

## [0.1.0] - 2026-11-13

The first public release. Built from source; written for an NVIDIA DGX
Spark-class box (GB10, 128 GB unified memory, Linux aarch64).

- Scenes: named sets of models from a trusted local registry, with `extends:`
  and `${PORT}`/`${MODEL_ID}` macros, transactional activation with rollback,
  adoption of running models, and resume after a restart.
- Memory admission against both a declared budget and live `MemAvailable`, an
  OOM watchdog that sheds models by priority, optional admission from measured
  footprints, and a memory-pressure (PSI) warning.
- Per-model systemd scopes and limits, measured footprints (host, GPU per
  process, swap) with `lumbridge-computer footprints`, and Docker models
  tracked by container.
- Supervision that never restarts a busy engine: process exit, stall, readiness
  and an idle chat probe.
- An OpenAI-compatible gateway that routes by `model`, streams untouched, and
  holds requests while a sleeping or on-demand model wakes.
- vLLM level-2 sleep with a worker extension that reloads weights and the MTP
  drafter in seconds.
- Telemetry from vLLM, SGLang, llama.cpp and NeMo metrics into local SQLite;
  `usage` reports; eval suites.
- `doctor` and `apply --dry-run` for the box (hardware, Supabase, Tailscale,
  models); `apply --only supabase-floor`.
- `snapshot`, a sanitized status line; an MCP server and a read-only HTTP API.
