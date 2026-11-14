# Roadmap

What Lumbridge Computer does today is in the [README](../README.md) and the
[change log](../CHANGELOG.md). This is what is planned, in no fixed order.
Nothing here is promised.

## Core contracts

1. **Models**: local, vetted launch recipes with pinned revisions and measured footprints.
2. **Scenes**: named sets of models, activated through admission control.
3. **Evals**: versioned prompts and assertions with portable results.
4. **Artifacts**: weights, adapters and datasets, downloaded resumably and checksummed.
5. **Jobs**: inference, evaluation, download and conversion lifecycles.

Training is outside Lumbridge Computer. It admits and serves a model only
after its weights are on the box, hashed and registered.

## Planned

- `lumbridge-computer model pull`: resumable, checksum-verified downloads.
- Eval comparison, regression thresholds, warmup policy, and load tests.
- TTFT and latency percentiles from the engines' own metrics, and retention
  compaction for the telemetry database.
- Scene hooks for preflight, post-activation eval gates, rollback, and schedules.
- Gateway aliases, so clients follow the active Scene without configuration changes.
- A fair-share Jobs scheduler driven by observed queue pressure and latency
  (see [telemetry.md](telemetry.md#four-lane-scheduling-policy)).
- Admission that makes room by evicting idle models and retrying, instead of
  refusing.
- More than one kind of box: memory pools beyond GB10's single unified pool.

## Open-source housekeeping

- CI on aarch64 as well as x86_64.
- Versioned JSON schemas for the Scene, registry and eval formats.
