# Contributing

Report bugs and request features as issues in this repository. Discuss bigger changes there before writing a PR. Report security bugs by email instead: see [SECURITY.md](SECURITY.md). Keep PRs small and add tests for behaviour changes.

## Setup

Install Rust 1.88 or newer (with `rustfmt` and `clippy`), a C toolchain and Git. The README's install section has the commands for Ubuntu. Nothing else is needed to build and test: the tests use no GPU, Docker, Tailscale or network.

## Checks

```sh
scripts/ci.sh
scripts/scrub-check
```

`scripts/ci.sh` runs what CI runs: `cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings` and `cargo test --locked`. `scripts/scrub-check` fails on private network addresses, home folders, email addresses and token formats that are not listed in `.scrub-allow`. Use made-up examples in code, fixtures and docs: user `sam` (`/home/sam`), host `box1`, tailnet `tail0000.ts.net`, `example.com`.

See [AGENTS.md](AGENTS.md) for the project rules.

## Formats

Changes to the `lumbridge/v1` Scene, registry or eval formats need fixtures, notes on backward compatibility, and docs ([docs/scene-spec.md](docs/scene-spec.md)) in the same PR. Never add secrets, private model tokens, personal voice samples, internal host names or commands to Scene or eval manifests.

Performance changes should include before/after numbers and the hardware, model and runtime configuration needed to reproduce them.

PRs target `main`. Write what users will notice in CHANGELOG.md. By sending a PR you agree that your change is licensed under Apache-2.0, like the rest of the project.
