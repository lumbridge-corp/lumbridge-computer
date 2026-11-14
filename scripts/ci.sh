#!/usr/bin/env bash
# What CI runs, and what to run before a pull request: scripts/ci.sh
# Format, clippy with warnings as errors, and the whole test suite. The tests
# need no GPU, Docker, Tailscale or network: they use fixtures and fake servers
# bound to 127.0.0.1:0, and every model root they create is a temporary folder.
set -euo pipefail
cd "$(dirname "$0")/.."
# Variables a real box sets must not leak into the tests.
for v in $(env | grep -oE '^LUMBRIDGE_COMPUTER_[A-Za-z0-9_]*' || true); do unset "$v"; done
rustc --version
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --no-fail-fast -- --test-threads "${RUST_TEST_THREADS:-4}"
