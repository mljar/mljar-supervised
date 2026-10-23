#!/usr/bin/env bash
set -euo pipefail
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$benchmark_dir/.."
exec python3 -u benchmarks/tabarena/comparison/reevaluate.py "$@"
