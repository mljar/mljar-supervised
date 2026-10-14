#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the TabArena comparison, one MLJAR version per terminal:

  # Terminal 1
  ./benchmarks/run.sh 1.3.2

  # Terminal 2
  ./benchmarks/run.sh 1.4.0rc3

Each version runs the small suite in Compete mode with a 3600-second budget
per dataset and n_jobs=15. Python output is unbuffered for live console logs.

Environments are installed automatically; no activation is needed. RC3 uses
CUDA and prepares pretrained weights before training. The first setup needs
uv, internet access, and several GB of disk space.

Extra runner options may follow the version, for example:
  ./benchmarks/run.sh 1.4.0rc3 --setup-only
  ./benchmarks/run.sh 1.4.0rc3 --suite smoke --seconds 60
  ./benchmarks/run.sh 1.3.2 --datasets credit-g --seconds 60

Completed results are retained and skipped when rerunning the same command.
RC3 has its own environment and output directory; RC1 and RC2 outputs are preserved.
RC1 remains supported. Use --candidate-version 1.4.0rc1 with 1.3.2 for a
fresh RC1 comparison. See benchmarks/tabarena/comparison/README.md to compare
RC3 against an existing 1.3.2 run without repeating baseline training.
EOF
}

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

benchmark_version="$1"
shift
case "$benchmark_version" in
    1.3.2|1.4.0rc1|1.4.0rc2|1.4.0rc3) ;;
    *)
        printf 'Unsupported MLJAR version: %s\n' "$benchmark_version" >&2
        usage >&2
        exit 2
        ;;
esac

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$benchmark_dir/.."

exec python3 -u benchmarks/tabarena/comparison/run.py \
    --version "$benchmark_version" \
    --suite small \
    --seconds 3600 \
    --n-jobs 15 \
    "$@"
