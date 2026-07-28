#!/usr/bin/env bash
#
# run_all_build_modes.sh — run benchmark.sh + scaling sweep for each BUILD_MODE.
#
#   default     — compiler default
#   O3          — -O3
#   O3compress  — -O3 --offload-compress
#
# Usage:
#   ./run_all_build_modes.sh
#   GFX_TARGETS=gfx950 RUNS=5 ./run_all_build_modes.sh
#
# Outputs:
#   artifacts/benchmark/results.csv
#   artifacts/_singlekernel_scaling/data_{default,O3,O3compress}.csv
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RUNS="${RUNS:-5}"
TRIALS="${TRIALS:-3}"
mkdir -p artifacts/benchmark artifacts/_singlekernel_scaling

RESULTS_CSV="artifacts/benchmark/results.csv"
echo "mode,gfx_targets,compile_fat_ms,compile_spirv_ms,bin_fat_bytes,bin_spirv_bytes,first_fat_ms,first_spirv_ms,steady_fat_ms,steady_spirv_ms" > "$RESULTS_CSV"

GFX_TARGETS_1="${GFX_TARGETS:-gfx950}"
GFX_TARGETS_2="${GFX_TARGETS_2:-gfx942 gfx950}"

slug() { echo "$1" | tr ' ;' '__'; }

for mode in default O3 O3compress; do
    export BUILD_MODE="$mode"
    echo ""
    echo "################################################################"
    echo "# BUILD_MODE=$mode"
    echo "################################################################"

    export BENCHMARK_RESULTS_CSV="$RESULTS_CSV"

    export GFX_TARGETS="$GFX_TARGETS_1"
    ./benchmark.sh "$RUNS" 2>&1 | tee "artifacts/benchmark/${mode}_$(slug "$GFX_TARGETS_1").log" | tail -25

    export GFX_TARGETS="$GFX_TARGETS_2"
    ./benchmark.sh "$RUNS" 2>&1 | tee "artifacts/benchmark/${mode}_$(slug "$GFX_TARGETS_2").log" | tail -15

    TRIALS="$TRIALS" ./run_singlekernel_scaling.sh 2>&1 | tee "artifacts/_singlekernel_scaling/${mode}.log" | grep -E '^[0-9]|wrote'
    mv -f "artifacts/_singlekernel_scaling/data.csv" \
        "artifacts/_singlekernel_scaling/data_${mode}.csv"
done

echo ""
echo "Wrote $RESULTS_CSV"
ls -la artifacts/_singlekernel_scaling/data_*.csv
echo "Next: python3 plot_single_kernel_by_mode.py plot_single_kernel_scaling_by_mode.py"
