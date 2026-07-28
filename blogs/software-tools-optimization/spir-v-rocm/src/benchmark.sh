#!/bin/bash
#
# benchmark.sh — Compare build time, binary size, and runtime performance
# between arch-specific and SPIR-V compiled HIP binaries.
#
# Usage:
#   ./benchmark.sh [runs]
#   BUILD_MODE=default ./benchmark.sh 5
#   BUILD_MODE=O3compress GFX_TARGETS=gfx950 ./benchmark.sh 5
#
# BUILD_MODE: default | O3 | O3compress  (see build_mode.inc.sh)
#
# When BENCHMARK_RESULTS_CSV is set, appends one result row (used by run_all_build_modes.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=build_mode.inc.sh
source "$SCRIPT_DIR/build_mode.inc.sh"

IFS=' ;' read -r -a GFX_TARGETS <<< "${GFX_TARGETS:-gfx1201 gfx1100}"
RUNS="${1:-5}"

SRC="$SCRIPT_DIR/custom_kernel_standard.hip"
SRC_SPIRV="$SCRIPT_DIR/custom_kernel_spirv.hip"
BIN_FAT="$SCRIPT_DIR/vecadd_fat"
BIN_SPIRV="$SCRIPT_DIR/vecadd_spirv"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

separator() {
    echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"
}

check_compiler() {
    if command -v /opt/rocm/llvm/bin/clang++ &>/dev/null; then
        CLANGXX="/opt/rocm/llvm/bin/clang++"
    elif command -v clang++ &>/dev/null; then
        CLANGXX="clang++"
    else
        echo -e "${RED}Error: clang++ not found. Ensure ROCm is installed and in PATH.${NC}"
        exit 1
    fi
    echo "Using compiler: $CLANGXX"
    echo "BUILD_MODE:   $BUILD_MODE ($BUILD_MODE_LABEL)"
    echo "Flags:        ${OPT_FLAGS:-<none>} ${COMPRESS_FLAG:-}"
    echo ""
}

clear_spirv_cache() {
    echo -e "${BOLD}Clearing SPIR-V / comgr JIT caches...${NC}"
    rm -rf "${HOME}/.cache/comgr" /tmp/comgr-*
    echo "  Removed ~/.cache/comgr/ and /tmp/comgr-*"
    echo ""
}

compile_fat() {
    local arch_flags=""
    for target in "${GFX_TARGETS[@]}"; do
        arch_flags+=" --offload-arch=$target"
    done
    echo -e "${BOLD}Compiling fat binary (${GFX_TARGETS[*]})...${NC}"
    echo "  clang++ -x hip $OPT_FLAGS $COMPRESS_FLAG ${arch_flags} ..."
    local start end
    start=$(date +%s%N)
    # shellcheck disable=SC2086
    $CLANGXX -x hip $OPT_FLAGS $COMPRESS_FLAG $arch_flags "$SRC" -o "$BIN_FAT"
    end=$(date +%s%N)
    COMPILE_FAT_MS=$(( (end - start) / 1000000 ))
    echo "  Compile time: ${COMPILE_FAT_MS} ms"
}

compile_spirv() {
    echo -e "${BOLD}Compiling SPIR-V (--offload-arch=amdgcnspirv)...${NC}"
    local start end
    start=$(date +%s%N)
    # shellcheck disable=SC2086
    $CLANGXX -x hip $OPT_FLAGS --offload-arch=amdgcnspirv $COMPRESS_FLAG "$SRC_SPIRV" -o "$BIN_SPIRV"
    end=$(date +%s%N)
    COMPILE_SPIRV_MS=$(( (end - start) / 1000000 ))
    echo "  Compile time: ${COMPILE_SPIRV_MS} ms"
}

inspect_binary() {
    local label="$1"
    local bin="$2"
    local size
    size=$(stat --printf="%s" "$bin")
    BIN_SIZES["$label"]=$size
    echo "  $label: $(numfmt --to=iec "$size") ($size bytes)"
}

inspect_all_binaries() {
    echo ""
    echo -e "${BOLD}Binary sizes:${NC}"
    inspect_binary "fat (${GFX_TARGETS[*]})" "$BIN_FAT"
    inspect_binary "amdgcnspirv" "$BIN_SPIRV"
}

run_benchmark() {
    local bin="$1"
    local label="$2"
    local -n times_ref="$3"
    times_ref=()

    echo -e "${BOLD}Running $label ($RUNS runs):${NC}"
    for ((i = 1; i <= RUNS; i++)); do
        local start end elapsed
        start=$(date +%s%N)
        "$bin" > /dev/null 2>&1
        end=$(date +%s%N)
        elapsed=$(( (end - start) / 1000000 ))
        times_ref+=("$elapsed")
        echo "  Run $i: ${elapsed} ms"
    done
}

compute_stats() {
    local -n arr="$1"
    local sum=0 min=999999999 max=0
    for t in "${arr[@]}"; do
        sum=$((sum + t))
        ((t < min)) && min=$t
        ((t > max)) && max=$t
    done
    local count=${#arr[@]}
    local avg=$((sum / count))
    local first="${arr[0]}"
    local sum_rest=0
    for ((i = 1; i < count; i++)); do
        sum_rest=$((sum_rest + arr[i]))
    done
    local avg_rest=0
    if ((count > 1)); then
        avg_rest=$((sum_rest / (count - 1)))
    fi
    echo "$avg $min $max $first $avg_rest"
}

append_results_csv() {
    [[ -z "${BENCHMARK_RESULTS_CSV:-}" ]] && return 0
    local targets="${GFX_TARGETS[*]}"
    targets="${targets// /;}"
    printf '%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$BUILD_MODE" "$targets" \
        "$COMPILE_FAT_MS" "$COMPILE_SPIRV_MS" \
        "${BIN_SIZES[$fat_label]}" "${BIN_SIZES[amdgcnspirv]}" \
        "${FIRST[fat]}" "${FIRST[spirv]}" \
        "${AVG_REST[fat]}" "${AVG_REST[spirv]}" \
        >> "$BENCHMARK_RESULTS_CSV"
}

print_summary() {
    declare -A AVG MIN MAX FIRST AVG_REST
    read -r AVG[fat] MIN[fat] MAX[fat] FIRST[fat] AVG_REST[fat] \
        <<< "$(compute_stats FAT_TIMES)"
    read -r AVG[spirv] MIN[spirv] MAX[spirv] FIRST[spirv] AVG_REST[spirv] \
        <<< "$(compute_stats SPIRV_TIMES)"

    fat_label="fat (${GFX_TARGETS[*]})"

    separator
    echo -e "${BOLD}Summary  [BUILD_MODE=$BUILD_MODE]${NC}"
    separator

    printf "%-25s %18s %14s\n" "" "$fat_label" "amdgcnspirv"
    echo "-----------------------------------------------------------"
    printf "%-25s %15s ms %11s ms\n" "Compile time"       "$COMPILE_FAT_MS"          "$COMPILE_SPIRV_MS"
    printf "%-25s %15s B  %11s B\n"  "Binary size"        "${BIN_SIZES[$fat_label]}" "${BIN_SIZES[amdgcnspirv]}"
    printf "%-25s %15s ms %11s ms\n" "First run (cold)"   "${FIRST[fat]}"            "${FIRST[spirv]}"
    printf "%-25s %15s ms %11s ms\n" "Avg (all $RUNS runs)" "${AVG[fat]}"              "${AVG[spirv]}"
    if ((RUNS > 1)); then
        printf "%-25s %15s ms %11s ms\n" "Avg (excl. first run)" "${AVG_REST[fat]}"   "${AVG_REST[spirv]}"
    fi
    separator

    if ((RUNS > 1)); then
        local jit_overhead=$((FIRST[spirv] - AVG_REST[spirv]))
        echo ""
        echo -e "${BOLD}Notes:${NC}"
        echo "  - SPIR-V first-run JIT overhead: ~${jit_overhead} ms"
        echo "  - Steady-state: fat ${AVG_REST[fat]} ms  vs  SPIR-V ${AVG_REST[spirv]} ms"
    fi

    append_results_csv
}

main() {
    separator
    echo -e "${BOLD}HIP Benchmark: Fat (${GFX_TARGETS[*]}) vs SPIR-V${NC}"
    separator
    echo ""

    check_compiler
    declare -A BIN_SIZES

    compile_fat
    compile_spirv
    inspect_all_binaries

    echo ""
    separator
    echo -e "${BOLD}Runtime Performance${NC}"
    separator
    echo ""

    clear_spirv_cache

    declare -a FAT_TIMES SPIRV_TIMES
    run_benchmark "$BIN_FAT" "Fat binary (${GFX_TARGETS[*]})" FAT_TIMES
    echo ""
    run_benchmark "$BIN_SPIRV" "SPIR-V (amdgcnspirv)" SPIRV_TIMES

    echo ""
    print_summary

    rm -f "$BIN_FAT" "$BIN_SPIRV"
}

main
