#!/usr/bin/env bash
#
# run_singlekernel_scaling.sh — sweep amdgcnspirv + 1/2/4/8/15 native targets.
#
# Usage:
#   ./run_singlekernel_scaling.sh
#   BUILD_MODE=default ./run_singlekernel_scaling.sh
#   BUILD_MODE=O3compress TRIALS=5 ./run_singlekernel_scaling.sh
#
# Output: artifacts/_singlekernel_scaling/data.csv
#   (run_all_build_modes.sh renames to data_<BUILD_MODE>.csv)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=build_mode.inc.sh
source "$SCRIPT_DIR/build_mode.inc.sh"

TRIALS="${TRIALS:-3}"

CLANGXX="${CLANGXX:-/opt/rocm/llvm/bin/clang++}"
[[ ! -x "$CLANGXX" ]] && CLANGXX="clang++"

OUT_DIR="artifacts/_singlekernel_scaling"
mkdir -p "$OUT_DIR"
OUT_CSV="$OUT_DIR/data.csv"

SRC_FAT="custom_kernel_standard.hip"
SRC_SPV="custom_kernel_spirv.hip"
BIN="/tmp/vecadd_scaling_$$"
trap 'rm -f "$BIN"' EXIT

declare -a ROWS=(
    "0|amdgcnspirv|amdgcnspirv"
    "1|gfx950|gfx950"
    "2|gfx942 gfx950|gfx942 gfx950"
    "4|gfx942 gfx950 gfx1100 gfx1101|gfx942 gfx950 gfx1100 gfx1101"
    "8|gfx906 gfx908 gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1201|gfx906 gfx908 gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1201"
    "15|gfx900 gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201|gfx900 gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201"
)

median_ms() {
    local -a nums sorted
    mapfile -t nums
    local n=${#nums[@]}
    (( n == 0 )) && { echo 0; return; }
    mapfile -t sorted < <(printf '%s\n' "${nums[@]}" | sort -n)
    local half=$(( n / 2 ))
    if (( n % 2 == 1 )); then
        echo "${sorted[$half]}"
    else
        echo $(( (sorted[half-1] + sorted[half]) / 2 ))
    fi
}

compile_one() {
    local arch_list="$1" src="$2" flags=""
    for a in $arch_list; do flags+=" --offload-arch=$a"; done
    local start end
    start=$(date +%s%N)
    # shellcheck disable=SC2086
    $CLANGXX -x hip $OPT_FLAGS $COMPRESS_FLAG $flags "$src" -o "$BIN" 2>/dev/null
    end=$(date +%s%N)
    echo $(( (end - start) / 1000000 ))
}

echo "compiler:    $CLANGXX"
echo "BUILD_MODE:  $BUILD_MODE ($BUILD_MODE_LABEL)"
echo "flags:       ${OPT_FLAGS:-<none>} ${COMPRESS_FLAG:-}"
echo "trials:      $TRIALS"
echo "writing:     $OUT_CSV"
echo
printf "%-4s %-66s %12s %12s\n" "n" "arch_list" "compile_ms" "bin_bytes"
printf -- '-%.0s' {1..100}; echo

printf "mode,n,arch_list,compile_ms,bin_bytes\n" > "$OUT_CSV"

for row in "${ROWS[@]}"; do
    IFS="|" read -r n arch_list_csv arch_list <<< "$row"
    if [[ "$n" == "0" ]]; then
        src="$SRC_SPV"
    else
        src="$SRC_FAT"
    fi

    compile_ms=$(
        for ((i = 0; i < TRIALS; i++)); do
            compile_one "$arch_list" "$src"
        done | median_ms
    )
    bin_bytes=$(stat --printf="%s" "$BIN")

    printf "%-4s %-66s %12s %12s\n" "$n" "$arch_list_csv" "$compile_ms" "$bin_bytes"
    printf '%s,%s,"%s",%s,%s\n' "$BUILD_MODE" "$n" "$arch_list_csv" "$compile_ms" "$bin_bytes" >> "$OUT_CSV"
done

echo
echo "wrote $OUT_CSV (BUILD_MODE=$BUILD_MODE)"
