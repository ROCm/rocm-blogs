#!/usr/bin/env bash
# Builds the FP8 GEMM benchmark.
#
# Usage:
#   bash build.sh                       # defaults to gfx1201
#   bash build.sh gfx1200
#   ROCM_PATH=/opt/rocm-7.0 bash build.sh
#
# Then:
#   ./gemm_fp8
set -euo pipefail

ARCH="${1:-gfx1201}"
CXX="${CXX:-amdclang++}"
ROCM="${ROCM_PATH:-/opt/rocm}"

case "${ARCH}" in
    gfx12*) ;;
    *) echo "error: FP8 WMMA requires gfx12 (RDNA 4); got ${ARCH}" >&2; exit 1 ;;
esac

CXXFLAGS="-O3 -std=c++17 --offload-arch=${ARCH} -mcumode -I${ROCM}/include"
LDFLAGS="-L${ROCM}/lib -lhipblaslt -Wl,-rpath,${ROCM}/lib"

SOURCES=(
    kernel.hip
    main.hip
)

OBJECTS=()
for source in "${SOURCES[@]}"; do
    object="${source%.hip}.o"
    echo "CXX  ${source}"
    ${CXX} ${CXXFLAGS} -c "${source}" -o "${object}"
    OBJECTS+=("${object}")
done

# --hip-link pulls in the HIP runtime; a plain link over objects does not do this the way
# a single-step compile-and-link of a .hip file does.
echo "LINK gemm_fp8"
${CXX} ${CXXFLAGS} --hip-link "${OBJECTS[@]}" -o gemm_fp8 ${LDFLAGS}

echo "Done. Run: ./gemm_fp8"
