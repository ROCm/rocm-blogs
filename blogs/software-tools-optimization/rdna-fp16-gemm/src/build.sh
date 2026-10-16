#!/usr/bin/env bash
# Builds the FP16 GEMM stage progression.
#
# Each stage is a separate translation unit, so they are compiled to objects and linked
# together. CU mode is required for reproducible numbers on RDNA.
#
# Usage:
#   ./build.sh gfx1100                  # RDNA3
#   ./build.sh gfx1201                  # RDNA4
#   ROCM_PATH=/opt/rocm-7.0 ./build.sh gfx1100
#
# Then:
#   ./gemm_stages
set -euo pipefail

ARCH="${1:-gfx1201}"
CXX="${CXX:-amdclang++}"
ROCM="${ROCM_PATH:-/opt/rocm}"

CXXFLAGS="-O3 -std=c++17 --offload-arch=${ARCH} -mcumode -I${ROCM}/include"
LDFLAGS="-L${ROCM}/lib -lhipblaslt -Wl,-rpath,${ROCM}/lib"

SOURCES=(
    stage01_naive.hip
    stage02_pipeline.hip
    stage03_bankconflicts.hip
    stage04_srd.hip
    stage05_deepk.hip
    stage07_interleave.hip
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
echo "LINK gemm_stages"
${CXX} ${CXXFLAGS} --hip-link "${OBJECTS[@]}" -o gemm_stages ${LDFLAGS}

echo "Done. Run: ./gemm_stages"
