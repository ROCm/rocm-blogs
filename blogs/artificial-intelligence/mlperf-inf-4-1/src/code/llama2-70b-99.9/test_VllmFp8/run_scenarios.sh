#!/bin/bash

set -xeu

export PACKAGE_DRAFT_DIR=${PACKAGE_DRAFT_DIR:-/lab-hist/package-draft}
export SYSTEM_NAME=${SYSTEM_NAME:-TEST-MI300X}
export NUM_ITERS=${NUM_ITERS:-1}
export SUBMISSION=${SUBMISSION:-1}
export TS_START_BENCHMARKS=${TS_START_BENCHMARKS:-`date +%m%d-%H%M%S`}
export RESULTS_DIR=${LAB_CLOG}/${TS_START_BENCHMARKS}
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"

echo "Running Offline"
./run_tests_Offline.sh
echo "Done Offline"

echo "Running Server"
./run_tests_Server.sh
echo "Done Server"

echo "Done Benchmarks"
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"

echo "Packaging and checking submission results"
python ../submission/package_submission.py \
    --base-package-dir ${PACKAGE_DRAFT_DIR} \
    --system-name ${SYSTEM_NAME} \
    --input-dir ${RESULTS_DIR}