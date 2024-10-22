#!/bin/bash

set -xeu

NUM_ITERS=${NUM_ITERS:-1}
export SUBMISSION=${SUBMISSION:-1}
export SCENARIO="Offline"
export TS_START_BENCHMARKS=${TS_START_BENCHMARKS:-`date +%m%d-%H%M%S`}
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"


for i in $(seq 1 ${NUM_ITERS})
do
        echo "Running $SCENARIO - Performance run $i/$NUM_ITERS"
        ITER=$i bash test_VllmFp8_Offline_perf.sh
done
echo "Running $SCENARIO - Accuracy"
bash test_VllmFp8_Offline_acc.sh
echo "Running $SCENARIO - Audit"
bash test_VllmFp8_Offline_audit.sh
echo "Done"
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"
