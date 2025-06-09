concurrency_values=(1 2 4 8 16 32 64 128 256)

for concurrency in "${concurrency_values[@]}"; do
    python3 -m sglang.bench_serving \
        --dataset-name random \
        --random-range-ratio 1 \
        --num-prompt 256 \
        --random-input 3200 \
        --random-output 800 \
        --max-concurrency "${concurrency}"
done
