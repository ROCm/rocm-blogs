# Ensure model path is passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

mkdir -p results

QPS="inf"


Req_In_Out=("1:128:2048" "2:128:2048" "4:128:2048" "8:128:2048"
"16:128:2048" "32:128:2048" "64:128:2048" "128:128:2048"
"256:128:2048" "1:2048:2048" "2:2048:2048" "4:2048:2048" "8:2048:2048"
"16:2048:2048" "32:2048:2048" "64:2048:2048" "128:2048:2048"
"256:2048:2048" "1:2048:128" "2:2048:128" "4:2048:128" "8:2048:128"
"16:2048:128" "32:2048:128" "64:2048:128" "128:2048:128"
"256:2048:128")

models="$1"

for model in "${models[@]}"; do
    modelname=$(basename  "$model")
    for req_in_out in "${Req_In_Out[@]}"; do
        con=$(echo "$req_in_out" | awk -F':' '{ print $1 }')
        inp=$(echo "$req_in_out" | awk -F':' '{ print $2 }')
        out=$(echo "$req_in_out" | awk -F':' '{ print $3 }')
        for qps in $QPS; do
            echo "[INFO] model=$model req=256 inp=$inp out=$out con=$con qps=$qps"
            python3 /app/vllm/benchmarks/benchmark_serving.py \
            --backend vllm \
            --model "$model" \
            --dataset-name random \
            --num-prompts 256 \
            --random-input-len "$inp" \
            --random-output-len "$out" \
            --random-range-ratio 1.0 \
            --ignore-eos \
            --max-concurrency "$con" \
            --port 8000 \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result \
            --result-dir results/ \
            --result-filename "${modelname}_${tp}_req256_i${inp}_o${out}_c${con}_q${qps}.json" \
            --request-rate "$qps"
        done
    done
done
