# dev-README

## Set Up a Development Docker Container

Start the container using the `start.sh` script.  Refer to `code/llama2-70b-99.9/lab_setup_VllmFp8/env.sh` for relevant host-side data locations and env vars.

``` bash
cd code/llama2-70b-99.9/lab_setup_VllmFp8
./start.sh
```

In the container, install dependencies using the scripts in `lab_setup_VllmFp8/ct-scripts`.

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/lab_setup_VllmFp8/ct-scripts
./install-hipblaslt.sh
./install-deps.sh
./install-loadgen.sh
./install-rocm-bwtest.sh
./install-vllm.sh
./setup-tgemm.sh
# Only required for hyper parameter tuning
./install-optuna.sh
# Only required for Rocm Profile Data tracing
./install-rpd-trace.sh
```

## Run MLPerf Test Scenarios in Development Container

### Offline Scenario

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
./test_VllmFp8_Offline_acc.sh
./test_VllmFp8_Offline_perf.sh
```

### Server Scenario (without openAI server)

Using AsyncLLMEngine of vLLM directly.

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
./test_VllmFp8_AsyncServer_acc.sh
./test_VllmFp8_AsyncServer_perf.sh
```

## Hyper parameter tuning

- Offline scenario

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
./test_VllmFP8_Offline_Optuna_hpt.sh
```

## Profiling with Rocm Profile Data

- Offline scenario

```bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
./test_rpd_VllmFp8_Offline_perf.sh
```

- Server Scenario

```bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
./test_rpd_VllmFp8_AsyncServer_perf.sh
```

## Run MLPerf Compliance Tests

- Offline Scenario

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
./test_VllmFp8_Offline_audit.sh
```

- Server Scenario

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
./test_VllmFp8_AsyncServer_audit.sh
```

## Instructions Specific to Model Quantization

Start the quantization container using the `start.sh` script.  Refer to `code/llama2-70b-99.9/lab_setup_VllmFp8/env.sh` for relevant host-side data locations and env vars.

``` bash
cd code/llama2-70b-99.9/lab_setup_VllmFp8
./start.sh
```

In the container, install Quark.

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/tools/quark-0.1.0+a9827f5-mlperf/
pip install quark-0.1.0+a9827f5-py39-none-any.whl
```

In the container, quantize the model using the calibration dataset

``` bash
model_dir=/data/llm/llama2-70b-chat
output_dir=/data/llm/llama2-70b-chat/quantized-scratch/quark_share/modelzoo/llama2_70b_wfp8_afp8_ofp8_nomerge/json-safetensors
calib_dataset=/data/open_orca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz

cd /lab-mlperf-inference/code/llama2-70b-99.9/tools/quark-0.1.0+a9827f5-mlperf/
cd examples/torch/language_modeling/
python3 quantize_quark.py --model_dir $model_dir \
    --output_dir $output_dir \
    --quant_scheme w_fp8_a_fp8_o_fp8 \
    --dataset $calib_dataset \
    --num_calib_data 1000 \
    --model_export vllm_adopted_safetensors \
    --no_weight_matrix_merge
```

Download KV cache scales from https://github.com/vllm-project/vllm/blob/38c4b7e863570a045308af814c72f4504297222e/tests/fp8_kv/llama2-70b-fp8-kv/kv_cache_scales.json.

## Log

2024-06-24  Init repo copied from https://github.com/AMD-AI/mlperf-inference/commit/044247b352e808873653caec2ec94c902771c46a.\
2024-07-12  Move AsyncOpenAI Server implementation to archive
