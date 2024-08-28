set -e
export ROCR_VISIBLE_DEVICES=0

model_name="Llama-2-70b-chat-hf"
export MODEL_DIR="/data/models/${model_name}"
output_dir="quantized_models/${model_name}"
calib_dataset="/data/open-orca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
echo -n "--------------------------------------------\n"
echo -n "W FP8 A FP8 KV_Cache FP8"
python3 quantize_quark.py --model_dir $MODEL_DIR \
                          --output_dir $output_dir \
                          --quant_scheme w_fp8_a_fp8_o_fp8 \
                          --dataset $calib_dataset \
                          --num_calib_data 1000 \
                          --model_export vllm_adopted_safetensors \
                          --no_weight_matrix_merge

