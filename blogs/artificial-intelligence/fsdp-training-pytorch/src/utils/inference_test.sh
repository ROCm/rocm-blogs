
fsdp_checkpoint_path=$1
hf_model_path=$2
model_version=$3
prompt_for_test_file=$4
python ./src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py --fsdp_checkpoint_path $fsdp_checkpoint_path --consolidated_model_path ./fsdp_fine_tune_results/fsdp_model_finetuned_1_${model_version}_hf --HF_model_path_or_name $hf_model_path 
python ./recipes/quickstart/inference/local_inference/inference.py --model_name ./fsdp_fine_tune_results/fsdp_model_finetuned_1_${model_version}_hf --prompt_file  $prompt_for_test_file
