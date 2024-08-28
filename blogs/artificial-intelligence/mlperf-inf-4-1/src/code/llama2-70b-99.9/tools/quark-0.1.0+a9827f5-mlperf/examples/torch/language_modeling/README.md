# Language Model Quantization using Quark

This document provides examples of quantizing and exporting the language models(OPT, Llama...) using Quark. We offer several scripts for various quantization strategies. If users wish to apply their own **Customer Settings** for the `calibration method`, `dataset`, or `quant config`, they can refer to the [User Guide](./../../../docs/source/md_sources/user_guide.md) for help.

## Models supported

| Model Family | Model Name              |  FP16    | BFP16   | FP8(+FP8_KV_CACHE) | W_UIN4(Per group)+A_BF16(+AWQ/GPTQ) | INT8    | SmoothQuant | FP8 SafeTensors Export | INT8 ONNX Export |
| ------------ | ----------------------- |  ------- | ------- | ------------------ | ----------------------------------- | ------- | ----------- | ---------------------- | ---------------- |
| LLAMA 2      | meta-llama/Llama-2-*-hf |  &check; | &check; | &check;            | &check;                             | &check; | &check;     | &check;                | &check;          |
| LLAMA 3      | meta-llama/Llama-3-*-hf |  &check; | &check; | &check;            | &check;                             | &check; | &check;     | &check;                | &check;          |
| OPT          | facebook/opt-*          |  &check; | &check; | &check;            | &check;                             | &check; | &check;     | &times;                | &check;          |
| Qwen 1.5     | Qwen/Qwen1.5-*          |  &check; | &check; | &check;            | &check;                             | &check; | &check;     | &times;                | &check;          |

Note: `*` represents different model sizes, such as `7b`.

## Preparation

For Llama2 models, download the HF Llama2 checkpoint. The Llama2 models checkpoint can be accessed by submitting a permission request to Meta. For additional details, see the [Llama2 page on Huggingface](https://huggingface.co/docs/transformers/main/en/model_doc/llama2). Upon obtaining permission, download the checkpoint to the `[llama2_checkpoint_folder]`.

## Quantization & Export Scripts

You can run the following python scripts in the `examples/torch/language_modeling` path. Here we use Llama2-7b as an example.

### **Recipe 1: Evaluation of Llama2 float16 model without quantization**

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --skip_quantization

Llama2-7b perplexity with wikitext dataset (on A100 GPU): 5.4720

### **Recipe 2: Quantization of Llama2 with AWQ (W_uint4 A_float16 per_group asymmetric)**

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_uint4_per_group_asym \
                              --num_calib_data 128 \
                              --quant_algo awq \
                              --dataset pileval_for_awq_benchmark \
                              --seq_len 512

Llama2-7b perplexity with wikitext dataset (on A100 GPU): 5.6168

### **Recipe 3: Quantization of & vLLM_Adopt_SafeTensors_Export Llama2 with W_int4 A_float16 per_group symmetric**

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_int4_per_group_sym \
                              --num_calib_data 128 \
                              --model_export vllm_adopted_safetensors

If the code runs successfully, it will produce one JSON file and one .safetensor file in `[output_dir]` and the terminal will display `[Quark] Quantized model exported to ... successfully.`
Llama2-7b perplexity with wikitext dataset (on A100 GPU): 5.7912

### **Recipe 4: Quantization & vLLM_Adopt_SafeTensors_Export of W_FP8_A_FP8 with FP8 KV cache**

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_fp8_a_fp8 \
                              --kv_cache_dtype fp8 \
                              --num_calib_data 128 \
                              --model_export vllm_adopted_safetensors

If the code runs successfully, it will produce one JSON file and one .safetensor file in `[output_dir]` and the terminal will display `[Quark] Quantized model exported to ... successfully.`

Llama2-7b perplexity with wikitext dataset (on A100 GPU): 5.5133

### **Recipe 5: Quantization & vLLM_Adopt_SafeTensors_Export of only W_FP8_A_FP8**

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_fp8_a_fp8 \
                              --num_calib_data 128 \
                              --model_export vllm_adopted_safetensors

If the code runs successfully, it will produce one JSON file and one .safetensor file in `[output_dir]` and the terminal will display `[Quark] Quantized model exported to ... successfully.`

Llama2-7b perplexity with wikitext dataset (on A100 GPU): 5.5093

### **Recipe 6: Quantization & vLLM_Adopt_SafeTensors_Export of W_FP8_A_FP8_O_FP8**

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_fp8_a_fp8_o_fp8 \
                              --num_calib_data 128 \
                              --model_export vllm_adopted_safetensors

If the code runs successfully, it will produce one JSON file and one .safetensor file in `[output_dir]` and the terminal will display `[Quark] Quantized model exported to ... successfully.`

Llama2-7b perplexity with wikitext dataset (on A100 GPU): 5.5487

### **Recipe 7: Quantization & vLLM_Adopt_SafeTensors_Export of W_FP8_A_FP8_O_FP8 without weight scaling factors of gate_proj and up_proj merged** And if option "--no_weight_matrix_merge" is not set, weight scaling factors of gate_proj and up_proj are merged

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_fp8_a_fp8_o_fp8 \
                              --num_calib_data 128 \
                              --model_export vllm_adopted_safetensors \
                              --no_weight_matrix_merge

If the code runs successfully, it will produce one JSON file and one .safetensor file in `[output_dir]` and the terminal will display `[Quark] Quantized model exported to ... successfully.`

### **Recipe 8: Quantization & Torch compile of W_INT8_A_INT8_PER_TENSOR_SYM**

    python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_int8_a_int8_per_tensor_sym \
                              --num_calib_data 128 \
                              --device cpu \
                              --data_type bfloat16 \
                              --model_export torch_compile

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
