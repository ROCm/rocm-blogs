#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import platform
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from typing import Optional, Tuple
from dataclasses import replace
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.torch.quantization.config.config import QuantizationConfig, Config
from quark.torch.quantization.config.custom_config import FP8_PER_TENSOR_SPEC, \
                                                        DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG, \
                                                        DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG, \
                                                        DEFAULT_W_INT4_PER_TENSOR_CONFIG, \
                                                        DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG, \
                                                        DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG, \
                                                        DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG, \
                                                        DEFAULT_W_INT4_PER_CHANNEL_CONFIG, \
                                                        DEFAULT_AWQ_CONFIG, \
                                                        DEFAULT_SMOOTH_QUANT_CONFIG, \
                                                        DEFAULT_GPTQ_CONFIG, \
                                                        DEFAULT_W_UINT4_PER_GROUP_CONFIG, \
                                                        DEFAULT_FLOAT16_CONFIG, \
                                                        DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG

from data_preparation import get_calib_dataloader


def get_config(args: argparse.Namespace, model_type: str) -> Config:

    quant_config = None
    if args.quant_scheme == 'w_fp8_a_fp8':
        quant_config = Config(global_quant_config=DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG)
    elif args.quant_scheme == 'w_fp8_a_fp8_o_fp8':
        quant_config = Config(global_quant_config=DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG)
    elif args.quant_scheme == 'w_int4_per_tensor':
        quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_TENSOR_CONFIG)
    elif args.quant_scheme == 'w_int4_per_channel_sym':
        quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_CHANNEL_CONFIG, )
    elif args.quant_scheme == 'w_int4_per_group_sym':
        quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG, )
    elif args.quant_scheme == 'w_uint4_per_group_asym':
        if args.quant_algo == 'awq':
            quant_config = DEFAULT_AWQ_CONFIG
            algo_config_file = 'models/' + model_type + '/awq_config.json'
            with open(algo_config_file, 'r') as file:
                algo_config_info = json.load(file)
            assert quant_config.algo_config is not None and hasattr(quant_config.algo_config, 'scaling_layers')
            quant_config.algo_config.scaling_layers = algo_config_info['scaling_layers']
            quant_config.algo_config.model_decoder_layers = algo_config_info['model_decoder_layers']
            quant_config.algo_config.embedding_layers = algo_config_info['embedding_layers']
        elif args.quant_algo == 'smoothquant':
            quant_config = DEFAULT_SMOOTH_QUANT_CONFIG
            algo_config_file = 'models/' + model_type + '/awq_config.json'
            with open(algo_config_file, 'r') as file:
                algo_config_info = json.load(file)
            assert quant_config.algo_config is not None and hasattr(quant_config.algo_config, 'scaling_layers')
            quant_config.algo_config.scaling_layers = algo_config_info['scaling_layers']
            quant_config.algo_config.model_decoder_layers = algo_config_info['model_decoder_layers']
            quant_config.algo_config.embedding_layers = algo_config_info['embedding_layers']
        elif args.quant_algo == 'gptq':
            quant_config = DEFAULT_GPTQ_CONFIG
            algo_config_file = 'models/' + model_type + '/gptq_config.json'
            with open(algo_config_file, 'r') as file:
                algo_config_info = json.load(file)
            assert quant_config.algo_config is not None and hasattr(quant_config.algo_config, 'inside_layer_modules')
            quant_config.algo_config.inside_layer_modules = algo_config_info['inside_layer_modules']
            quant_config.algo_config.model_decoder_layers = algo_config_info['model_decoder_layers']
            quant_config.algo_config.embedding_layers = algo_config_info['embedding_layers']
        else:
            quant_config = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG)
    elif args.quant_scheme == 'w_uint4_a_bfloat16_per_group_asym':
        quant_config = Config(global_quant_config=DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG)
    elif args.quant_scheme == 'w_int8_a_int8_per_tensor_sym':
        quant_config = Config(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG)
    elif args.quant_scheme == 'w_int8_a_int8_per_tensor_sym_dynamic':
        quant_config = Config(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG)
    else:
        quant_config = Config(global_quant_config=DEFAULT_FLOAT16_CONFIG)
    if args.kv_cache_dtype is not None:
        if args.kv_cache_dtype == "fp8":
            KV_CACHE_CFG = {
                "*.v_proj":
                QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                   output_tensors=FP8_PER_TENSOR_SPEC,
                                   weight=FP8_PER_TENSOR_SPEC),
                "*.k_proj":
                QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                   output_tensors=FP8_PER_TENSOR_SPEC,
                                   weight=FP8_PER_TENSOR_SPEC),
            }
            quant_config = replace(quant_config, layer_quant_config=KV_CACHE_CFG)

    if model_type in ["llama", 'opt', 'qwen2']:
        EXCLUDE_LAYERS = ["lm_head"]
        quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    # Args checking
    if args.quant_algo is not None and args.quant_scheme != 'w_uint4_per_group_asym':
        raise ValueError(f"{args.quant_algo} onlyNone support quant_schema as w_uint4_per_group_asym")
    if args.quant_algo is not None and model_type is None:
        raise ValueError(f"{args.quant_algo} not support for current model")
    return quant_config


def get_tokenizer(ckpt_path: str, max_seq_len: int = 2048, model_type: Optional[str] = None) -> AutoTokenizer:
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,
                                              model_max_length=max_seq_len,
                                              padding_side="left",
                                              trust_remote_code=True,
                                              use_fast=False)
    if model_type and model_type == "qwen2":
        # qwen2 use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def get_model(ckpt_path: str, data_type: str = 'auto', device: str = "cuda") -> Tuple[nn.Module, torch.dtype]:

    if data_type == 'float16':
        model_dtype = torch.float16
    elif data_type == 'bfloat16':
        model_dtype = torch.bfloat16
    elif data_type == 'float32':
        model_dtype = torch.float32
    elif data_type == 'auto':
        model_dtype = data_type
    else:
        raise ValueError(f"{data_type} not support for current model")
    device = "auto" 
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device, torch_dtype=model_dtype, trust_remote_code=True)
    model.eval()
    model_dtype = next(model.parameters()).dtype

    return model, model_dtype


def get_model_type(model: nn.Module) -> str:
    MODEL_NAME_PATTERN_MAP = {
        "Llama": "llama",
        "Falcon": 'falcon',
        "Mistral": "llama",
        "OPT": "opt",
        "QWen2": 'qwen2',
        "Gemma": 'gemma',
        "ChatGLM": 'chatglm'
    }
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    raise NotImplementedError(
        f"This example script may not support the current model type. Please configure as necessary. The supported models include: {', '.join([i for i in MODEL_NAME_PATTERN_MAP.keys()])}."
    )


@torch.no_grad()
def ppl_eval(model: nn.Module, testenc: AutoTokenizer, dev: str) -> None:
    # Set sequence length as 2048 for wikitext dataset evaluation
    seqlen_for_eval = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen_for_eval

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen_for_eval):((i + 1) * seqlen_for_eval)].to(dev)
        lm_logits = model(batch)['logits']
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen_for_eval):((i + 1) * seqlen_for_eval)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen_for_eval
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))
    print("\nPerplexity: {}".format(ppl.item()))


def main(args: argparse.Namespace) -> None:
    # 1. Define original model
    print("\nLoading model ...")
    model, model_dtype = get_model(args.model_dir, args.data_type, args.device)
    model_type = get_model_type(model)
    tokenizer = get_tokenizer(args.model_dir, max_seq_len=args.seq_len, model_type=model_type)

    # 2. Set quantization configuration
    if not args.skip_quantization:
        quant_config = get_config(args, model_type)

    # 3. Define calibration dataloader(still need this step for weight only and dynamic quantization)
    print("\nLoading dataset ...")
    calib_dataloader = get_calib_dataloader(dataset_name=args.dataset,
                                            tokenizer=tokenizer,
                                            batch_size=args.batch_size,
                                            num_calib_data=args.num_calib_data,
                                            seqlen=args.seq_len,
                                            device=args.device)

    # 4. In-place replacement with quantized modules in model
    if not args.skip_quantization:
        from quark.torch import ModelQuantizer
        quantizer = ModelQuantizer(quant_config)
        model = quantizer.quantize_model(model, calib_dataloader)

    # 5. (Optional) Model exporting
    if args.model_export is not None:
        # If user want to export the quantized model, please freeze the quantized model first
        model = quantizer.freeze(model)
    # Export option 1: .json and .safetensors
    if args.model_export == "vllm_adopted_safetensors":
        print("\nExporting json and safetensors...")
        with torch.inference_mode():
            export_path = args.output_dir
            from quark.torch import ModelExporter
            from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG, EMPTY_EXPORTER_CONFIG
            config = EMPTY_EXPORTER_CONFIG if args.no_weight_matrix_merge else DEFAULT_EXPORTER_CONFIG
            exporter = ModelExporter(config=config, export_dir=export_path)
            exporter.export_model_info(model, model_type, model_dtype, export_type="vllm-adopt")
    # Export option 2: onnx
    elif args.model_export == "onnx":
        print("\nExporting onnx graph...")
        with torch.inference_mode():
            export_path = args.output_dir
            batch_iter = iter(calib_dataloader)
            input_args = next(batch_iter)
            if args.quant_scheme in ["w_int4_per_channel_sym", "w_uint4_per_group_asym", "w_int4_per_group_sym", "w_uint4_a_bfloat16_per_group_asym"]:
                uint4_int4_flag = True
            else:
                uint4_int4_flag = False

            from quark.torch import ModelExporter
            from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG, EMPTY_EXPORTER_CONFIG
            config = EMPTY_EXPORTER_CONFIG if args.no_weight_matrix_merge else DEFAULT_EXPORTER_CONFIG
            exporter = ModelExporter(config=config, export_dir=export_path)
            exporter.export_onnx_model(model, input_args, uint4_int4_flag=uint4_int4_flag)
    # Export option 3: torch.compile
    elif args.model_export == "torch_compile":
        print("\nCalling PyTorch 2 torch.compile...")
        # Note: The model after torch.compile may not be able to export to other format
        model = torch.compile(model)

    # 6. (Optional) Model Evaluation
    if not args.skip_evaluation:
        print("\nEvaluating ...")
        if args.eval_metric == "ppl":
            testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

            if platform.system() == "Windows":
                print("testdata use subset 200 on windows")
                testenc = tokenizer("\n\n".join(testdata['text'][:200]), return_tensors='pt')
            else:
                testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
            if (args.use_ppl_eval_for_kv_cache):
                from ppl_compute_for_kv_cache import ppl_eval_for_kv_cache
                ppl_eval_for_kv_cache(model, testenc, args.ppl_eval_for_kv_cache_context_size,
                                      args.ppl_eval_for_kv_cache_sample_size, args.ppl_eval_for_kv_cache_patch_size,
                                      args.device)
            else:
                ppl_eval(model, testenc, args.device)
        elif args.eval_metric == "rouge":
            from rouge_evaluate import rouge_eval
            rouge_eval(model, args.rouge_eval_pkl_path, tokenizer, device="cuda", batch_size=args.eval_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        help="Specify where the HuggingFace model is. This example support Llama, OPT models",
                        required=True)
    parser.add_argument("--dataset",
                        help="Dataset for calibration",
                        default="pileval")
                        # default="pileval",
                        # choices=["pileval", "wikitext", "pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark"])
    parser.add_argument("--device", help="Device for running the quantizer", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--data_type", help="Datatype of the model", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
    parser.add_argument("--skip_quantization", action='store_true')
    parser.add_argument("--skip_evaluation", action='store_true')
    parser.add_argument("--eval_metric", help="Model evaluate metric", default="ppl", choices=["ppl", "rouge"])
    parser.add_argument("--rouge_eval_pkl_path", help="Pickle dataset path for rouge evaluation")

    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument("--eval_batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration.", type=int, default=512)
    parser.add_argument("--output_dir", default="exported_model")
    parser.add_argument("--no_weight_matrix_merge", help="Whether to merge weight matrix when dump vllm-adopt quantized model", action='store_true')

    parser.add_argument("--quant_scheme",
                        help="Supported quant_scheme in the script. \
                            If there is no suitable quantization strategy among the options, \
                            users can customize the quantization configuration according to their own needs. \
                            If None, the model will be quantized by float16",
                        default=None,
                        choices=[
                            "w_fp8_a_fp8", "w_int4_per_channel_sym", "w_uint4_per_group_asym", "w_int4_per_group_sym",
                            "w_uint4_a_bfloat16_per_group_asym", "w_int8_a_int8_per_tensor_sym",
                            "w_int8_a_int8_per_tensor_sym_dynamic", "w_fp8_a_fp8_o_fp8", None
                        ])
    parser.add_argument("--kv_cache_dtype", help="KV Cache dtype.", default=None, choices=["fp8", None])
    parser.add_argument("--quant_algo",
                        help="Quantization Algorithms.",
                        default=None,
                        choices=["awq", "smoothquant", "gptq", None])

    parser.add_argument("--model_export", help="Model export format", default=None, choices=[None, "torch_compile", "onnx", "vllm_adopted_safetensors"])

    parser.add_argument("--use_ppl_eval_for_kv_cache", action='store_true')
    parser.add_argument("--ppl_eval_for_kv_cache_context_size",
                        type=int,
                        help="Context size used in PPL evaluation for KV cache.",
                        default=1024)
    parser.add_argument("--ppl_eval_for_kv_cache_sample_size",
                        type=int,
                        help="Sample size used in PPL evaluation for KV cache.",
                        default=512)
    parser.add_argument("--ppl_eval_for_kv_cache_patch_size",
                        type=int,
                        help="Patch size used in PPL evaluation for KV cache.",
                        default=None)

    args = parser.parse_args()

    main(args)
