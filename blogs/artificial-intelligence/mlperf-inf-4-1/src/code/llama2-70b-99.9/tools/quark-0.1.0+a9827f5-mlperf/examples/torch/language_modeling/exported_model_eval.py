#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from typing import Tuple, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME_PATTERN_MAP = {
    "Llama": "llama",
    "Falcon": 'falcon',
    "Mistral": "llama",
    "OPT": "opt",
    "QWen2": 'qwen2',
    "Gemma": 'gemma',
    "ChatGLM": 'chatglm'
}


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


def get_model(ckpt_path: str, device: str = "cuda") -> Tuple[nn.Module, torch.dtype]:

    model_kwargs = {"torch_dtype": "auto"}
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto", **model_kwargs, trust_remote_code=True)
    model.eval()

    model_dtype = next(model.parameters()).dtype

    return model, model_dtype


def get_model_type(model: nn.Module) -> Optional[str]:
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


@torch.no_grad()
def ppl_eval(model: nn.Module, testenc: AutoTokenizer, dev: str) -> None:
    # Set sequence length as 2048 for wikitext dataset evaluation
    model.seqlen = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        lm_logits = model(batch)['logits']
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print("\nPerplexity: {}".format(ppl.item()))


def main(args: argparse.Namespace) -> None:
    print("\nLoading model ...")
    model, _ = get_model(args.model_dir, args.device)
    model_type = get_model_type(model)
    tokenizer = get_tokenizer(args.model_dir, max_seq_len=args.seq_len, model_type=model_type)

    print("\nRestore quantized model from json and safetensors file ...")
    from quark.torch import from_exported_files
    model = from_exported_files(model, args.json_dir, args.safetensors_dir)

    from quark.torch import ModelQuantizer
    model = ModelQuantizer.freeze(model)

    print("\nEvaluating ...")
    if args.eval_metric == "ppl":
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        ppl_eval(model, testenc, args.device)
    elif args.eval_metric == "rouge":
        from rouge_evaluate import rouge_eval
        rouge_eval(model, args.rouge_eval_pkl_path, tokenizer, device="cuda", batch_size=args.batch_size)

    if args.export_onnx:
        print("\nExporting onnx graph...")
        with torch.inference_mode():
            export_path = args.output_dir
            batch_iter = iter(testdata)
            input_args = next(batch_iter)

            from quark.torch import ModelExporter
            from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
            exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
            exporter.export_onnx_model(model, input_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        help="Specify where the HuggingFace model is. This example support Llama, OPT models",
                        required=True)
    parser.add_argument("--dataset",
                        help="Dataset for calibration",
                        default="pileval",
                        choices=["pileval", "wikitext", "pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark"])
    parser.add_argument("--json_dir", help="Specify the directory of export json file", required=True)
    parser.add_argument("--safetensors_dir", help="Specify the directory of export safetensors file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
    parser.add_argument("--export_onnx", action='store_true')
    parser.add_argument("--output_dir", default="exported_model")
    parser.add_argument("--eval_metric", help="Model evaluate metric", default="ppl", choices=["ppl", "rouge"])
    parser.add_argument("--rouge_eval_pkl_path", help="Pickle dataset path for rouge evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    args = parser.parse_args()

    main(args)
