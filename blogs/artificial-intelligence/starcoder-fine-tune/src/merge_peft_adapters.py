from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hf_user_id", type=str, default=None)

    return parser.parse_args()

def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16 
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model_version = args.peft_model_path.strip().split('/')[-1]
    model_name = args.base_model_name_or_path.strip().split('/')[-1]

    model.save_pretrained(f"{model_name}-{model_version}-merged", push_to_hub=args.push_to_hub)
    tokenizer.save_pretrained(f"{model_name}-{model_version}-merged", push_to_hub=args.push_to_hub)
    print(f"Model saved to {model_name}-{model_version}-merged")
    if args.push_to_hub:
        print("Model pushed to your Hugging Face account.")

if __name__ == "__main__" :
    main()
