"""Counts the reasoning and output tokens for the given queries and summarizes."""
import json
import requests
import argparse
from transformers import AutoTokenizer

def count_tokens(tokenizer, text):
    if text in (None, "None"):
        return 0
    return len(tokenizer.encode(str(text), add_special_tokens=False))

def run_query(prompt, model, api_url, disable_reasoning=False):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 38912,
        "chat_template_kwargs": {
            "enable_thinking": not disable_reasoning
        }
    }

    response = requests.post(api_url, json=data)
    response.raise_for_status()
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Analyze token usage from reasoning and content outputs.")
    parser.add_argument("--input", type=str, default="mult.jsonl", help="Path to JSONL input file.")
    parser.add_argument("--num-prompts", type=int, default=None, help="Limit number of prompts to run.")
    parser.add_argument("--disable-reasoning", action="store_true", help="Switch off reasoning mode in queries.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name to use.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions", help="vLLM API endpoint.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    total_reasoning_tokens = 0
    total_content_tokens = 0
    num_queries = 0

    with open(args.input, "r") as f:
        for i, line in enumerate(f):
            if args.num_prompts is not None and i >= args.num_prompts:
                break

            query = json.loads(line)
            prompt = query["prompt"]

            result = run_query(prompt, args.model, args.api_url, args.disable_reasoning)
            msg = result["choices"][0]["message"]

            reasoning_text = msg.get("reasoning_content", "")
            final_text = msg.get("content", "")
            r_tokens = count_tokens(tokenizer, reasoning_text)
            c_tokens = count_tokens(tokenizer, final_text)

            total_reasoning_tokens += r_tokens
            total_content_tokens += c_tokens
            num_queries += 1

            print(f"Prompt: {prompt}' Reasoning tokens: {r_tokens}. Final content tokens: {c_tokens}")

    print("="*40)
    print(f"Total queries: {num_queries}")
    print(f"Total reasoning tokens: {total_reasoning_tokens}")
    print(f"Total final content tokens: {total_content_tokens}")
    print(f"Average reasoning tokens: {total_reasoning_tokens/num_queries:.2f}")
    print(f"Average content tokens: {total_content_tokens/num_queries:.2f}")

if __name__ == "__main__":
    main()
