import os
import torch
import transformers
import click


# Use Click to parse command-line arguments
@click.command
@click.option("--tune", is_flag=True)
def main(tune):
    # Set some variables
    seq_len = 256  # Max sequence length to generate
    n_batches = 8  # Number of batches to time
    n_warmup = 2  # Number of warmup batches
    prompt = ["Hello Earthlings!"]  # Input prompt

    # We can enable tuning by setting the environment variables within the code - as long as we do so before
    # using torch. This is often less cumbersome than passing the environment variables each time
    if tune:
        print("Tuning enabled")
        os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"  # Enable tuning
        os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "src/llm_result.csv"  # Specify output file

    # Retrieve the model and tokenizer
    model = "google/gemma-2b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModelForCausalLM.from_pretrained(model).to("cuda")

    # Set the model to use a static KV cache - see https://huggingface.co/docs/transformers/main/en/llm_optims?static-kv=generation_config#static-kv-cache-and-torchcompile
    model.generation_config.cache_implementation = "static"

    # Tokenize our input.
    # Use padding with `pad_to_multiple_of` to minimize the number of GEMMs to tune
    # Larger values => Less GEMMs to tune, but more potential overhead for shorter prompts
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, pad_to_multiple_of=8).to("cuda")

    # Determine how many tokens to generate. Here, we need to subtract the number of tokens in the prompt to keep the same
    # overall sequence length
    n_tokens = seq_len - inputs["input_ids"].shape[1]  # number of tokens to generate

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    for i in range(n_batches + n_warmup):
        # Don't start timing until we've finished our warmup iters
        if i == n_warmup:
            torch.cuda.synchronize()
            t0.record()

        # Generate!
        model.generate(
            **inputs,
            max_new_tokens=n_tokens,  # Force the model to generate exactly n_tokens before stopping
            min_new_tokens=n_tokens,
            use_cache=True,  # Ensure we use the kv-cache
        )

    # Complete timing, synchronize, and compute elapsed time
    t1.record()
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000

    tokens_per_second = n_batches * n_tokens / dt
    print(f"  Tokens/second: {tokens_per_second:0.4f} ({n_tokens*n_batches} tokens, {dt:0.2f} seconds)")


if __name__ == "__main__":
    main()
