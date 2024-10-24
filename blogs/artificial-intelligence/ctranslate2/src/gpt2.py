import ctranslate2
import transformers

generator = ctranslate2.Generator("gpt2_ct2", device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# Unconditional generation.
start_tokens = [tokenizer.bos_token]
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))

# Conditional generation.
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("It is"))
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))