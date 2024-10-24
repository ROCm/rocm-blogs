import ctranslate2
import sentencepiece as spm
import time

# Load the SentencePiece model
sp = spm.SentencePieceProcessor(model_file="sentencepiece.model")

# Input text to translate
input_text = "Hello world!"
input_tokens = sp.encode(input_text, out_type=str)

# Function to perform translation and measure latency and tokens per second
def translate_and_time(translator):
    start_time = time.time()
    results = translator.translate_batch([input_tokens])
    end_time = time.time()
    latency = end_time - start_time

    # Decode the translated tokens
    output_tokens = results[0].hypotheses[0]
    output_text = sp.decode(output_tokens)

    # Calculate tokens per second
    num_output_tokens = len(output_tokens)
    tokens_per_second = num_output_tokens / latency

    return output_text, latency, tokens_per_second

# Load the default (float32) model
translator_float32 = ctranslate2.Translator(
    "ende_ctranslate2/", device="cuda", compute_type="float32"
)
output_text_float32, latency_float32, tps_float32 = translate_and_time(translator_float32)

# Load the int8 quantized model
translator_int8 = ctranslate2.Translator(
    "ende_ctranslate2_int8/", device="cuda", compute_type="int8"
)
output_text_int8, latency_int8, tps_int8 = translate_and_time(translator_int8)

# Print the results
print("Default (float32) model translation:")
print(f"Output: {output_text_float32}")
print(f"Latency: {latency_float32:.4f} seconds")
print(f"Tokens per second: {tps_float32:.2f}\n")

print("Int8 quantized model translation:")
print(f"Output: {output_text_int8}")
print(f"Latency: {latency_int8:.4f} seconds")
print(f"Tokens per second: {tps_int8:.2f}\n")

# Calculate the speedup in tokens per second
speedup_tps = tps_int8 / tps_float32
print(f"Speedup in tokens per second with int8 quantization: {speedup_tps:.2f}x faster")