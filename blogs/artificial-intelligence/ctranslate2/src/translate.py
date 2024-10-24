import ctranslate2
import sentencepiece as spm

translator = ctranslate2.Translator("ende_ctranslate2/", device="cuda")
sp = spm.SentencePieceProcessor("sentencepiece.model")

input_text = "Good Morning!"
input_tokens = sp.encode(input_text, out_type=str)

results = translator.translate_batch([input_tokens])

output_tokens = results[0].hypotheses[0]
output_text = sp.decode(output_tokens)

print(output_text)
