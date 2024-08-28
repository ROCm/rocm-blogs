
import pickle
import numpy as np
import nltk
import evaluate
from tqdm import tqdm
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.nn.functional import pad
from transformers import AutoTokenizer
import gzip


gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 1,
    "do_sample": False
}

def prepare_openorca(dataset_path: str) -> Tuple[List[List[int]], List[int], List[List[int]], List[str]]:
    try:
        with gzip.open(dataset_path, 'rb') as fh:
            pass
        open_fn = gzip.open
    except:
        open_fn = open

    # Load from OpenORCA
    with open_fn(dataset_path, 'rb') as fh:
        orca_df = pickle.load(fh)
    
    source_ids = orca_df['tok_input'].tolist()
    source_lengths = orca_df['tok_input_length'].tolist()
    target_ids = orca_df['tok_output'].tolist()
    target_texts = orca_df["output"].tolist()

    print(f"Loaded {len(source_lengths)} samples from {dataset_path}")
    return source_ids, source_lengths, target_ids, target_texts


def postprocess_text(preds: List[str], targets: List[str]) -> Tuple[List[str], List[str]]:
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def rouge_eval(model: nn.Module, dataset_path: str, tokenizer: AutoTokenizer, device: str, batch_size: int) -> None:
    metric = evaluate.load("rouge")
    nltk.download('punkt')

    source_ids, source_lengths, target_ids, target_texts = prepare_openorca(dataset_path)
    samples_num = len(source_ids)

    # start inferencing
    max_seq_len = 1024
    batch_size = 1 if batch_size is None else batch_size
    target_required = []
    preds_token_ids = []
    gen_tok_len = 0
    for idx in tqdm(range(0, samples_num, batch_size)):
        input_ids_tensor = []
        input_masks_tensor = []
        input_len = []
        for q in range(batch_size):
            input_length = source_lengths[idx + q]
            input_ids = torch.tensor(source_ids[idx + q], device=device, dtype=torch.int32).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int32, device=device)
            input_ids_tensor.append(pad(input_ids, (max_seq_len - input_length, 0, 0, 0), value=tokenizer.pad_token_id))
            input_masks_tensor.append(pad(attention_mask, (max_seq_len - input_length, 0, 0, 0), value=0))
            input_len.append(input_length)
            target_required.append(target_texts[idx + q])
        input_ids_tensor = torch.cat(input_ids_tensor)
        input_masks_tensor = torch.cat(input_masks_tensor)

        assert input_ids_tensor.shape == input_masks_tensor.shape
        assert input_ids_tensor.shape[0] <= batch_size

        pred_output_tokens = model.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    pad_token_id=tokenizer.pad_token_id,
                    **gen_kwargs
                )
        pred_tokens = pred_output_tokens[:, max_seq_len:].cpu().numpy()
        pred_tokens_extend = np.zeros((pred_tokens.shape[0], pred_tokens.shape[1] * 2), dtype=pred_tokens.dtype)
        pred_tokens_extend[:, ::2] = pred_tokens
        gen_tok_len += pred_tokens_extend.size
        preds_token_ids.extend(pred_tokens_extend)

    preds_decoded_text = tokenizer.batch_decode(preds_token_ids, skip_special_tokens=True)
    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    gen_num = len(preds)

    result = {**result,
              'gen_len': np.sum(prediction_lens),
              'gen_num': gen_num,
              'gen_tok_len': gen_tok_len,
              'tokens_per_sample': round(gen_tok_len / gen_num, 1)
              }

    print("\nResults\n")
    print(result)
