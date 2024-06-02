import torch
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import GPT2Tokenizer
from torch_int.models.opt import Int8OPTForCausalLM
import sys
import os
import gc
from torch.nn.functional import pad

class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        count = 0
        for batch in self.dataset:
            print("count: ", count)
            count = count + 1
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            # print("input_ids: ", input_ids)
            label = input_ids[:, -1]
            print("label: ", label)
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            # print("outputs: ", outputs)
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            print("pred: ", pred)
            total += label.size(0)
            hit += (pred == label).sum().item()
            # if (count == 1):
            #     print("exit")
            #     sys.exit(0)

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy


def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))


from datasets import load_dataset
fp16_model_id = 'facebook/opt-13b'
tokenizer = GPT2Tokenizer.from_pretrained(fp16_model_id)
dataset = load_dataset('lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer)

# fp16_model = OPTForCausalLM.from_pretrained(fp16_model_id, torch_dtype=torch.float16, device_map='auto')
# print_model_size(fp16_model)
# acc_fp16, lantecy_fp16 = evaluator.evaluate(fp16_model)
# print(f'FP16 accuracy: {acc_fp16}, per-sample lantecy: {lantecy_fp16:.3f}ms')
# del fp16_model
# gc.collect()
# torch.cuda.empty_cache()

model_smoothquant = Int8OPTForCausalLM.from_pretrained('mit-han-lab/opt-13b-smoothquant', torch_dtype=torch.float16, device_map='auto')
print_model_size(model_smoothquant)
acc_smoothquant, lantecy_smoothquant = evaluator.evaluate(model_smoothquant)
print(f'SmoothQuant INT8 accuracy: {acc_smoothquant}, per-sample lantecy: {lantecy_smoothquant:.3f}ms')
