import datasets
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, BertConfig
from transformers import AutoTokenizer, BertConfig, BertForPreTraining, Trainer, TrainingArguments
import torch

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


from accelerate.logging import get_logger
from accelerate.utils import set_seed
import argparse


def train_func():
    set_seed(42) #To replicate results, set the seed to same number
    parser = argparse.ArgumentParser()
    parser.add_argument('--BATCH_SIZE', type=int, default = 8) # 32 is the global batch size, since I use 8 GPUs
    parser.add_argument('--EPOCHS', type=int, default=200)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dataset_file', type=str, default= './wikiTokenizedValid.hf')
    parser.add_argument('--lr', default = 0.00005, type=float)
    parser.add_argument('--output_dir', default = './acc_valid/')
    args = parser.parse_args()

    if args.train:
        args.dataset_file = './wikiTokenizedTrain.hf'
        args.output_dir = './acc/'

    print(args)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    collater = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    tokenized_dataset = datasets.load_from_disk(args.dataset_file)
    tokenized_dataset_valid = datasets.load_from_disk('./wikiTokenizedValid.hf')

    model = BertForPreTraining(config=BertConfig.from_pretrained("bert-base-cased"))
    optimizer = torch.optim.Adam(model.parameters(), lr =args.lr)

    accelerator = Accelerator()
    device = accelerator.device
    model.to(accelerator.device)
    train_args = TrainingArguments(output_dir=args.output_dir, overwrite_output_dir =True, per_device_train_batch_size =args.BATCH_SIZE, logging_first_step=True, 
                                   logging_strategy='epoch', evaluation_strategy = 'epoch', save_strategy ='epoch', num_train_epochs=args.EPOCHS,save_total_limit=50)
    t = Trainer(model, args = train_args, data_collator=collater, train_dataset = tokenized_dataset, optimizers=(optimizer, None), eval_dataset = tokenized_dataset_valid)
    t.train() #resume_from_checkpoint=True)

# logger = get_logger(__name__, log_level="INFO")
train_func()
