import os
import time
import numpy as np
# import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
# from torch.nn.functional import pad
# from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
#import utils
import copy
import pickle

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

import random


class Dataset:
    def __init__(self,
        # tokenizer,
        total_sample_count=24576,
        perf_count_override=None,
        dataset_path=None,
        device="cpu"
    ):
        # self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.device = device

        self.load_processed_dataset()

        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count
    

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn("Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))

        log.info("Loading dataset...")
        import pandas as pd
        processed_data = pd.read_pickle(self.dataset_path)

        input_tokens = processed_data['tok_input']

        self.input_ids = []
        self.input_lens = []
        self.attention_masks = []

        for ids in input_tokens:
            #input_ids = torch.tensor(ids, dtype=torch.int32).view(1,-1).to(self.device)
            input_ids = ids
            #attn_mask = torch.ones_like(input_ids)
            self.input_ids.append(input_ids)
            #self.attention_masks.append(attn_mask)
            #self.input_lens.append(input_ids.shape[-1])
            self.input_lens.append(len(input_ids))
        log.info("Finished loading dataset.")

    
    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None):
        pass
    
    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass