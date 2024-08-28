import logging
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from dataclasses import dataclass


import mlperf_loadgen as lg
# from vllm_mlperf.dataset import Dataset
from dataset import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)





# # Defaults for VLLM EngineArgs
# @dataclass
# class DefaultEngineInput:
#     max_model_len: int = 2048
#     block_size: int = 16
#     swap_space: int = 0    # GiB
#     gpu_memory_utilization: float = 0.95
#     max_context_len_to_capture: int = 2048
#     enforce_eager: bool = True
#     disable_custom_all_reduce: bool = True
#     max_num_batched_tokens: int = 65536
#     max_num_seqs:int = 10000
#     enable_chunked_prefill: bool = True



class SUT:
    """ Use this class to plug in to SUT. The methods are called by main.py for MLPerf tests. """
    def __init__(self, 
        model_path=None,
        dataset_path=None,
        dtype="float16",
        device="cuda:0",
        total_sample_count=24576,
        model_max_length = None,
        debug=False,
    ):
        log.info(f"Init SUT")
        self.debug = debug
        if self.debug:
            log.setLevel(logging.DEBUG)

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.dtype = dtype
        self.device = device
        self.model_max_length = model_max_length
        self.total_sample_count = total_sample_count
        
        self.tokenizer = None
        self.data_object = None
        self.qsl = None

        self.stop_test = False

        # if 'cuda' in self.device:
        #     assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.init_tokenizer()
        self.init_qsl()
        self.load()
    
    def init_tokenizer(self):
        pass

    
    def init_qsl(self):
        self.data_object = Dataset(
            # self.tokenizer,
            dataset_path=self.dataset_path,
            total_sample_count=self.total_sample_count,
            device=self.device,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

    def start(self):
        """ Start the SUT before LoadGen initiates the test. """
        pass
    
    def stop(self):
        """ Stop the SUT when LoadGen signals that the test is done. """
        self.stop_test = True

    def load(self):
        pass

    def get_sut(self):
        pass
    
    def get_qsl(self):
        pass

    def predict(self):
        pass

    def issue_queries(self,  query_samples):
        """ LoadGen sends in queries here. """
        pass

    def flush_queries(self):
        pass

    def __del__(self):
        pass



