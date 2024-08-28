import logging
import dataclasses
from dataclasses import dataclass
from queue_llm import QueueLLM
from vllm import SamplingParams
import multiprocessing as mp
import os
import asyncio
from SUTVllm import SamplingParamsInput
import logging
import numa_helpers as nh
import threading
from rpd_trace_utils import rpd_trace_range_async, rpd_trace_range, rpd_trace_range_non_timed
import queue
import gc
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

@dataclass
class SamplingParamsInput:
    max_tokens: int = 1024
    min_tokens: int = 1
    temperature: float = 0
    repetition_penalty: float = 1
    frequency_penalty: float = 0
    ignore_eos: bool = False
    detokenize: bool = False
    early_stopping=False
    use_beam_search=False
    skip_special_tokens=False


class SyncServer:

    SIG_RUN = 1
    SIG_STOP = 2

    def __init__(
        self,
        devices,
        dtype,
        model_path,
        quantization,
        quantization_param_path,
        quantized_weights_path,
        kv_cache_dtype,
        qdata_in,
        qdata_first_token,
        qdata_out,
        qstatus_out: mp.Queue,
        tokenizer,
        llm_kwargs,
    ):
        self.qdata_in = qdata_in
        self.qdata_first_token = qdata_first_token
        self.qdata_out = qdata_out
        self.qstatus_out = qstatus_out
        self.devices = devices
        self.dtype = dtype
        self.model_path = model_path
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.quantized_weights_path = quantized_weights_path
        self.kv_cache_dtype = kv_cache_dtype
        self.tokenizer = tokenizer
        self.engine = None
        self.process = None
        self.llm_kwargs = llm_kwargs

    @rpd_trace_range_non_timed("SUT:Worker")
    def start(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])
        self.process = mp.Process(target=self.launch)
        self.process.start()

    @rpd_trace_range_non_timed("SUT:Worker")
    def launch(self):
        nh.set_affinity_by_device(self.devices[0])

        self.llm_kwargs["tensor_parallel_size"] = 1  #self.tp
        self.llm_kwargs["dtype"] = self.dtype
        self.llm_kwargs["model"] = self.model_path
        self.llm_kwargs["quantization"] = self.quantization
        self.llm_kwargs["disable_log_stats"] = True if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else False
        self.llm_kwargs["kv_cache_dtype"] = self.kv_cache_dtype
        self.llm_kwargs["skip_tokenizer_init"] = True

        if self.quantization_param_path:
            self.llm_kwargs["quantization_param_path"] = self.quantization_param_path
        if self.quantized_weights_path:
            self.llm_kwargs["quantized_weights_path"] = self.quantized_weights_path

        sp_config = SamplingParamsInput()
        sp_kwargs = dataclasses.asdict( sp_config )

        self.log(f"llm_kwargs={self.llm_kwargs}")
        self.log(f"sp_kwargs={sp_kwargs}")

        sampling_params = SamplingParams(**sp_kwargs)
        self.engine = QueueLLM(input_queue=self.qdata_in,
                               first_token_queue=self.qdata_first_token,
                               result_queue=self.qdata_out,
                               sampling_params=sampling_params,
                               **self.llm_kwargs)

        self.signal_running()
        use_tqdm = False if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else True
        self.engine.start(use_tqdm=use_tqdm)

    def signal_running(self):
        self.qstatus_out.put_nowait(SyncServer.SIG_RUN)


    def is_running(self):
        try:
            return self.qstatus_out.get_nowait() == SyncServer.SIG_RUN
        except:
            return False


    def log(self, message):
        log.info(f"Server {self.devices} - {message}")


    def error(self, message):
        log.error(f"Server {self.devices} - {message}")

