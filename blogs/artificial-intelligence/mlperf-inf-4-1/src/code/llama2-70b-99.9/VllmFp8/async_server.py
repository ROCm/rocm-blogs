import logging
import dataclasses
from dataclasses import dataclass
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
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


class AsyncServer:

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
        qdata_out,
        qstatus_out: mp.Queue,
        tokenizer,
        llm_kwargs,
    ):
        self.qdata_in = qdata_in
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
        self.llm_kwargs["disable_log_requests"] = True if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else False
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

        engine_args = AsyncEngineArgs(
            worker_use_ray=False,
            engine_use_ray=False,
            **self.llm_kwargs
        )

        self.sampling_params = SamplingParams(**sp_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args=engine_args, start_engine_loop=True)

        self.signal_running()
        self.run()

    def signal_running(self):
        self.qstatus_out.put_nowait(AsyncServer.SIG_RUN)

    @rpd_trace_range("SUT:Worker")
    def run(self):
        async_event_loop = asyncio.new_event_loop()
        async_thread = threading.Thread(target=run_async_event_loop, args=([async_event_loop]), daemon=True)
        async_thread.start()
        self.log("Processing started...")
        while True:
            try:
                sample = self.qdata_in.get()
                if sample is None:
                    self.error("qdata_in got end signal...")
                    break
                asyncio.run_coroutine_threadsafe(self.generate_v2(sample), async_event_loop)
            except queue.Empty:
                break


    def is_running(self):
        try:
            return self.qstatus_out.get_nowait() == AsyncServer.SIG_RUN
        except:
            return False


    async def generate_v2(self, sample):
        prompt_token_ids = sample[0]
        sample_ids = sample[1]
        is_warm_up = sample[2]
        await asyncio.wait([asyncio.create_task(self.generate((prompt_token_ids[i], sample_ids[i], is_warm_up))) for i in range(len(sample_ids))])


    async def generate(self, sample):
        prompt_token_ids = sample[0]
        request_id = str(sample[1])
        is_warm_up = sample[2]
        results_generator = self.engine.generate({"prompt_token_ids": prompt_token_ids}, self.sampling_params, request_id)
        output_token_ids = []
        is_first_token = True
        async for request_output in results_generator:
            output_token_ids = request_output.outputs[0].token_ids
            if is_first_token:
                is_first_token = False
                self.qdata_out.send([output_token_ids, request_id, True, is_warm_up])
        self.qdata_out.send([output_token_ids, request_id, False, is_warm_up])


    def log(self, message):
        log.info(f"Server {self.devices} - {message}")


    def error(self, message):
        log.error(f"Server {self.devices} - {message}")


def run_async_event_loop(async_event_loop):
    asyncio.set_event_loop(async_event_loop)
    async_event_loop.run_forever()
