import logging
import mlperf_loadgen as lg
import dataclasses
from dataclasses import dataclass
import numpy as np
import array

import multiprocessing as mp
import os
import time
from threading import Thread
from rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed

from SUT import SUT
from vllm_helpers import llm_tp1, LLM_MODEL_LOAD_DONE, LLM_DONE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)


# Defaults for VLLM SamplingParams
@dataclass
class SamplingParamsInput:
    n: int = 1
    max_tokens: int = 1024
    min_tokens: int = 1
    temperature: float = 0
    repetition_penalty: float = 1
    frequency_penalty: float = 0
    ignore_eos: bool = False
    detokenize: bool = False    # Check that vllm version supports this
    early_stopping=False
    use_beam_search=False


class LlmProcTP1():
    def __init__(self,
        device,
        dtype,
        model,
        quantization,
        quantization_param_path,
        quantized_weights_path,
        qdata_in,
        qdata_out,
        llm_kwargs,
    ):
        self.qdata_in = qdata_in # conn.Connection()
        self.qdata_out = qdata_out # conn.Connection()
        self.qstatus_out = mp.Queue()
        self.device = device
        self.dtype = dtype
        self.model_path = model
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.quantized_weights_path = quantized_weights_path
        self.init_llm(llm_kwargs)

    def init_llm(self, llm_kwargs):
        llm_kwargs["tensor_parallel_size"] = 1  #self.tp
        llm_kwargs["dtype"] = self.dtype
        llm_kwargs["model"] = self.model_path
        llm_kwargs["quantization"] = self.quantization
        llm_kwargs["disable_log_stats"] = True if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else False
        llm_kwargs["skip_tokenizer_init"] = True

        if self.quantization_param_path:
            llm_kwargs["quantization_param_path"] = self.quantization_param_path
        if self.quantized_weights_path:
            llm_kwargs["quantized_weights_path"] = self.quantized_weights_path

        sp_config = SamplingParamsInput()
        sp_kwargs = dataclasses.asdict( sp_config )

        log.info(f"llm_kwargs={llm_kwargs}")
        log.info(f"sp_kwargs={sp_kwargs}")

        os.environ["HIP_VISIBLE_DEVICES"] = f"{self.device}"
        self.llm_proc = mp.Process(target=llm_tp1, args=(llm_kwargs, sp_kwargs, self.qdata_in, self.qdata_out, self.qstatus_out, self.device))
        self.llm_proc.start()

    def check_llm_loaded(self):
        while True:
            status = self.qstatus_out.get()
            if status == LLM_MODEL_LOAD_DONE:
                log.info(f"LLM is loaded")
                break


class SUTVllmFp8Offline_ntp1(SUT):
    """ Extend SUT for the llama-2-70b N*TP1 VLLM implementation. """
    def __init__(self,
        model_path=None,
        dataset_path=None,
        dtype="float16",
        device="cuda:0",
        total_sample_count=24576,
        model_max_length = None,
        debug=False,
        tp = 1,
        quantization: str = None,
        quantization_param_path: str = None,
        quantized_weights_path: str = None,
        kv_cache_dtype: str = 'auto',
        dp = 1,
        warmup_duration = 0,
        sorting: str = None,
        llm_kwargs = None
    ):
        log.info(f"Init SUTVllm")
        super().__init__(
            model_path=model_path,
            dataset_path=dataset_path,
            dtype=dtype,
            device=device,
            total_sample_count=total_sample_count,
            model_max_length=model_max_length,
        )
        self.tp = tp
        self.quantization = "fp8"
        self.quantization_param_path = quantization_param_path
        self.quantized_weights_path = quantized_weights_path
        self.kv_cache_dtype = kv_cache_dtype

        self.init_sampling_params()

        self.dp = dp

        self.qdata_in_senders = []
        self.qdata_out_receivers = []
        self.qstatus_out = mp.Queue()

        self.llm_procs = []
        self.llm_objs = []
        self.warmup_duration = warmup_duration * 60
        self.sorting = sorting
        self.sample_ids = []
        self.completion_threads = []
        self.llm_kwargs = llm_kwargs
        self.gpu_stats = {}
        self.start_t = time.time()
        self.infer_start_t = time.time()

    @rpd_trace_range_non_timed("SUT:Main")
    def init_llms(self):
        for device in range(self.dp):
            qdata_in_receiver, qdata_in_sender = mp.Pipe(False)
            qdata_out_receiver, qdata_out_sender = mp.Pipe(False)
            self.qdata_in_senders.append(qdata_in_sender)
            self.qdata_out_receivers.append(qdata_out_receiver)
            llm_obj = LlmProcTP1(device, self.dtype, self.model_path, self.quantization,
                    self.quantization_param_path, self.quantized_weights_path,
                    qdata_in_receiver, qdata_out_sender, self.llm_kwargs)
            self.llm_objs.append(llm_obj)

        for obj in self.llm_objs:
            obj.check_llm_loaded()

    @rpd_trace_range_non_timed("SUT:Main")
    def start_completion_threads(self):
        for i in range(self.dp):
            self.completion_threads.append(Thread(target=self.completion, args=(i,)))
            self.completion_threads[-1].start()


    def init_sampling_params(self):
        pass

    @rpd_trace_range_non_timed("SUT:Main")
    def warmup(self):
        # log.info("starting warmup")
        # duration = 0
        # while duration < self.warmup_duration:
        #     start = time.time()
        #     for i in range(self.dp):
        #         # TODO: generate random token ids
        #         prompt_token_ids = [self.data_object.input_ids[i] for i in range(1500)]
        #         self.qdata_in_senders[i].send((0, None, prompt_token_ids))
        #         log.info(f"Put prompt tokens in qdata_in  |  qsize = {self.qdata_in_senders[i].qsize()}")

        #     n_done = 0
        #     while n_done < self.dp:
        #         try:
        #             item = self.qdata_out_receivers.get_nowait()
        #             if item:
        #                 n_done += 1
        #             log.info(f"{n_done =}")
        #         except:
        #             pass
        #     duration += time.time() - start
        # log.info("Warmup finished")
        pass

    @rpd_trace_range_non_timed("SUT:Main")
    def stop(self):
        for t in self.completion_threads:
            t.join()
        log.info(f"Total time spent with run: {time.time() - self.start_t}")
        

    @rpd_trace_range_non_timed("SUT:Main")
    def start(self):
        log.info(f"SUT start")
        self.init_llms()
        self.start_completion_threads()
        self.warmup()
        self.infer_start_t = time.time()
        log.info(f"Time spent from start to inference start: {self.infer_start_t - self.start_t}")

    @rpd_trace_range("SUT:Main")
    def make_ranges(self, query_samples):
        query_chunk_size = (len(query_samples) + self.dp - 1) // self.dp
        ranges = []
        for i in range(self.dp):
            start = i * query_chunk_size
            end = start + query_chunk_size
            if end > len(query_samples):
                end = None
            ranges.append((start, end))
        return ranges

    @rpd_trace_range("SUT:Main")
    def sort_by_length(self, query_samples, weight=1):
        reord_start = time.time_ns()
        ranges = self.make_ranges(query_samples)
        evened_out_samples = self.even_out_token_count(query_samples, ranges[0][1] - ranges[0][0])
        reordered_samples = []
        for start, stop in ranges:
            chunk = evened_out_samples[start:stop]
            chunk.sort(key=lambda sample: weight * len(self.data_object.input_ids[sample.index]))
            reordered_samples.extend(chunk)
        reord_dur = (time.time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples

    @rpd_trace_range("SUT:Main")
    def sort_lexicog(self, query_samples):
        reord_start = time.time_ns()
        ranges = self.make_ranges(query_samples)
        evened_out_samples = self.even_out_token_count(query_samples, ranges[0][1] - ranges[0][0])
        reordered_samples = []
        for start, stop in ranges:
            chunk = evened_out_samples[start:stop]
            chunk.sort(key=lambda sample: self.data_object.input_ids[sample.index])
            reordered_samples.extend(chunk)
        reord_dur = (time.time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples

    @rpd_trace_range("SUT:Main")
    def even_out_token_count(self, query_samples, query_chunk_size):
        full_buckets = []
        buckets = [[] for _ in range(self.dp)]
        bucket_sizes = [0 for _ in range(self.dp)]
        for sample in query_samples:
            smallest_bucket = bucket_sizes.index(min(bucket_sizes))
            buckets[smallest_bucket].append(sample)
            bucket_sizes[smallest_bucket] += len(self.data_object.input_ids[sample.index])
            if len(buckets[smallest_bucket]) == query_chunk_size and len(buckets) > 1:
                full_buckets.append(buckets[smallest_bucket])
                del buckets[smallest_bucket]
                del bucket_sizes[smallest_bucket]
        reordered_samples = []
        for bucket in full_buckets + buckets:
            reordered_samples.extend(bucket)
        return reordered_samples
    

    @rpd_trace_range("SUT:Main")
    def sort_samples(self, query_samples):
        log.info(f"Sorting samples in {self.sorting} order")
        if self.sorting == "ascending":
            return self.sort_by_length(query_samples, weight=1)
        elif self.sorting == "descending":
            return self.sort_by_length(query_samples, weight=-1)
        elif self.sorting == "lexicographic":
            return self.sort_lexicog(query_samples)
        else:
            return (self.make_ranges(query_samples), query_samples)

    @rpd_trace_range("SUT:Main")
    def update_gpu_stats(self, device, time):
        self.gpu_stats[device] = time - self.infer_start_t
        if len(self.gpu_stats) == self.dp:
            first = min(self.gpu_stats, key=self.gpu_stats.get)
            last = max(self.gpu_stats, key=self.gpu_stats.get)
            delay = self.gpu_stats[last] - self.gpu_stats[first]
            self.gpu_stats["delay"] = delay
            log.info(f"GPU inference durations: {self.gpu_stats}")

    @rpd_trace_range("SUT:Main")
    def post_proc(self, response):
        start, end, output_token_ids = response
        log.info(f"Got item  |  start, end = {start}, {end}  |  n outputs = {len(output_token_ids)}")

        output_sample_ids = self.sample_ids[start : end]
        assert len(output_sample_ids) == len(output_token_ids)

        log.info(f"Signaling LoadGen output")

        try:
            for i in range(len(output_token_ids)):
                response_array = array.array("B", np.array(output_token_ids[i], np.int32).tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(output_sample_ids[i], bi[0], bi[1], len(output_token_ids[i]))]
                lg.QuerySamplesComplete(response)
        except:
            log.info(f"Error sending completed response to LoadGen")

    def completion(self, device):
        while True:
            try:
                response = self.qdata_out_receivers[device].recv()
                if response == LLM_DONE:
                    log.info(f"Query chunk done for GPU {device}")
                    # self.update_gpu_stats(device, time.time())
                    break
                self.post_proc(response)                
            except:
                pass

    @rpd_trace_range("SUT:Main")
    def send_tokens(self, sender_id, start, end, prompt_token_ids):
        self.qdata_in_senders[sender_id].send((start, end, prompt_token_ids[start : end]))

    @rpd_trace_range("SUT:Main")
    def issue_queries(self, query_samples):
        log.info(f"Issue queries  |  number of queries = {len(query_samples)}")
        ranges, query_samples = self.sort_samples(query_samples)
        self.sample_ids = [query_samples[i].id for i in range(len(query_samples))]
        prompt_token_ids = [self.data_object.input_ids[query_samples[i].index] for i in range(len(query_samples))]
        log.info(f"Converted queries to prompt tokens  |  number of queries = {len(prompt_token_ids)}")

        for i, (start, end) in enumerate(ranges):
            self.send_tokens(i, start, end, prompt_token_ids)
            log.info(f"Put prompt tokens in pipe #{i}")
        
        for i in range(self.dp):
            self.qdata_in_senders[i].send(None)
