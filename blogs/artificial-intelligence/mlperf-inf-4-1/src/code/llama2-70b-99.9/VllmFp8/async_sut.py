import logging
import mlperf_loadgen as lg
import multiprocessing as mp
import time
import numpy as np
import array

from SUT import SUT
from async_server import AsyncServer
from rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed
import threading
import sys
from datetime import datetime
import queue

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)


class AsyncServerSUT(SUT):
    def __init__(self,
        model_path=None,
        dataset_path=None,
        dtype="float16",
        device="cuda:0",
        total_sample_count=24576,
        model_max_length = None,
        tp = 1,
        quantization: str = None,
        quantization_param_path: str = None,
        quantized_weights_path: str = None,
        kv_cache_dtype: str = 'auto',
        dp: int = 1,
        model_name: str = "llama2-70b",
        llm_kwargs = None,
        enable_batcher = False,
        batcher_threshold = None,
        gpu_batch_size = None,
    ):
        log.info(f"Init SUTVllmFP8Server")
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
        self.dp = dp
        self.model_name = model_name

        self.servers = {}
        self.output_collector_threads = []
        self.device_counter = 0
        self.llm_kwargs = llm_kwargs
        self.warm_up_done = []
        self.warm_up_sample_count_per_server = 50
        self.total_sample_count = total_sample_count
        self.n_finished = 0
        self.stopped = False
        self.enable_batcher = enable_batcher
        if self.enable_batcher:
            self.batcher_threshold = batcher_threshold
            self.gpu_batch_size = gpu_batch_size
            self.batcher_queue = mp.Queue()
            self.batcher_thread = threading.Thread(target=self.batch_samples_loop, args=())


    @rpd_trace_range_non_timed("SUT:Main")
    def start(self):
        for i in range(self.dp):
            devices = [i]

            qdata_in = mp.Queue(maxsize=10000)
            qdata_out_receiver, qdata_out_sender = mp.Pipe(False)
            qstatus_out = mp.Queue()

            server = AsyncServer(
                devices,
                self.dtype,
                self.model_path,
                self.quantization,
                self.quantization_param_path,
                self.quantized_weights_path,
                self.kv_cache_dtype,
                qdata_in,
                qdata_out_sender,
                qstatus_out,
                self.tokenizer,
                self.llm_kwargs)

            self.servers[i] = {
                "server" : server,
                "qdata_in" : qdata_in,
                "qdata_out" : qdata_out_receiver,
                "qstatus_out": qstatus_out,
                }

            self.servers[i]["server"].start()
            self.warm_up_done.append(threading.Event())
            self.output_collector_threads.append(threading.Thread(target=self.send_outputs, args=([qdata_out_receiver, self.warm_up_done[i], i]), daemon=True))
            self.output_collector_threads[-1].start()

        if self.enable_batcher:
            log.info(f"Server enabling batcher")
            self.batcher_thread.start()

        for index in self.servers:
            while True:
                log.info(f"i={index} | Polling server...")
                if self.servers[index]["server"].is_running():
                    log.info(f"i={index} | Server is ready")
                    break
                else:
                    time.sleep(10)


    @rpd_trace_range("SUT:Main")
    def send_samples(self, samples):
        prompt_token_ids = [self.data_object.input_ids[sample.index] for sample in samples]
        sample_ids = [str(sample.id) for sample in samples]
        self.servers[self.next_device_id()]["qdata_in"].put_nowait((prompt_token_ids, sample_ids, False))


    @rpd_trace_range("SUT:Main")
    def batch_samples_loop(self):

        batched_samples = self.batcher_queue.get()
        timeout_stamp = time.time()
        while True:
            if len(batched_samples) != 0 and (
                    len(batched_samples) >= self.gpu_batch_size
                    or time.time() - timeout_stamp >= self.batcher_threshold
            ):  # max batch or time limit exceed
                # log.info(f"Formed batch of {len(batched_samples[:self.gpu_batch_size])} samples")
                self.send_samples(batched_samples[:self.gpu_batch_size])
                batched_samples = batched_samples[self.gpu_batch_size:]
                timeout_stamp = time.time()

            try:
                samples = self.batcher_queue.get(timeout=self.batcher_threshold)
            except queue.Empty:
                continue

            if samples is None:  # None in the queue indicates the SUT want us to exit
                break
            batched_samples += samples


    @rpd_trace_range("SUT:Main")
    def issue_queries(self, query_samples):
        # num_samples = len(query_samples)
        # log.info(f"[Server] Received {num_samples} samples")
        # for i in range(0, num_samples, self.gpu_batch_size):
            # Construct batches
            # actual_batch_size = (self.gpu_batch_size if num_samples - i > self.gpu_batch_size else num_samples - i)
        if self.enable_batcher:
            self.batcher_queue.put(query_samples)
        else:
            for sample in query_samples:
                self.send_sample(sample, False)

        

    def print_finished(self):
        # time
        now = datetime.now()
        now_mon = "0"
        if now.month < 10:
            now_mon += str(now.month)
        else:
            now_mon = str(now.month)

        now_day = "0"
        if now.day < 10:
            now_day += str(now.day)
        else:
            now_day = str(now.day)

        now_hr = "0"
        if now.hour < 10:
            now_hr += str(now.hour)
        else:
            now_hr = str(now.hour)

        now_min = "0"
        if now.minute < 10:
            now_min += str(now.minute)
        else:
            now_min = str(now.minute)

        now_sec = "0"
        if now.second < 10:
            now_sec += str(int(now.second))
        else:
            now_sec = str(int(now.second))

        tm = str(now.year) + "-" + now_mon + "-" + now_day + " " + now_hr + ":" + now_min + ":" + now_sec + " INFO     SUT - "
        msg = '\r' + tm + 'Processed prompts: ' + str(self.n_finished) + ' '
        sys.stdout.write(msg)
        sys.stdout.flush()


    @rpd_trace_range("SUT:Main")
    def post_proc(self, reponse):
        processed_output = reponse[0]
        sample_id = int(reponse[1])
        is_first_token = reponse[2]
        response_array = array.array("B", np.array(processed_output, np.int32).tobytes())
        bi = response_array.buffer_info()
        response = [lg.QuerySampleResponse(sample_id, bi[0], bi[1], len(processed_output))]
        if is_first_token:
            lg.FirstTokenComplete(response)
        else:
            lg.QuerySamplesComplete(response)
            self.n_finished += 1



    def send_outputs(self, qdata_out, warm_up_done, server_index):
        self.log("Collecting outputs started...")
        warm_up_output_counter = 0
        while(True):
            reponse = qdata_out.recv()
            is_warm_up = reponse[3]
            if is_warm_up:
                warm_up_output_counter += 1
                if warm_up_output_counter % 10 == 0:
                    self.log(f"Server [{server_index}] warm-up in progress...")
                # we receive 2 outputs per sample, first token complete, and all tokens complete
                if warm_up_output_counter == (2 * self.warm_up_sample_count_per_server):
                    warm_up_done.set()
                    self.log(f"Server [{server_index}] warm-up completed")
                continue
            self.post_proc(reponse)

            if not self.stopped:
                self.print_finished()



    @rpd_trace_range_non_timed("SUT:Main")
    def warm_up(self):
        self.log("Warm-up started...")
        for i in range(self.dp * self.warm_up_sample_count_per_server):
            self.send_sample(Sample(i), True)
        self.log("Warm-up samples sent")
        for i in range(self.dp):
            self.log(f"Waiting for server[{i}] warm-up to complete...")
            self.warm_up_done[i].wait()
        self.log("Warm-up completed")


    @rpd_trace_range_non_timed("SUT:Main")
    def stop(self):
        if self.enable_batcher:
            self.batcher_queue.put(None)
        for index in self.servers:
            self.servers[index]["qdata_in"].put(None)
        self.stopped = True
        time.sleep(10)


    @rpd_trace_range("SUT:Main")
    def next_device_id(self):
        next_div_id = self.device_counter
        self.device_counter = (self.device_counter + 1) % len(self.servers)
        return next_div_id


    @rpd_trace_range("SUT:Main")
    def issue_queries_old(self, query_samples):
        for sample in query_samples:
            self.send_sample(sample, False)


    def send_sample(self, sample, is_warmup):
        prompt_token_ids = self.data_object.input_ids[sample.index]
        self.servers[self.next_device_id()]["qdata_in"].put_nowait(([prompt_token_ids], [sample.id], is_warmup))


    def log(self, message: str):
        log.info(f"SUT - {message}")


class Sample:
    def __init__(self, index):
        self.index = index
        self.id = str(index)
