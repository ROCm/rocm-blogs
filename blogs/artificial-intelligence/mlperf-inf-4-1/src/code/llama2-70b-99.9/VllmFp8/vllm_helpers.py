import logging
from vllm import LLM, SamplingParams
import multiprocessing as mp
import numa_helpers as nh
from multiprocessing import connection as conn
from rpd_trace_utils import rpd_trace_range
import gc
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)


LLM_MODEL_LOAD_DONE = "LLM_MODEL_LOAD_DONE"
LLM_DONE = "DONE"

@rpd_trace_range("SUT:Worker")
def run_vllm(llm, prompt_token_ids, sampling_params):

    return llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else True,
    )

@rpd_trace_range("SUT:Worker")
def collect_tokens(pred_output_tokens):
    return [output.outputs[0].token_ids for output in pred_output_tokens]

@rpd_trace_range("SUT:Worker")
def send_back_tokens(qdata_out, items):
    qdata_out.send(items)

def llm_tp1(
    llm_kwargs: dict,
    sp_kwargs: dict,
    qdata_in: conn.Connection,
    qdata_out: conn.Connection,
    qstatus_out: mp.Queue,
    device: int,
):
    nh.set_affinity_by_device(device)
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(**sp_kwargs)

    qstatus_out.put(LLM_MODEL_LOAD_DONE)
    while True:        
        try:
            item = qdata_in.recv()
            if item is None:
                log.info(f"LLM is stopping")
                qdata_out.send(LLM_DONE)
                break
            
            start, end, prompt_token_ids = item

            pred_output_tokens = run_vllm(llm, prompt_token_ids, sampling_params)
            log.info(f"VLLM finished")

            processed_output = collect_tokens(pred_output_tokens)
            log.info(f"oputput tokens collected")

            send_back_tokens(qdata_out, (start, end, processed_output))
            log.info(f"Processed output | start, end = {start}, {end}")
        except:
            pass

