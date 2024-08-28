""" Main function to be called for MLPerf. """
import logging
import multiprocessing as mp
import dataclasses
import mlperf_loadgen as lg
import argparse
import os
import logging
import sys
from sync_sut import SyncServerSUT as SUTServer
from dataclasses import dataclass
from rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-70b-chat-hf", help="Model name")
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy mode")
    parser.add_argument("--dtype", type=str, default="float32", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--device", type=str,  choices=["cpu", "cuda:0"], default="cpu", help="device to use")
    parser.add_argument("--audit-conf", type=str, default="audit.conf", help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--mlperf-conf", type=str, default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--total-sample-count", type=int, default=24576, help="Number of samples to use in benchmark.")
    parser.add_argument("--output-log-dir", type=str, default="output-logs", help="Where logs are saved")
    parser.add_argument("--enable-log-trace", action="store_true", help="Enable log tracing. This file can become quite large")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to process queries")
    parser.add_argument("--backend", choices=["pytorch", "vllm"], default="pytorch", help="Backend")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size in vllm")
    parser.add_argument("-dp", "--data-parallel-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--output-len", type=int, default=1024, help="Output length")
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS token in output generation.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization to use")
    parser.add_argument("--quantization-param-path", type=str, default=None, help="Path to quantized model parameters")
    parser.add_argument("--quantized-weights-path", type=str, default=None, help="Path to quantized weights")
    parser.add_argument("--kv-cache-dtype", type=str, default='auto', help="KV cache dtype")
    parser.add_argument("--model-name", type=str, default="MLperf-Llama2-70b", help="Server model name")
    parser.add_argument("--enable-warm-up", action="store_true", help="Run warm-up before benchmak")
    parser.add_argument("--enable-batcher", action="store_true", help="Enable harness side batching")

    # Tunable arguments for hyper parameter tuning
    parser.add_argument("--max-num-seqs", type=int, default=1024, help="Maximum number of sequences per vLLM iteration")
    parser.add_argument("--max-num-batched-tokens", type=int, default=49152, help="Maximum number of batched tokens per vLLM iteration")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97, help="The fraction of GPU memory to be used for the model executor in vLLM")
    parser.add_argument("--enforce-eager", type=bool, default=None, help="Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility.")
    parser.add_argument("--block-size", type=int, default=None, help="Token block size for contiguous chunks of tokens for vLLM")
    parser.add_argument("--swap-space", type=int, default=None, help="CPU swap space size (GiB) per GPU for vLLM")
    parser.add_argument("--disable-custom-all-reduce", type=bool, default=None, help="Disable the custom all-reduce kernel in vLLM and fall back to NCCL")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=512, help="Maximum sequence length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode")
    parser.add_argument("--sorting", type=str, default=None, choices=["ascending", "descending", "lexicographic", "skip"], help="Sorting method applied to samples")
    parser.add_argument("--batcher-threshold", type=float, default=0.3, help="Maximum time to wait before sending a batch to vLLM")
    parser.add_argument("--gpu-batch-size", type=int, default=96, help="Maximum batch size on harness side")

    args = parser.parse_args()
    return args


@dataclass
class DefaultEngineInput:
    max_model_len: int = 2048
    block_size: int = 16
    swap_space: int = 0    # GiB
    gpu_memory_utilization: float = 0.99
    max_seq_len_to_capture: int = 1024
    # RPD tracing crashes when eager mode is off
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = True
    max_num_batched_tokens: int = 65536
    max_num_seqs:int = 768
    # Chunked prefill can't be used together with prefix caching
    # enable_prefix_caching: bool = True
    # enable_chunked_prefill: bool = True


def setup_llm_kwargs(args):
    llm_config = DefaultEngineInput()
    llm_config.block_size = args.block_size if args.block_size else llm_config.block_size
    llm_config.swap_space = args.swap_space if args.swap_space else llm_config.swap_space
    llm_config.gpu_memory_utilization = args.gpu_memory_utilization if args.gpu_memory_utilization else llm_config.gpu_memory_utilization
    llm_config.max_seq_len_to_capture = args.max_seq_len_to_capture if args.max_seq_len_to_capture else llm_config.max_seq_len_to_capture
    llm_config.enforce_eager = args.enforce_eager if args.enforce_eager else llm_config.enforce_eager
    llm_config.disable_custom_all_reduce = args.disable_custom_all_reduce if args.disable_custom_all_reduce else llm_config.disable_custom_all_reduce
    llm_config.max_num_batched_tokens = args.max_num_batched_tokens if args.max_num_batched_tokens else llm_config.max_num_batched_tokens
    llm_config.max_num_seqs = args.max_num_seqs if args.max_num_seqs else llm_config.max_num_seqs
    return dataclasses.asdict(llm_config)

@rpd_trace_range("MLPERF")
def harness_main(args):
    log.info(f"Args = {args}")

    llm_kwargs = setup_llm_kwargs(args)

    scenario_map = {
        "offline": lg.TestScenario.Offline,
        "server": lg.TestScenario.Server,
    }

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "llama2-70b", args.scenario)
    settings.FromConfig(args.user_conf, "llama2-70b", args.scenario)

    if args.total_sample_count != 24576:
        settings.min_query_count = args.total_sample_count
        settings.max_query_count = args.total_sample_count

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        log.warning("Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet")
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    sut = SUTServer(
        model_path=args.model_path,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        device=args.device,
        total_sample_count=args.total_sample_count,
        tp=args.tensor_parallel_size,
        dp=args.data_parallel_size,
        quantization=args.quantization,
        quantization_param_path=args.quantization_param_path,
        quantized_weights_path=args.quantized_weights_path,
        kv_cache_dtype=args.kv_cache_dtype,
        llm_kwargs=llm_kwargs,
        enable_batcher=args.enable_batcher,
        batcher_threshold=args.batcher_threshold,
        gpu_batch_size=args.gpu_batch_size,
    )

    sut.start()

    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("SUT - Starting Benchmark run")
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, settings, log_settings, args.audit_conf)
    log.info("Completed benchmark run")

    sut.stop()

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)

def main():
    mp.set_start_method("spawn")
    log.info(f"mp.get_context:{mp.get_context()}")
    args = get_args()
    harness_main(args)

if __name__ == "__main__":
    main()
