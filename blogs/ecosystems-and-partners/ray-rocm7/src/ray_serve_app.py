from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
    }, 
    deployment_config={
        "max_ongoing_requests": 1,
        "autoscaling_config": {
            "target_ongoing_requests": 1,
            "min_replicas": 1,
            "max_replicas": 8,
            "upscale_delay_s": 15
         }
    },
    # Pass the desired accelerator type
    accelerator_type="AMD-Instinct-MI300X-OAM",
    # You can customize the engine arguments (e.g. vLLM engine kwargs)
    engine_kwargs={
        "tensor_parallel_size": 1,
    },
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)