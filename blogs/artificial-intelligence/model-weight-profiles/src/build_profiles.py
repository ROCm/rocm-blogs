"""Build the ten YAML profiles used by this blog."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

ROOT = Path(__file__).parent
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1}
FAMILIES = ("embedding", "attention", "dense", "conv", "other")


def model(model_id, revision, kind, activation=None):
    """Describe one Hub model in the profile set."""
    return {"model_id": model_id, "revision": revision, "kind": kind, "activation": activation}


MODELS = {
    "bert_base_uncased.profile.yaml": model("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594", "bert"),
    "llama_3_1_8b.profile.yaml": model("unsloth/Meta-Llama-3.1-8B", "e9a141a2091ea561b96483212645a2a05e6f99fc", "llama"),
    "llama_3_1_8b_instruct.profile.yaml": model("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1", "llama"),
    "llama_3_1_70b_instruct.profile.yaml": model("unsloth/Meta-Llama-3.1-70B-Instruct", "1fdd0a465a29664d155aee8a9f77c55a65cc8d5f", "llama"),
    "llama_3_1_8b_instruct_fp8.profile.yaml": model("amd/Llama-3.1-8B-Instruct-FP8-KV", "e7d3fe2d518920b13c9d192cd5bc355f9beb093f", "llama"),
    "llama_3_1_70b_instruct_fp8.profile.yaml": model("amd/Llama-3.1-70B-Instruct-FP8-KV", "7fa054de6ec7ff9d2fbc8fd78c9d638d62890c76", "llama"),
    "deepseek_r1.profile.yaml": model("deepseek-ai/DeepSeek-R1", "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad", "deepseek", "deepseek"),
    "qwen3_vl_30b_a3b_instruct.profile.yaml": model("Qwen/Qwen3-VL-30B-A3B-Instruct", "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c", "qwen", "qwen"),
    "qwen3_vl_30b_a3b_instruct_fp8.profile.yaml": model("Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "d9748a51ae66354c4dad665aab2c71f26cf2c8cd", "qwen", "qwen"),
}


# Ordered because scales and expert tensors also contain ordinary projection names.
RULES = [
    (r"visual\.pos_embed", "other", "vision_positional_embedding"),
    (r"\.experts\..*scale", "other", "expert_quant_scales"),
    (r"(?:input|weight|kv)_scale|weight_scale_inv", "other", "quant_scales"),
    (r"patch_embed", "conv", "patch_embed"),
    (r"(word|position|token_type)_embeddings", "embedding", "{}_embeddings"),
    (r"embed_tokens", "embedding", "token_embeddings"),
    (r"lm_head", "embedding", "lm_head"),
    (r"\.mlp\.gate\.weight", "other", "moe_router"),
    (r"\.mlp\.experts\.(?:gate_up|down)_proj", "dense", "expert_weights"),
    (r"\.experts\.\d+\.(?:gate|up|down)_proj", "dense", "expert_weights"),
    (r"shared_experts\.(?:gate|up|down)_proj", "dense", "shared_expert_weights"),
    (r"self_attn\.([qkvo]_proj)", "attention", "{}"),
    (r"self_attn.*(?:q_norm|k_norm|layernorm)", "attention", "attn_norm"),
    (r"self_attn", "attention", "attention_weights"),
    (r"visual.*\.attn\.", "attention", "vision_attention"),
    (r"attention.*layernorm", "attention", "layer_norm"),
    (r"attention.*bias", "attention", "projection_bias"),
    (r"attention", "attention", "projection_weights"),
    (r"input_layer_?norm", "attention", "layer_norm"),
    (r"post_attention_layer_?norm", "dense", "layer_norm"),
    (r"visual.*(?:layernorm|\.norm)", "other", "vision_norm"),
    (r"visual.*(?:linear_fc|merger)", "dense", "vision_mlp"),
    (r"(?:gate|up|down)_proj", "dense", "mlp_weights"),
    (r"(intermediate|(?<!attention\.)output)\.dense\.weight", "dense", "ffn_weights"),
    (r"(?<!attention\.)output\.layernorm", "dense", "layer_norm"),
    (r"(pooler|cls\.(predictions\.transform|seq_relationship))\.dense", "dense", "output_heads"),
    (r"e_score_correction_bias", "other", "moe_router_bias"),
    (r"\.(eh_proj|enorm|hnorm|shared_head)", "other", "nextn_predict"),
    (r"layernorm|\.norm", "other", "norm"),
]


ARCHITECTURE = {
    "bert": [("", "", "model_type hidden_size num_hidden_layers num_attention_heads intermediate_size vocab_size max_position_embeddings type_vocab_size")],
    "llama": [("", "", "model_type hidden_size num_hidden_layers num_attention_heads num_key_value_heads intermediate_size vocab_size max_position_embeddings tie_word_embeddings")],
    "deepseek": [
        ("", "", "model_type hidden_size num_hidden_layers num_attention_heads num_key_value_heads intermediate_size vocab_size max_position_embeddings num_nextn_predict_layers"),
        ("", "moe", "n_routed_experts num_experts_per_tok n_shared_experts first_k_dense_replace moe_intermediate_size"),
        ("quantization_config", "quantization", "quant_method fmt weight_block_size"),
    ],
    "qwen": [
        ("", "", "model_type tie_word_embeddings"),
        ("text_config", "", "hidden_size num_hidden_layers num_attention_heads num_key_value_heads intermediate_size vocab_size head_dim"),
        ("vision_config", "vision", "depth hidden_size patch_size in_channels"),
        ("text_config", "moe", "num_experts num_experts_per_tok moe_intermediate_size"),
    ],
}


ACTIVATION = {
    "deepseek": {
        "source": "deepseek_v3_moe_activation_rules",
        "exclude_layer": "num_hidden_layers",
        "ratio": ("num_experts_per_tok", "n_routed_experts"),
        "experts": r"\.experts\.\d+\.",
    },
    "qwen": {
        "source": "qwen3_vl_moe_language_token_rules",
        "exclude": "visual",
        "ratio": ("text_config.num_experts_per_tok", "text_config.num_experts"),
        "experts": r"\.mlp\.experts\.",
        "expert_axis": 0,
    },
}


VASWANI_TRANSFORMER = {
    "architecture": {"model_type": "transformer", "hidden_size": 512, "num_hidden_layers": 6,
                     "num_attention_heads": 8, "intermediate_size": 2048, "vocab_size": 37000},
    "groups": {
        "embedding": {"shared_embedding_and_pre_softmax": (18944000, [37000, 512])},
        "attention": {"encoder_self_attn": (6291456, [512, 512]),
                      "decoder_self_attn": (6291456, [512, 512]),
                      "decoder_cross_attn": (6291456, [512, 512])},
        "dense": {"encoder_ffn": (12582912, [2048, 512]),
                  "decoder_ffn": (12582912, [2048, 512]),
                  "ffn_bias": (30720, [2560])},
        "other": {"layer_norm": (30720, [1024])},
    },
}


def at(data, path):
    """Read a dotted path from a model config."""
    for key in path.split(".") if path else ():
        data = data[key]
    return data


def architecture(config, kind):
    """Keep the architecture fields used by the blog."""
    result = {}
    for source, target, names in ARCHITECTURE[kind]:
        values = at(config, source)
        destination = result.setdefault(target, {}) if target else result
        destination.update((name, values[name]) for name in names.split())
    return result


def fetch(spec):
    """Fetch a model config and its Safetensors headers."""
    metadata = get_safetensors_metadata(spec["model_id"], revision=spec["revision"])
    paths = sorted(metadata.files_metadata)
    files = HfApi().get_paths_info(spec["model_id"], paths, revision=spec["revision"], repo_type="model")
    config_file = hf_hub_download(spec["model_id"], "config.json", revision=spec["revision"])
    tensors = {}
    for shard in metadata.files_metadata.values():
        for name, tensor in shard.tensors.items():
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            parameters = math.prod(shape)
            tensors[name] = {"parameters": parameters, "bytes": parameters * DTYPE_BYTES[dtype],
                             "dtype": dtype, "shape": shape}
    return sum(file.size for file in files), tensors, json.loads(Path(config_file).read_text())


def classify(name):
    """Map one tensor name to a displayed family and group."""
    for pattern, family, group in RULES:
        if match := re.search(pattern, name.lower()):
            return family, group.format(*match.groups())
    return "other", "other"


def families(tensors):
    """Aggregate tensors into the profile's displayed groups."""
    grouped = defaultdict(lambda: defaultdict(list))
    for name, tensor in tensors.items():
        family, group = classify(name)
        grouped[family][group].append(tensor)
    result = {}
    for family in FAMILIES:
        if family not in grouped:
            continue
        groups = {}
        for name, rows in sorted(grouped[family].items()):
            dtypes = {tensor["dtype"] for tensor in rows}
            if len(dtypes) != 1:
                raise ValueError(f"mixed dtypes in {family}.{name}: {dtypes}")
            shapes = {tuple(tensor["shape"]) for tensor in rows}
            groups[name] = {
                "parameters": sum(tensor["parameters"] for tensor in rows),
                "bytes": sum(tensor["bytes"] for tensor in rows),
                "dtype": dtypes.pop(),
                "tensor_shape": list(shapes.pop()) if len(shapes) == 1 else None,
            }
        result[family] = total(groups)
    return result


def total(groups):
    """Add totals to a collection of parameter groups."""
    return {"parameters": sum(group["parameters"] for group in groups.values()),
            "bytes": sum(group["bytes"] for group in groups.values()), "param_groups": groups}


def view(family_data, bytes_on_disk=None):
    """Add totals to one stored or activated profile view."""
    result = {"bytes_accounted": sum(data["bytes"] for data in family_data.values()),
              "parameters": sum(data["parameters"] for data in family_data.values()),
              "families": family_data}
    if bytes_on_disk is not None:
        result = {"bytes_on_disk": bytes_on_disk, **result}
    return result


def activated(tensors, kind, config):
    """Apply one model's per-token MoE routing rules."""
    recipe = ACTIVATION[kind]
    layer = at(config, recipe["exclude_layer"]) if "exclude_layer" in recipe else None
    numerator, denominator = (at(config, path) for path in recipe["ratio"])
    active = {}
    for name, tensor in tensors.items():
        if ((recipe.get("exclude") and recipe["exclude"] in name.lower())
                or (layer is not None and re.search(rf"layers\.{layer}\.", name))):
            continue
        tensor = dict(tensor)
        if re.search(recipe["experts"], name):
            tensor["parameters"] = tensor["parameters"] * numerator // denominator
            tensor["bytes"] = tensor["bytes"] * numerator // denominator
            if "expert_axis" in recipe:
                tensor["shape"] = list(tensor["shape"])
                tensor["shape"][recipe["expert_axis"]] = numerator
        active[name] = tensor
    return view(families(active))


def hub_profile(spec):
    """Build one Hub model's stored and activated views."""
    bytes_on_disk, tensors, config = fetch(spec)
    sources = {"architecture": "hub_config_json",
               "stored_params": "hub_safetensors_headers + hub_paths_info"}
    profile = {"model_id": spec["model_id"], "architecture": architecture(config, spec["kind"]),
               "sources": sources, "stored_params": view(families(tensors), bytes_on_disk)}
    if spec["activation"]:
        sources["activated_params"] = ACTIVATION[spec["activation"]]["source"]
        profile["activated_params"] = activated(tensors, spec["activation"], config)
    return profile


def vaswani_profile():
    """Reconstruct the Transformer-base profile from the paper."""
    family_data = {}
    for family, values in VASWANI_TRANSFORMER["groups"].items():
        groups = {name: {"parameters": parameters, "bytes": parameters * 4, "dtype": "F32",
                         "tensor_shape": shape} for name, (parameters, shape) in values.items()}
        family_data[family] = total(groups)
    stored = view(family_data)
    return {"model_id": "Vaswani et al. Transformer", "architecture": VASWANI_TRANSFORMER["architecture"],
            "sources": {"architecture": "attention_is_all_you_need_table_3",
                        "stored_params": "reconstructed_from_paper"},
            "stored_params": {"bytes_on_disk": stored["bytes_accounted"], **stored}}


def profile_names():
    """List every profile generated by this script."""
    return ["vaswani_transformer_base.profile.yaml", *MODELS]


def build_profile(name):
    """Build one named profile."""
    return vaswani_profile() if name.startswith("vaswani_") else hub_profile(MODELS[name])


def main():
    """Write profiles, or compare rebuilt profiles with disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", choices=profile_names())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failed = []
    for name in args.profiles or profile_names():
        profile = build_profile(name)
        path = ROOT / name
        if args.check:
            if yaml.safe_load(path.read_text()) != profile:
                failed.append(name)
            print(("DIFF " if name in failed else "ok ") + name)
        else:
            path.write_text(yaml.safe_dump(profile, sort_keys=False))
            print(name)
    if failed:
        raise SystemExit(f"{len(failed)} profiles differ")


if __name__ == "__main__":
    main()
