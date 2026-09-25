"""Regression tests for the profiles published with this blog."""

import unittest

import yaml

from build_profiles import ROOT, build_profile, classify, families, profile_names


class ProfileBuilderTests(unittest.TestCase):
    def test_representative_tensor_names(self):
        cases = {
            "bert.encoder.layer.0.attention.self.query.weight": ("attention", "projection_weights"),
            "model.layers.0.self_attn.q_proj.weight": ("attention", "q_proj"),
            "model.layers.3.mlp.experts.7.down_proj.weight": ("dense", "expert_weights"),
            "model.layers.0.mlp.experts.down_proj_scale_inv": ("other", "expert_quant_scales"),
            "model.visual.pos_embed.weight": ("other", "vision_positional_embedding"),
            "model.visual.blocks.0.norm1.weight": ("other", "vision_norm"),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify(name), expected)

    def test_mixed_dtype_group_is_rejected(self):
        tensors = {
            "unknown.a": {"parameters": 1, "bytes": 2, "dtype": "BF16", "shape": [1]},
            "unknown.b": {"parameters": 1, "bytes": 4, "dtype": "F32", "shape": [1]},
        }
        with self.assertRaisesRegex(ValueError, "mixed dtypes"):
            families(tensors)

    def test_every_published_profile_rebuilds(self):
        published = {path.name for path in ROOT.glob("*.profile.yaml")}
        self.assertEqual(set(profile_names()), published)
        for name in profile_names():
            with self.subTest(profile=name):
                expected = yaml.safe_load((ROOT / name).read_text())
                built = build_profile(name)
                self.assertEqual(built, expected)
                self.assert_consistent(built)

    def test_profiles_support_blog_claims(self):
        load = lambda name: yaml.safe_load((ROOT / name).read_text())
        self.assertEqual(load("vaswani_transformer_base.profile.yaml")["stored_params"]["parameters"], 63_045_632)
        deepseek = load("deepseek_r1.profile.yaml")
        self.assertEqual(deepseek["stored_params"]["bytes_accounted"], 688_574_839_360)
        self.assertEqual(round(deepseek["activated_params"]["bytes_accounted"] / 1e9, 1), 39.5)
        deepseek_other = deepseek["activated_params"]["families"]["other"]["param_groups"]
        self.assertEqual(deepseek_other["expert_quant_scales"]["parameters"], 1_247_232)
        self.assertEqual(deepseek_other["quant_scales"]["parameters"], 926_808)
        qwen = load("qwen3_vl_30b_a3b_instruct_fp8.profile.yaml")
        self.assertEqual(qwen["stored_params"]["bytes_accounted"], 32_251_808_224)
        self.assertEqual(qwen["activated_params"]["bytes_accounted"], 3_988_819_968)
        stored = qwen["stored_params"]["families"]
        active = qwen["activated_params"]["families"]
        self.assertEqual(stored["other"]["param_groups"]["expert_quant_scales"]["parameters"], 1_769_472)
        self.assertEqual(stored["other"]["param_groups"]["quant_scales"]["parameters"], 55_296)
        self.assertEqual(stored["dense"]["param_groups"]["expert_weights"]["parameters"], 28_991_029_248)
        self.assertEqual(active["dense"]["parameters"], 1_811_939_328)

    def assert_consistent(self, profile):
        for view_name in ("stored_params", "activated_params"):
            if view_name not in profile:
                continue
            view = profile[view_name]
            self.assertEqual(view["parameters"], sum(f["parameters"] for f in view["families"].values()))
            self.assertEqual(view["bytes_accounted"], sum(f["bytes"] for f in view["families"].values()))
            for family in view["families"].values():
                groups = family["param_groups"].values()
                self.assertEqual(family["parameters"], sum(group["parameters"] for group in groups))
                self.assertEqual(family["bytes"], sum(group["bytes"] for group in family["param_groups"].values()))


if __name__ == "__main__":
    unittest.main()
