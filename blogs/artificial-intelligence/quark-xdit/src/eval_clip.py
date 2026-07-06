#!/usr/bin/env python3
"""CLIP score evaluator for the FLUX MXFP4 ASM + compile study.

CLIP score = 100 * max(cos(image_embed, text_embed), 0), the standard
convention. Uses transformers.CLIPModel directly rather than
torchmetrics.CLIPScore, because torchmetrics 1.9 is incompatible with
transformers 5.x (get_image_features now returns BaseModelOutputWithPooling,
not a tensor — torchmetrics calls .norm() on it and crashes). The projected
embedding is taken from .pooler_output (transformers 5.x) or used directly
(transformers 4.x returned a tensor).

Prompt order is recovered from the benchmark's gen_results.json when present
(so it matches the saved PNGs exactly); otherwise it falls back to the first
N COCO captions in file order.

Example:
    python3 eval_clip.py \
        --image_dir /workspace/runs/mxfp4_compile_100/images \
        --output_dir /workspace/runs/mxfp4_compile_100/clip_vitb16 \
        --coco_annotations /workspace/data/coco/annotations/captions_val2017.json \
        --num_prompts 100 \
        --model_name openai/clip-vit-base-patch16
"""

import argparse
import glob
import json
import os
import time

import torch
from PIL import Image


def load_coco_prompts(path: str, n: int) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    seen, prompts = set(), []
    for ann in data["annotations"]:
        if ann["image_id"] not in seen:
            seen.add(ann["image_id"])
            prompts.append(ann["caption"].strip())
        if len(prompts) >= n:
            break
    return prompts


def resolve_prompts(args, n_images: int) -> list[str]:
    """Prefer the gen_results.json next to the image dir for exact order."""
    run_dir = os.path.dirname(os.path.abspath(args.image_dir.rstrip("/")))
    gr = os.path.join(run_dir, "gen_results.json")
    if os.path.isfile(gr):
        d = json.load(open(gr))
        ps = [s["prompt"] for s in d.get("per_sample", [])]
        if len(ps) >= n_images:
            return ps[:n_images]
    return load_coco_prompts(args.coco_annotations, n_images)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--coco_annotations", required=True)
    ap.add_argument("--num_prompts", type=int, default=100)
    ap.add_argument("--model_name", default="openai/clip-vit-base-patch16",
                    help="Any Hugging Face CLIP id.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    if not paths:
        raise SystemExit(f"No PNGs in {args.image_dir}")
    paths = paths[: args.num_prompts]
    prompts = resolve_prompts(args, len(paths))
    assert len(paths) == len(prompts), f"{len(paths)} imgs vs {len(prompts)} prompts"

    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CLIP model: {args.model_name} on {dev}")
    model = CLIPModel.from_pretrained(args.model_name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.model_name)

    scores = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (p, prompt) in enumerate(zip(paths, prompts)):
            img = Image.open(p).convert("RGB")
            inp = proc(text=[prompt], images=[img], return_tensors="pt",
                       padding=True, truncation=True).to(dev)
            img_emb = model.get_image_features(pixel_values=inp["pixel_values"])
            txt_emb = model.get_text_features(input_ids=inp["input_ids"],
                                              attention_mask=inp.get("attention_mask"))
            if not torch.is_tensor(img_emb):   # transformers 5.x
                img_emb = img_emb.pooler_output
            if not torch.is_tensor(txt_emb):
                txt_emb = txt_emb.pooler_output
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
            cos = (img_emb * txt_emb).sum(-1).clamp(min=0)
            scores.append(100.0 * cos.item())
            if (i + 1) % 25 == 0:
                print(f"  CLIP {i + 1}/{len(paths)} (latest={scores[-1]:.3f})")
    took = time.perf_counter() - t0

    mean = sum(scores) / len(scores)
    smin, smax = min(scores), max(scores)
    print(f"CLIP Score: mean={mean:.4f} min={smin:.4f} max={smax:.4f} "
          f"({len(scores)} imgs, {took:.1f}s)")

    res = {
        "metric": "clip", "model": args.model_name,
        "num_images": len(scores),
        "score": {"mean": mean, "min": smin, "max": smax},
        "per_image": [
            {"idx": i, "prompt": prompts[i], "image": paths[i], "score": scores[i]}
            for i in range(len(scores))
        ],
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.output_dir, "eval_summary.txt"), "w") as f:
        f.write(
            "CLIP Evaluation\n" + "=" * 50 + "\n"
            + f"Model     : {args.model_name}\n"
            + f"Images    : {len(scores)}\n"
            + f"CLIP mean : {mean:.4f}\n"
            + f"CLIP min  : {smin:.4f}\n"
            + f"CLIP max  : {smax:.4f}\n"
        )
    print(f"[done] {os.path.join(args.output_dir, 'eval_summary.txt')}")


if __name__ == "__main__":
    main()
