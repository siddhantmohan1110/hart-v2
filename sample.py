import argparse
import copy
import datetime
import json
import os
import random
import re
import textwrap
import time
from time import time as now

import numpy as np
import torch
import torchvision
from PIL import Image
from PIL import ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
from hart.utils.datasets import load_mjhq

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _annotate_tensor(img_tensor, caption, scale=1.0):
    """Add an overlaid caption to an image tensor and return a float tensor in [0,1]."""
    img_uint8 = img_tensor.clamp(0, 1).mul(255).to(torch.uint8)
    pil_img = Image.fromarray(img_uint8.permute(1, 2, 0).cpu().numpy())

    if scale is not None and scale < 1.0:
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        new_size = (max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale)))
        pil_img = pil_img.resize(new_size, resample=resample)

    img_rgba = pil_img.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)
    font = ImageFont.load_default()
    caption = textwrap.shorten(caption, width=120, placeholder="…")
    text_w, text_h = draw.textsize(caption, font=font)
    pad = 4
    rect_w = min(pil_img.width, text_w + 2 * pad)
    rect_h = text_h + 2 * pad
    rect_x0 = 0
    rect_y0 = pil_img.height - rect_h
    rect_y0 = max(0, rect_y0)
    draw.rectangle(
        [(rect_x0, rect_y0), (rect_x0 + rect_w, rect_y0 + rect_h)],
        fill=(0, 0, 0, 180),
    )
    draw.text((rect_x0 + pad, rect_y0 + pad), caption, fill=(255, 255, 255, 255), font=font)

    annotated = img_rgba.convert("RGB")
    return pil_to_tensor(annotated).float().div(255.0)


def _save_grid(images, prompts, cluster_id, out_path, nrow=8, scale=1.0):
    captioned = [
        _annotate_tensor(img, f"cluster {cluster_id} | {prompt}", scale=scale) for img, prompt in zip(images, prompts)
    ]
    grid = torchvision.utils.make_grid(torch.stack(captioned), nrow=nrow)
    grid_np = grid.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(grid_np).save(out_path)


def _sanitize_filename(prompt: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return safe or "prompt"


def _save_images(
    sample_imgs: torch.Tensor,
    prompts: list[str],
    output_dir: str,
    prompt_indices: list[int],
    sample_idx: int,
    resize_to: int | None,
    prefix: str,
):
    sample_imgs_np = (
        sample_imgs.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )  # (B, 3, H, W)
    os.makedirs(output_dir, exist_ok=True)
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC

    for prompt_idx, prompt, img_np in zip(prompt_indices, prompts, sample_imgs_np):
        pil_img = Image.fromarray(np.transpose(img_np, (1, 2, 0)))
        if resize_to:
            pil_img = pil_img.resize((resize_to, resize_to), resample=resample)
        fname = f"{prefix}{prompt_idx:04d}_{sample_idx:02d}_{_sanitize_filename(prompt)}.png"
        pil_img.save(os.path.join(output_dir, fname))


def _repeat_fhat(f_hat, batch_size):
    repeat_shape = [batch_size] + [1] * (f_hat.dim() - 1)
    return f_hat.repeat(*repeat_shape)


def _batched_list(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _generate_images(
    prompts,
    text_model,
    text_tokenizer,
    infer_func,
    use_llm_system_prompt,
    max_token_length,
    cfg,
    seed,
    more_smooth,
    alpha_stage,
    shared_state=None,
    is_shared=True,
):
    (
        context_tokens,
        context_mask,
        context_position_ids,
        context_tensor,
    ) = encode_prompts(
        prompts,
        text_model,
        text_tokenizer,
        max_token_length,
        llm_system_prompt,
        use_llm_system_prompt,
    )

    output_imgs = infer_func(
        B=context_tensor.size(0),
        label_B=context_tensor,
        cfg=cfg,
        g_seed=seed,
        more_smooth=more_smooth,
        context_position_ids=context_position_ids,
        context_mask=context_mask,
        save_fhat=False,
        is_shared_hart=is_shared,
        alpha=alpha_stage,
        shared_state=shared_state,
    )
    return output_imgs


def _load_prompts_for_baseline(args):
    """Load prompts when running baseline (non-shared) inference."""
    dataset = (args.dataset or "").lower()
    if dataset == "imagenet":
        with open(args.imagenet_class_labels_path) as f:
            return [x.strip() for x in f.readlines()]
    if dataset == "mjhq":
        return load_mjhq(args.mjhq_metadata_path)
    return default_prompts


def main(args):
    device = torch.device("cuda")

    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, padding_side="left")
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    infer_func = (
        ema_model.autoregressive_infer_cfg
        if args.use_ema
        else model.autoregressive_infer_cfg
    )
    alpha_stage = args.alpha
    grid_scale = 1.0 if args.grid_full_res else 0.5
    image_resize = args.resize_individual_to
    timings = {"individual_generation": {}, "meta": {}}

    if args.shared_hart and not args.clustered_prompts:
        raise ValueError("shared_hart requires clustered_prompts to be set.")

    # Resolve experiment-scoped paths
    experiment_dir = args.experiment_name
    os.makedirs(experiment_dir, exist_ok=True)

    cluster_prompts_path = args.cluster_prompts_path or os.path.join(
        experiment_dir, f"{args.experiment_name}_cluster_prompts.json"
    )
    fhat_path = args.fhat_path or os.path.join(
        experiment_dir, f"{args.experiment_name}_fhat_centroids.pt"
    )

    # Load cluster prompts and fhat centroids when requested
    cluster_prompts = {}
    fhat_centroids = {}
    if args.clustered_prompts:
        with open(cluster_prompts_path) as f:
            cluster_prompts = json.load(f)
        if args.shared_hart:
            fhat_centroids = torch.load(fhat_path, map_location="cuda")

    shared_grid_dir = os.path.join(experiment_dir, "cluster_grids_shared")
    nonshared_grid_dir = os.path.join(experiment_dir, "cluster_grids_nonshared")
    baseline_grid_dir = os.path.join(experiment_dir, "baseline_grids")
    shared_img_dir = os.path.join(experiment_dir, "cluster_images_shared")
    nonshared_img_dir = os.path.join(experiment_dir, "cluster_images_nonshared")
    baseline_img_dir = os.path.join(experiment_dir, "baseline_images")
    timing_output_path = os.path.join(experiment_dir, "timing_profile_inference.json")

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):
            # Clustered workflow
            if args.clustered_prompts:
                cluster_items = sorted(cluster_prompts.items(), key=lambda kv: int(kv[0]))
                if args.shared_hart:
                    for cluster_id_str, prompts in cluster_items:
                        cluster_id = int(cluster_id_str)
                        if cluster_id not in fhat_centroids:
                            continue
                        f_hat = fhat_centroids[cluster_id].to(device)

                        if args.generate_grids:
                            shared_imgs = []
                            if args.enable_timing and args.warmup_iterations > 0:
                                pass  # grid timing excluded; skip warmup here
                            for batch_prompts in _batched_list(prompts, args.batch_size):
                                batch_state = {"f_hat": _repeat_fhat(f_hat, len(batch_prompts))}
                                imgs = _generate_images(
                                    batch_prompts,
                                    text_model,
                                    text_tokenizer,
                                    infer_func,
                                    args.use_llm_system_prompt,
                                    args.max_token_length,
                                    args.cfg,
                                    args.seed,
                                    args.more_smooth,
                                    alpha_stage,
                                    shared_state=batch_state,
                                    is_shared=True,
                                )
                                shared_imgs.extend(list(imgs))
                            if shared_imgs:
                                os.makedirs(shared_grid_dir, exist_ok=True)
                                _save_grid(
                                    shared_imgs,
                                    prompts,
                                    cluster_id,
                                    os.path.join(shared_grid_dir, f"cluster_{cluster_id}_shared.png"),
                                    nrow=args.grid_nrow,
                                    scale=grid_scale,
                                )

                        if args.save_individual_images:
                            os.makedirs(shared_img_dir, exist_ok=True)
                            for sample_idx in range(args.num_images_per_prompt):
                                seed = args.seed + sample_idx
                                sample_imgs = []
                                if args.enable_timing:
                                    sample_start = now()
                                    if args.warmup_iterations > 0 and not args.generate_grids:
                                        for _ in range(args.warmup_iterations):
                                            for start_w in range(0, len(prompts), args.batch_size):
                                                warmup_batch = prompts[start_w : start_w + args.batch_size]
                                                warmup_state = {"f_hat": _repeat_fhat(f_hat, len(warmup_batch))}
                                                tmp = _generate_images(
                                                    warmup_batch,
                                                    text_model,
                                                    text_tokenizer,
                                                    infer_func,
                                                    args.use_llm_system_prompt,
                                                    args.max_token_length,
                                                    args.cfg,
                                                    seed,
                                                    args.more_smooth,
                                                    alpha_stage,
                                                    shared_state=warmup_state,
                                                    is_shared=True,
                                                )
                                for start in range(0, len(prompts), args.batch_size):
                                    batch_prompts = prompts[start : start + args.batch_size]
                                    batch_state = {"f_hat": _repeat_fhat(f_hat, len(batch_prompts))}
                                    imgs = _generate_images(
                                        batch_prompts,
                                        text_model,
                                        text_tokenizer,
                                        infer_func,
                                        args.use_llm_system_prompt,
                                        args.max_token_length,
                                        args.cfg,
                                        seed,
                                        args.more_smooth,
                                        alpha_stage,
                                        shared_state=batch_state,
                                        is_shared=True,
                                    )
                                    sample_imgs.extend(list(imgs))
                                if sample_imgs:
                                    _save_images(
                                        torch.stack(sample_imgs),
                                        prompts,
                                        shared_img_dir,
                                        prompt_indices=list(range(len(prompts))),
                                        sample_idx=sample_idx,
                                        resize_to=image_resize,
                                        prefix=f"cluster_{cluster_id}_shared_",
                                    )
                        if args.enable_timing:
                            timings["individual_generation"].setdefault("shared_clusters", {})[
                                str(cluster_id)
                            ] = now() - sample_start
                else:
                    for cluster_id_str, prompts in cluster_items:
                        cluster_id = int(cluster_id_str)

                        if args.generate_grids:
                            cluster_imgs = []
                            if args.enable_timing and args.warmup_iterations > 0:
                                pass  # grid timing excluded; skip warmup here
                            for batch_prompts in _batched_list(prompts, args.batch_size):
                                imgs = _generate_images(
                                    batch_prompts,
                                    text_model,
                                    text_tokenizer,
                                    infer_func,
                                    args.use_llm_system_prompt,
                                    args.max_token_length,
                                    args.cfg,
                                    args.seed,
                                    args.more_smooth,
                                    alpha_stage,
                                    shared_state=None,
                                    is_shared=False,
                                )
                                cluster_imgs.extend(list(imgs))
                            if cluster_imgs:
                                os.makedirs(nonshared_grid_dir, exist_ok=True)
                                _save_grid(
                                    cluster_imgs,
                                    prompts,
                                    cluster_id,
                                    os.path.join(nonshared_grid_dir, f"cluster_{cluster_id}_individual.png"),
                                    nrow=args.grid_nrow,
                                    scale=grid_scale,
                                )

                        if args.save_individual_images:
                            os.makedirs(nonshared_img_dir, exist_ok=True)
                            for sample_idx in range(args.num_images_per_prompt):
                                seed = args.seed + sample_idx
                                sample_imgs = []
                                if args.enable_timing:
                                    sample_start = now()
                                    if args.warmup_iterations > 0 and not args.generate_grids:
                                        for _ in range(args.warmup_iterations):
                                            for start_w in range(0, len(prompts), args.batch_size):
                                                warmup_batch = prompts[start_w : start_w + args.batch_size]
                                                tmp = _generate_images(
                                                    warmup_batch,
                                                    text_model,
                                                    text_tokenizer,
                                                    infer_func,
                                                    args.use_llm_system_prompt,
                                                    args.max_token_length,
                                                    args.cfg,
                                                    seed,
                                                    args.more_smooth,
                                                    alpha_stage,
                                                    shared_state=None,
                                                    is_shared=False,
                                                )
                                for start in range(0, len(prompts), args.batch_size):
                                    batch_prompts = prompts[start : start + args.batch_size]
                                    imgs = _generate_images(
                                        batch_prompts,
                                        text_model,
                                        text_tokenizer,
                                        infer_func,
                                        args.use_llm_system_prompt,
                                        args.max_token_length,
                                        args.cfg,
                                        seed,
                                        args.more_smooth,
                                        alpha_stage,
                                        shared_state=None,
                                        is_shared=False,
                                    )
                                    sample_imgs.extend(list(imgs))
                                if sample_imgs:
                                    _save_images(
                                        torch.stack(sample_imgs),
                                        prompts,
                                        nonshared_img_dir,
                                        prompt_indices=list(range(len(prompts))),
                                        sample_idx=sample_idx,
                                        resize_to=image_resize,
                                        prefix=f"cluster_{cluster_id}_individual_",
                                    )
                        if args.enable_timing:
                            timings["individual_generation"].setdefault("nonshared_clusters", {})[
                                str(cluster_id)
                            ] = now() - sample_start

            # Baseline (non-shared) generation using dataset or default prompts
            if not args.clustered_prompts:
                baseline_prompts = _load_prompts_for_baseline(args)

                if args.generate_grids:
                    baseline_imgs = []
                    if args.enable_timing and args.warmup_iterations > 0:
                        pass  # grid timing excluded; skip warmup here
                    for batch_prompts in _batched_list(baseline_prompts, args.batch_size):
                        imgs = _generate_images(
                            batch_prompts,
                            text_model,
                            text_tokenizer,
                            infer_func,
                            args.use_llm_system_prompt,
                            args.max_token_length,
                            args.cfg,
                            args.seed,
                            args.more_smooth,
                            alpha_stage,
                            shared_state=None,
                            is_shared=False,
                        )
                        baseline_imgs.extend(list(imgs))
                    if baseline_imgs:
                        os.makedirs(baseline_grid_dir, exist_ok=True)
                        _save_grid(
                            baseline_imgs,
                            baseline_prompts,
                            f"{args.dataset or 'baseline'}",
                            os.path.join(baseline_grid_dir, "baseline_individual.png"),
                            nrow=args.grid_nrow,
                            scale=grid_scale,
                        )

                if args.save_individual_images:
                    os.makedirs(baseline_img_dir, exist_ok=True)
                    for sample_idx in range(args.num_images_per_prompt):
                        seed = args.seed + sample_idx
                        sample_imgs = []
                        if args.enable_timing:
                            sample_start = now()
                            if args.warmup_iterations > 0 and not args.generate_grids:
                                for _ in range(args.warmup_iterations):
                                    for start_w in range(0, len(baseline_prompts), args.batch_size):
                                        warmup_batch = baseline_prompts[start_w : start_w + args.batch_size]
                                        tmp = _generate_images(
                                            warmup_batch,
                                            text_model,
                                            text_tokenizer,
                                            infer_func,
                                            args.use_llm_system_prompt,
                                            args.max_token_length,
                                            args.cfg,
                                            seed,
                                            args.more_smooth,
                                            alpha_stage,
                                            shared_state=None,
                                            is_shared=False,
                                        )
                        for start in range(0, len(baseline_prompts), args.batch_size):
                            batch_prompts = baseline_prompts[start : start + args.batch_size]
                            imgs = _generate_images(
                                batch_prompts,
                                text_model,
                                text_tokenizer,
                                infer_func,
                                args.use_llm_system_prompt,
                                args.max_token_length,
                                args.cfg,
                                seed,
                                args.more_smooth,
                                alpha_stage,
                                shared_state=None,
                                is_shared=False,
                            )
                            sample_imgs.extend(list(imgs))
                        if sample_imgs:
                            _save_images(
                                torch.stack(sample_imgs),
                                baseline_prompts,
                                baseline_img_dir,
                                prompt_indices=list(range(len(baseline_prompts))),
                                sample_idx=sample_idx,
                                resize_to=image_resize,
                                prefix="baseline_",
                            )
                    if args.enable_timing:
                        timings["individual_generation"]["baseline"] = now() - sample_start

    # Persist timing profile if requested
    if args.enable_timing:
        os.makedirs(experiment_dir, exist_ok=True)
        timings["meta"].update(
            {
                "shared_hart": args.shared_hart,
                "clustered_prompts": args.clustered_prompts,
                "num_images_per_prompt": args.num_images_per_prompt,
                "batch_size": args.batch_size,
                "resize_individual_to": args.resize_individual_to,
                "grid_scale": grid_scale,
                "warmup_iterations": args.warmup_iterations,
                "seed": args.seed,
            }
        )
        with open(timing_output_path, "w") as f:
            json.dump(timings, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ***********************************************************
    # Models
    # ***********************************************************
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="./../saved_models/hart-0.7b-1024px/llm",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="./../saved_models/Qwen2-VL-1.5B-Instruct/",
    )
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="./../saved_models/shieldgemma-2b",
    )

    # ***********************************************************
    # Prompts
    # ***********************************************************
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment folder to pull cluster outputs from (matches clustering_test.py).",
        default="exp1",
    )
    parser.add_argument(
        "--clustered_prompts",
        action="store_true",
        help="Use clustered prompts/f_hat outputs from an experiment (otherwise baseline prompts).",
    )
    parser.add_argument(
        "--shared_hart",
        action="store_true",
        help="Use shared f_hat centroids for clustered prompts (requires alpha).",
    )
    parser.add_argument(
        "--cluster_prompts_path",
        type=str,
        help="Path to cluster_prompts.json (defaults to experiment_name/{experiment_name}_cluster_prompts.json).",
        default=None,
    )
    parser.add_argument(
        "--fhat_path",
        type=str,
        help="Path to fhat_centroids.pt (defaults to experiment_name/{experiment_name}_fhat_centroids.pt).",
        default=None,
    )

    # ***********************************************************
    # Data for baseline case
    # ***********************************************************
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenet", "mjhq"],
        help="Dataset to use for baseline (non-shared) prompts. Defaults to built-in prompts.",
        default=None,
    )
    parser.add_argument(
        "--imagenet_class_labels_path",
        type=str,
        help="Path to ImageNet class labels (used when dataset=imagenet).",
        default="./../data/ImageNet/imagenet_classes.txt",
    )
    parser.add_argument(
        "--mjhq_metadata_path",
        type=str,
        help="Path to MJHQ meta_data.json (used when dataset=mjhq).",
        default="./../data/MJHQ-30K/meta_data.json",
    )

    # ***********************************************************
    # Inference
    # ***********************************************************
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    parser.add_argument(
        "--alpha",
        type=int,
        help="Stage index up to which shared HART is used (required if --shared_hart).",
        default=3,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for prompt processing to avoid CUDA OOM.",
        default=8,
    )

    # ***********************************************************
    # Grid outputs
    # ***********************************************************
    parser.add_argument(
        "--generate_grids",
        action="store_true",
        help="Generate and save grids (clustered or baseline, depending on mode).",
    )
    parser.add_argument(
        "--grid_nrow",
        type=int,
        help="Number of images per row in saved grids.",
        default=8,
    )
    parser.add_argument(
        "--grid_full_res",
        action="store_true",
        help="Save grid images at original resolution (otherwise downsampled to keep grids compact).",
    )

    # ***********************************************************
    # Individual image outputs
    # ***********************************************************
    parser.add_argument(
        "--save_individual_images",
        action="store_true",
        help="Save individual images for each prompt (shared, non-shared, and baseline).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        help="Number of images to generate per prompt when saving individual images.",
        default=1,
    )
    parser.add_argument(
        "--resize_individual_to",
        type=int,
        help="Resize saved individual images to this square resolution (omit to keep original).",
        default=256,
    )

    # ***********************************************************
    # Latency profiling
    # ***********************************************************
    parser.add_argument(
        "--enable_timing",
        action="store_true",
        help="Profile individual image generation (excludes grid creation) and save timing JSON.",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        help="GPU warmup iterations before timing (only when --enable_timing).",
        default=50,
    )
    args = parser.parse_args()

    main(args)
