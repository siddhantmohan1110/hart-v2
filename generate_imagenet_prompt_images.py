import argparse
import copy
import os
import re
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, set_seed

from hart.modules.models.transformer import HARTForT2I
from hart.utils import encode_prompts, llm_system_prompt


def _sanitize_filename(prompt: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return safe or "prompt"


def load_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def save_images(
    sample_imgs: torch.Tensor,
    prompts: list[str],
    output_dir: str,
    resize_to: int,
    prompt_indices: list[int],
    sample_idx: int,
) -> list[str]:
    sample_imgs_np = (
        sample_imgs.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )  # (B, 3, H, W)
    os.makedirs(output_dir, exist_ok=True)
    saved_files: list[str] = []
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC

    for prompt_idx, prompt, img_np in zip(prompt_indices, prompts, sample_imgs_np):
        pil_img = Image.fromarray(np.transpose(img_np, (1, 2, 0)))
        if resize_to:
            pil_img = pil_img.resize((resize_to, resize_to), resample=resample)
        fname = f"{prompt_idx:04d}_{sample_idx:02d}_{_sanitize_filename(prompt)}.png"
        pil_img.save(os.path.join(output_dir, fname))
        saved_files.append(fname)
    return saved_files


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device is required for HART inference.")

    set_seed(args.seed)
    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    model = HARTForT2I.from_pretrained(args.model_path).to(device)
    model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(torch.load(os.path.join(args.model_path, "ema_model.bin")))
    else:
        ema_model = None

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_log = []

    infer_func = ema_model.autoregressive_infer_cfg if args.use_ema else model.autoregressive_infer_cfg
    total_start = time.time()

    for sample_idx in range(args.num_images_per_prompt):
        seed = args.seed + sample_idx
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            prompt_indices = list(range(start, start + len(batch_prompts)))
            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                batch_prompts,
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.float16):
                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    save_fhat=False,
                    is_shared_hart=False,
                )

            saved_files = save_images(
                output_imgs.clone(),
                batch_prompts,
                args.output_dir,
                args.resize_to,
                prompt_indices=prompt_indices,
                sample_idx=sample_idx,
            )
            for fname, prompt in zip(saved_files, batch_prompts):
                prompt_log.append(f"{fname}\t{prompt}")
            print(
                f"Sample {sample_idx + 1}/{args.num_images_per_prompt}, "
                f"batch {start // args.batch_size + 1} saved ({len(batch_prompts)} images)."
            )

    total_time = time.time() - total_start
    print(
        f"Generated {len(prompts) * args.num_images_per_prompt} images in {total_time:.2f}s. "
        f"Outputs in {args.output_dir}"
    )

    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        f.write("\n".join(prompt_log))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate an image for each ImageNet label using HART."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="hart-0.7b-1024px/llm",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, defaults to Qwen2-VL-1.5B-Instruct.",
        default="Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="File with one prompt per line.",
        default="data/imagenet_classes.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to store generated images.",
        default="data/imagenet_prompt_images",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompts per batch.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg",
        type=float,
        help="Classifier-free guidance scale.",
        default=4.5,
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    parser.add_argument(
        "--resize_to",
        type=int,
        help="Resize each saved image to this square resolution.",
        default=256,
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        help="Number of images to generate for each prompt/label.",
        default=50,
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)
