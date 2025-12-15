import argparse
import copy
import datetime
import json
import os
import random
import textwrap
import time

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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARED_FHAT_PATH = os.path.join(REPO_ROOT, "fhat_centroids.pt")
CLUSTER_PROMPTS_PATH = os.path.join(REPO_ROOT, "cluster_prompts.json")


def _annotate_tensor(img_tensor, caption):
    """Add a caption strip to an image tensor and return a float tensor in [0,1]."""
    img_uint8 = img_tensor.clamp(0, 1).mul(255).to(torch.uint8)
    pil_img = Image.fromarray(img_uint8.permute(1, 2, 0).cpu().numpy())

    bar_height = 32
    new_img = Image.new("RGB", (pil_img.width, pil_img.height + bar_height), color=(0, 0, 0))
    new_img.paste(pil_img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default()
    caption = textwrap.shorten(caption, width=120, placeholder="…")
    draw.text((4, pil_img.height + 8), caption, fill=(255, 255, 255), font=font)

    return pil_to_tensor(new_img).float().div(255.0)


def _save_grid(images, prompts, cluster_id, out_path, nrow=8):
    captioned = [
        _annotate_tensor(img, f"cluster {cluster_id} | {prompt}") for img, prompt in zip(images, prompts)
    ]
    grid = torchvision.utils.make_grid(torch.stack(captioned), nrow=nrow)
    grid_np = grid.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(grid_np).save(out_path)


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
    alpha_stage = 3

    # Load cluster prompts and fhat centroids
    cluster_prompts_path = args.cluster_prompts_path or CLUSTER_PROMPTS_PATH
    fhat_path = args.fhat_path or SHARED_FHAT_PATH
    with open(cluster_prompts_path) as f:
        cluster_prompts = json.load(f)
    fhat_centroids = torch.load(fhat_path, map_location="cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):
            for cluster_id_str, prompts in cluster_prompts.items():
                cluster_id = int(cluster_id_str)
                out_prefix = os.path.join(args.output_dir, f"cluster_{cluster_id}")

                if not args.skip_shared and cluster_id in fhat_centroids:
                    f_hat = fhat_centroids[cluster_id].to(device)
                    imgs_shared = []
                    for batch_prompts in _batched_list(prompts, args.batch_size):
                        shared_state = {"f_hat": _repeat_fhat(f_hat, len(batch_prompts))}
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
                            shared_state=shared_state,
                            is_shared=True,
                        )
                        imgs_shared.extend(list(imgs))
                    if imgs_shared:
                        _save_grid(
                            imgs_shared,
                            prompts,
                            cluster_id,
                            f"{out_prefix}_shared.png",
                            nrow=args.grid_nrow,
                        )

                if args.generate_individual:
                    indiv_images = []
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
                        indiv_images.extend(list(imgs))
                    if indiv_images:
                        _save_grid(
                            indiv_images,
                            prompts,
                            cluster_id,
                            f"{out_prefix}_individual.png",
                            nrow=args.grid_nrow,
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="hart-0.7b-1024px/llm",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="pretrained_models/shieldgemma-2b",
    )
    parser.add_argument("--prompt", type=str, help="A single prompt.", default="")
    parser.add_argument("--prompt_list", nargs='+', type=str, help="Multiple prompts (space-separated)", default=None)
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
        "--sample_folder_dir",
        type=str,
        help="The folder where the image samples are stored",
        default="samples/",
    )
    parser.add_argument(
        "--cluster_prompts_path",
        type=str,
        help="Path to cluster_prompts.json generated by clustering_test.py.",
        default=CLUSTER_PROMPTS_PATH,
    )
    parser.add_argument(
        "--fhat_path",
        type=str,
        help="Path to fhat_centroids.pt generated by clustering_test.py.",
        default=SHARED_FHAT_PATH,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Where to write cluster grids.",
        default="cluster_samples",
    )
    parser.add_argument(
        "--grid_nrow",
        type=int,
        help="Number of images per row in saved grids.",
        default=8,
    )
    parser.add_argument(
        "--skip_shared",
        action="store_true",
        help="Skip shared-state generation; only run individual prompts.",
    )
    parser.add_argument(
        "--generate_individual",
        action="store_true",
        help="Generate per-prompt images without shared fhat (baseline).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for prompt processing to avoid CUDA OOM.",
        default=8,
    )
    args = parser.parse_args()

    main(args)
