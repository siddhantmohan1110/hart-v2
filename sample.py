import argparse
import copy
import datetime
import os
import random
import time

import numpy as np
import torch
import torchvision
from PIL import Image
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
SHARED_STATE_PATH = os.path.join(
    REPO_ROOT, "fhat", "fhat_kv_stage_3.pt"
)  # shared cache always uses stage 3

#  "tench",
#     "hen",
#     "magpie",
#     "quail",
#     "drake",
#     "conch",
#     "bittern",
#     "boxer",
#     "pug",
#     "chow",
#     "tabby",
#     "ladybug",
#     "bighorn",
#     "mink",
#     "quill"
def save_images(sample_imgs, sample_folder_dir, store_separately, prompts):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"{time.strftime("%Y%m%d_%H%M%S")}_sample_images.png"))
        print(f"Example images are saved to {sample_folder_dir}")
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(os.path.join(sample_folder_dir, f"{time.strftime("%Y%m%d_%H%M%S")}_{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


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

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()
    text_tokenizer_max_length = args.max_token_length

    # safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
    # safety_checker_model = AutoModelForCausalLM.from_pretrained(
    #     args.shield_model_path,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    # ).to(device)

    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_list:
        prompts = args.prompt_list
    else:
        print(
            "No prompt is provided. Will randomly sample 4 prompts from default prompts."
        )
        prompts = random.sample(default_prompts, 4)

    # for idx, prompt in enumerate(prompts):
    #     if safety_check.is_dangerous(
    #         safety_checker_tokenizer, safety_checker_model, prompt
    #     ):
    #         prompts[idx] = random.sample(default_prompts, 1)[0]
    #         print(
    #             f"Detected Unsafe prompt with index {idx}, will replace by one of default prompts."
    #         )

    # Load shared state (fhat_centroids) for specified cluster_ids
    # Use path relative to script location (one directory up from hart-v2/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    FHAT_CENTROIDS_PATH = os.path.join(SCRIPT_DIR, "..", "fhat_centroids.pt")
    # Default to cluster_id 14 if not specified
    cluster_ids = args.cluster_ids if args.cluster_ids is not None else [14] * len(prompts)
    shared_state_cache = None

    if os.path.exists(FHAT_CENTROIDS_PATH):
        fhat_centroids = torch.load(FHAT_CENTROIDS_PATH, map_location="cpu")
        print(f"Loaded fhat_centroids from {FHAT_CENTROIDS_PATH}")
        print(f"Available cluster IDs: {list(fhat_centroids.keys())}")

        # Ensure number of cluster_ids matches number of prompts
        if len(cluster_ids) != len(prompts):
            print(f"Warning: Number of cluster_ids ({len(cluster_ids)}) doesn't match number of prompts ({len(prompts)})")
            print(f"Will replicate first cluster_id to match batch size")
            cluster_ids = [cluster_ids[0]] * len(prompts)

        # Load and concatenate f_hats for each cluster_id
        fhat_list = []
        for cid in cluster_ids:
            if cid in fhat_centroids:
                fhat_list.append(fhat_centroids[cid])
                print(f"Loaded f_hat from cluster {cid} with shape {fhat_centroids[cid].shape}")
            else:
                print(f"Warning: Cluster ID {cid} not found in fhat_centroids")
                fhat_list.append(None)

        # Check if all f_hats were loaded successfully
        if all(f is not None for f in fhat_list):
            # Concatenate along batch dimension (dim=0)
            concatenated_fhat = torch.cat(fhat_list, dim=0)
            shared_state_cache = {'f_hat': concatenated_fhat}
            print(f"Concatenated f_hats from {len(cluster_ids)} clusters, final shape: {concatenated_fhat.shape}")
        else:
            print(f"Warning: Some cluster IDs were not found, proceeding without shared state")
    else:
        print(f"Warning: fhat_centroids not found at {FHAT_CENTROIDS_PATH}")

    alpha_stage = 3  # Use stage 3 for clustering

    start_time = time.time()
    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
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
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            infer_func = (
                ema_model.autoregressive_infer_cfg
                if args.use_ema
                else model.autoregressive_infer_cfg
            )
            output_imgs = infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=args.cfg,
                g_seed=args.seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
                save_fhat=False,
                is_shared_hart=True,
                alpha=alpha_stage,
                shared_state=shared_state_cache,
            )

    total_time = time.time() - start_time
    print(f"Generate {len(prompts)} images take {total_time:2f}s.")

    save_images(
        output_imgs.clone(), args.sample_folder_dir, args.store_seperately, prompts
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
        "--store_seperately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )
    parser.add_argument(
        "--cluster_ids",
        nargs='+',
        type=int,
        help="Cluster IDs to use f_hats from fhat_centroids.pt (one per prompt, space-separated)",
        default=None,
    )
    args = parser.parse_args()

    main(args)
