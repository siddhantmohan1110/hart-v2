import argparse
import json
import os
import time
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from hart.utils import encode_prompts, llm_system_prompt
from sample import save_images

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPT_MAP_PATH = os.path.join(REPO_ROOT, "hart-v2", "prompt_centroid_map.json")
FHAT_CENTROIDS_PATH = os.path.join(REPO_ROOT, "hart-v2", "fhat_centroids.pt")


def load_prompts(args):
    with open(args.prompt_centroid_map, "r") as f:
        prompt_cluster_map = json.load(f)
    available_prompts = list(prompt_cluster_map.keys())

    if args.prompt:
        if args.prompt not in prompt_cluster_map:
            raise KeyError(f"Prompt not found in centroid map: {args.prompt}")
        return [args.prompt], prompt_cluster_map

    if not available_prompts:
        raise ValueError(f"No prompts found in centroid map file {args.prompt_centroid_map}")

    return available_prompts, prompt_cluster_map


def main(args):
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(args.model_path).to(device)
    model.eval()

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    prompts, prompt_cluster_map = load_prompts(args)
    fhat_centroids = torch.load(args.fhat_centroids_path, map_location="cpu")

    infer_func = model.autoregressive_infer_cfg

    generated_images = []
    used_prompts = []
    print("generating  images ...")
    start_time = time.time()
    with torch.inference_mode(), torch.autocast(
        "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
    ):
        for start in tqdm(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[start : start + args.batch_size]

            cluster_ids = []
            for prompt in batch_prompts:
                if prompt not in prompt_cluster_map:
                    raise KeyError(f"Prompt not found in centroid map: {prompt}")
                cluster_id = prompt_cluster_map[prompt]
                if cluster_id not in fhat_centroids:
                    raise KeyError(f"Cluster id {cluster_id} not found in fhat centroids.")
                cluster_ids.append(cluster_id)

            # Concatenate f_hat tensors (each shaped [1, 32, 64, 64]) along batch dimension.
            fhat_batch = torch.cat([fhat_centroids[cid] for cid in cluster_ids], dim=0)
            shared_state_cache = {"f_hat": fhat_batch}

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
                alpha=3,
                shared_state=shared_state_cache,
            )

            generated_images.append(output_imgs)
            used_prompts.extend(batch_prompts)

    total_time = time.time() - start_time
    print(f"Generate {len(used_prompts)} images took {total_time:2f}s.")

    print("saving images...")
    all_imgs = torch.cat(generated_images, dim=0)
    save_images(all_imgs.clone(), args.sample_folder_dir, args.store_separately, used_prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using centroid f_hat cache.")
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="./../hart-0.7b-1024px/llm",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="./../Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument("--prompt", type=str, help="Single prompt to render.", default="")
    parser.add_argument("--seed", type=int, default=1)
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
        "--store_separately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )
    parser.add_argument(
        "--prompt_centroid_map",
        type=str,
        default=PROMPT_MAP_PATH,
        help="Path to prompt to centroid mapping json file.",
    )
    parser.add_argument(
        "--fhat_centroids_path",
        type=str,
        default=FHAT_CENTROIDS_PATH,
        help="Path to fhat centroids tensor checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of prompts to process per batch.",
    )
    args = parser.parse_args()

    main(args)
