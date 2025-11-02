import argparse
import copy
import datetime
import os
import random
import time
import math

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
from hart.utils import (
    default_prompts,
    encode_prompts,
    llm_system_prompt,
    safety_check,
    artificial_prompts
)


def save_images(sample_imgs, sample_folder_dir, store_separately, prompts):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images.png"))
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
            cur_img_store.save(os.path.join(sample_folder_dir, f"{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))

def save_final_grids(sample_imgs: torch.Tensor, sample_folder_dir: str, grid_size: int = 4) -> None:
    if sample_imgs.ndim != 4:
        raise ValueError("sample_imgs must be a 4D tensor (N, C, H, W).")
    num_imgs = sample_imgs.shape[0]
    if num_imgs == 0:
        return
    grid_cells = grid_size * grid_size
    grid_dir = os.path.join(sample_folder_dir, "final_grids")
    os.makedirs(grid_dir, exist_ok=True)
    for grid_idx in range(math.ceil(num_imgs / grid_cells)):
        start = grid_idx * grid_cells
        end = min(start + grid_cells, num_imgs)
        chunk = sample_imgs[start:end]
        if chunk.shape[0] < grid_cells:
            pad = chunk.new_zeros((grid_cells - chunk.shape[0], *chunk.shape[1:]))
            chunk = torch.cat([chunk, pad], dim=0)
        grid = torchvision.utils.make_grid(chunk, nrow=grid_size)
        grid_np = (
            grid.mul(255.0)
            .clamp_(0.0, 255.0)
            .permute(1, 2, 0)
            .to(torch.uint8)
            .numpy()
        )
        Image.fromarray(grid_np).save(
            os.path.join(grid_dir, f"{grid_idx:03d}_final_grid.png")
        )


def _pool_prompt_embeddings(context_tensor, context_mask, truncate_tokens=None):
    del context_mask  # context mask is not needed for truncated embedding clustering
    if truncate_tokens is not None:
        if truncate_tokens <= 0:
            raise ValueError("truncate_tokens must be positive when provided.")
        if context_tensor.size(1) < truncate_tokens:
            raise ValueError(
                f"context_tensor must have at least {truncate_tokens} tokens to truncate."
            )
        context_tensor = context_tensor[:, :truncate_tokens, :]
    return context_tensor.reshape(context_tensor.size(0), -1)


def _run_kmeans(embeddings, num_clusters, num_iters=20):
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")
    num_points = embeddings.shape[0] #batch_size
    if num_clusters > num_points:
        raise ValueError("num_clusters cannot exceed the number of embeddings.")
    centroids = embeddings[torch.randperm(num_points)[:num_clusters]].clone()
    assignments = torch.zeros(num_points, dtype=torch.long)
    for _ in range(max(num_iters, 1)):
        distances = torch.cdist(embeddings, centroids)
        new_assignments = distances.argmin(dim=1)
        if torch.equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for idx in range(num_clusters):
            mask = assignments == idx
            if mask.any():
                centroids[idx] = embeddings[mask].mean(dim=0)
            else:
                replacement_idx = torch.randint(0, num_points, ()).item()
                centroids[idx] = embeddings[replacement_idx]
    else:
        distances = torch.cdist(embeddings, centroids)
        assignments = distances.argmin(dim=1)
    return assignments, centroids


def _run_hdbscan(embeddings, min_cluster_size=5, min_samples=None):
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN clustering requires the `hdbscan` package. "
            "Install it with `pip install hdbscan`."
        ) from exc

    embeddings_np = embeddings.detach().cpu().numpy()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    clusterer.fit(embeddings_np)
    labels = clusterer.labels_
    if labels.size == 0:
        raise ValueError("HDBSCAN did not return any cluster labels.")

    assignments = torch.empty(len(labels), dtype=torch.long)
    label_mapping = {}
    next_cluster_id = 0
    for idx, label in enumerate(labels):
        if label >= 0:
            if label not in label_mapping:
                label_mapping[label] = next_cluster_id
                next_cluster_id += 1
            assignments[idx] = label_mapping[label]
        else:
            assignments[idx] = next_cluster_id
            next_cluster_id += 1

    if next_cluster_id == 0:
        raise ValueError("HDBSCAN did not identify any clusters.")

    return assignments, next_cluster_id



def _batched_indices(length: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)

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

    prompts: list[str] = []
    
    if args.use_artificial_prompts:
        prompts = artificial_prompts
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

    inference_time = 0.0
    start_time = time.time()
    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):

            (
                _, #context_tokens
                context_mask, # marking every osition that isn't the tokenizer's pad token
                context_position_ids, #  produces 1, 2, 3 wherever the mask is true and holds the previous value where it’s false
                context_tensor, #output of last hidden state
            ) = encode_prompts(
                prompts,
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            context_tensor_all = context_tensor.float().cpu()
            context_mask_all = context_mask.cpu()
            context_position_ids_all = context_position_ids.cpu()

            del context_tensor, context_mask, context_position_ids
            cluster_assignments = None
            cluster_context_tensor = None
            cluster_center_mask = None
            cluster_center_position_ids = None
            cluster_center_indices: list[int] = []
            pooled_embeddings = None
            cluster_centroids = None
            cluster_method = args.cluster_method.lower()
            if cluster_method == "kmeans":
                if (
                    args.num_clusters
                    and args.num_clusters > 0
                    and len(prompts) >= args.num_clusters
                ):
                    pooled_embeddings = _pool_prompt_embeddings(
                        context_tensor_all,
                        context_mask_all,
                        truncate_tokens=args.cluster_truncate_tokens,
                    ).float()
                    assignments, centroids = _run_kmeans(
                        pooled_embeddings.detach(),
                        min(args.num_clusters, len(prompts)),
                        args.kmeans_iters,
                    )
                    cluster_assignments = assignments
                    cluster_centroids = centroids
                else:
                    print(
                        "Skipping k-means clustering: ensure num_clusters "
                        "is positive and does not exceed the number of prompts."
                    )
            elif cluster_method == "hdbscan":
                if len(prompts) > 0:
                    pooled_embeddings = _pool_prompt_embeddings(
                        context_tensor_all,
                        context_mask_all,
                        truncate_tokens=args.cluster_truncate_tokens,
                    ).float()
                    assignments, _ = _run_hdbscan(
                        pooled_embeddings.detach(),
                        min_cluster_size=args.hdbscan_min_cluster_size,
                        min_samples=args.hdbscan_min_samples,
                    )
                    cluster_assignments = assignments
                else:
                    print("Skipping HDBSCAN clustering: no prompts available.")

            cluster_assignment_list: list[int] = [-1] * len(prompts)
            if cluster_assignments is not None:
                cluster_assignments = cluster_assignments.to(dtype=torch.long)
                cluster_assignment_list = cluster_assignments.tolist()
                cluster_count = int(cluster_assignments.max().item()) + 1
                for cluster_idx in range(cluster_count):
                    mask = cluster_assignments == cluster_idx
                    if mask.any():
                        member_indices = torch.nonzero(mask, as_tuple=False).view(-1)
                        if pooled_embeddings is not None:
                            member_embeddings = pooled_embeddings[member_indices]
                            if cluster_centroids is not None:
                                reference = cluster_centroids[cluster_idx]
                            else:
                                reference = member_embeddings.mean(dim=0)
                            distances = torch.norm(
                                member_embeddings - reference.unsqueeze(0), dim=1
                            )
                            best_idx = member_indices[distances.argmin()].item()
                        else:
                            best_idx = member_indices[0].item()
                        cluster_center_indices.append(best_idx)
                    else:
                        cluster_center_indices.append(0)

                if cluster_center_indices:
                    center_idx_tensor = torch.tensor(
                        cluster_center_indices, dtype=torch.long
                    )
                    cluster_context_tensor = context_tensor_all[center_idx_tensor]
                    cluster_center_mask = context_mask_all[center_idx_tensor]
                    cluster_center_position_ids = context_position_ids_all[
                        center_idx_tensor
                    ]

                assignment_list = cluster_assignment_list
                for cluster_idx in range(cluster_count):
                    member_prompts = [
                        prompts[p_idx]
                        for p_idx, cluster_id in enumerate(assignment_list)
                        if cluster_id == cluster_idx
                    ]
                    print(f"Cluster {cluster_idx}: {len(member_prompts)} prompts")
                    if cluster_center_indices:
                        center_prompt = prompts[cluster_center_indices[cluster_idx]]
                        print(f"  Center prompt: {center_prompt}")

            save_intermediate_flag = (
                args.save_autoregressive_steps or args.save_intermediate_grids
            )

            if save_intermediate_flag and prompts:
                intermediate_dir = os.path.join(
                    args.sample_folder_dir, "autoregressive_steps"
                )
                os.makedirs(intermediate_dir, exist_ok=True)
                cluster_log_path = os.path.join(
                    intermediate_dir, "prompt_clusters.txt"
                )
                with open(cluster_log_path, "w") as f:
                    for idx, prompt in enumerate(prompts):
                        cluster_id = cluster_assignment_list[idx]
                        f.write(f"{idx:04d}\tcluster={cluster_id}\t{prompt}\n")

            infer_func = (
                ema_model.autoregressive_infer_cfg
                if args.use_ema
                else model.autoregressive_infer_cfg
            )

            infer_model = ema_model if args.use_ema else model
            cluster_warmup_cache = None
            if (
                cluster_assignments is not None
                and cluster_context_tensor is not None
                and cluster_center_mask is not None
                and cluster_center_position_ids is not None
                and args.cluster_warmup_steps > 0
            ):
                print(
                    f"Preparing cluster warmup cache for "
                    f"{len(cluster_center_indices)} cluster centers "
                    f"over {args.cluster_warmup_steps} autoregressive steps."
                )
                cluster_warmup_cache = infer_model.prepare_cluster_warmup_cache(
                    cluster_context_tensor.to(device),
                    cluster_center_position_ids.to(device),
                    cluster_center_mask.to(device),
                    args.cluster_warmup_steps,
                    cfg=args.cfg,
                    top_p=0.0,
                    more_smooth=args.more_smooth,
                    g_seed=args.seed,
                )
            outputs = []
            for start, end in _batched_indices(
                len(prompts), args.prompts_per_batch
            ):
                context_tensor_chunk = context_tensor_all[start:end].to(device)
                context_mask_chunk = context_mask_all[start:end].to(device)
                context_position_ids_chunk = context_position_ids_all[start:end].to(
                    device
                )
                if cluster_assignments is not None:
                    cluster_assignments_chunk = cluster_assignments[start:end]
                else:
                    cluster_assignments_chunk = None
                chunk_start = time.time()
                output_chunk = infer_func(
                    B=context_tensor_chunk.size(0),
                    label_B=context_tensor_chunk,
                    cluster_assignments=cluster_assignments_chunk,
                    cluster_context_tensor=cluster_context_tensor,
                    cluster_warmup_steps=args.cluster_warmup_steps,
                    cluster_warmup_cache=cluster_warmup_cache,
                    cluster_center_context_mask=cluster_center_mask,
                    cluster_center_context_position_ids=cluster_center_position_ids,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids_chunk,
                    context_mask=context_mask_chunk,
                    save_autoregressive_steps=save_intermediate_flag,
                    sample_folder_dir=args.sample_folder_dir,
                    store_seperately=args.store_seperately,
                    prompts=prompts[start:end],
                    prompt_offset=start,
                )
                inference_time += time.time() - chunk_start
                outputs.append(output_chunk.detach().cpu())
            output_imgs = torch.cat(outputs, dim=0).clamp_(0.0, 1.0)

    total_time = time.time() - start_time
    print(
        f"Generate {len(prompts)} images in {total_time:2f}s "
        f"(batched inference {inference_time:2f}s)."
    )

    if args.save_final_grids:
        save_final_grids(output_imgs.clone(), args.sample_folder_dir)

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
        default="shieldgemma-2b",
    )
    parser.add_argument("--prompt", type=str, help="A single prompt.", default="")
    parser.add_argument(
        "--use_artificial_prompts",
        type=bool,
        help="Use artificial prompts",
        default=True,
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=4,
        help="Number of clusters to form over prompt embeddings.",
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=["none", "kmeans", "hdbscan"],
        default="kmeans",
        help="Clustering method to apply to prompt embeddings.",
    )
    parser.add_argument(
        "--cluster_warmup_steps",
        type=int,
        default=2,
        help="Number of low-resolution stages to guide via cluster centers.",
    )
    parser.add_argument(
        "--kmeans_iters",
        type=int,
        default=20,
        help="Number of k-means refinement iterations.",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size",
        type=int,
        default=5,
        help="Minimum cluster size when using HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan_min_samples",
        type=int,
        default=None,
        help="Minimum samples parameter for HDBSCAN (defaults to min_cluster_size).",
    )
    parser.add_argument(
        "--cluster_truncate_tokens",
        type=int,
        default=None,
        help="Truncate context embeddings to this many tokens before clustering.",
    )
    parser.add_argument(
        "--prompts_per_batch",
        type=int,
        default=8,
        help="Number of prompts to process per model forward pass.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=False)
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
        "--save_autoregressive_steps",
        help="Enable saving intermediate autoregressive stage images.",
        action="store_true",
    )
    parser.add_argument(
        "--save_intermediate_grids",
        help="Save 4x4 grids of intermediate autoregressive steps for each prompt.",
        action="store_true",
    )
    parser.add_argument(
        "--save_final_grids",
        help="Save final images in batched 4x4 grids with empty padding.",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
