import argparse
import copy
import datetime
import os
import random
import time
from typing import Any, Dict, List, Optional  # helper aliases for readability

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from hart.modules.models.transformer import HARTForT2I
from hart.modules.networks.utils import sample_with_top_k_top_p_
from hart.utils import (
    artificial_prompts,
    default_prompts,
    encode_prompts,
    llm_system_prompt,
    safety_check,
)
#setting pca to true by default, used this in hdbscan and K-means to cluster.
PCA=True
TOP_N_COMPONENTS=100

def save_images(
    sample_imgs,
    sample_folder_dir,
    store_separately,
    prompts,
    cluster_assignments=None,
    save_individual=False,
):
    os.makedirs(sample_folder_dir, exist_ok=True)
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_list = [int(c.item()) for c in cluster_assignments.detach().cpu()]
    elif cluster_assignments is not None:
        cluster_list = [int(c) for c in cluster_assignments]
    else:
        cluster_list = ["none"] * len(prompts)

    sample_imgs = sample_imgs.clamp(0.0, 1.0)
    sample_imgs_uint8 = (
        sample_imgs.mul(255).add_(0.5).clamp(0, 255).to(torch.uint8).cpu()
    )
    num_imgs = sample_imgs_uint8.shape[0]
    pil_images = []
    for img_idx in range(num_imgs):
        img_np = sample_imgs_uint8[img_idx].permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)
        pil_images.append(pil_img)
        cluster_id = cluster_list[img_idx] if img_idx < len(cluster_list) else "none"
        if save_individual:
            filename = os.path.join(
                sample_folder_dir,
                f"{img_idx:06d}_prompt{img_idx}_cluster{cluster_id}.png",
            )
            pil_img.save(filename)
            print(f"Image {img_idx} saved as {filename}.")

    prompt_lines = []
    for idx, prompt in enumerate(prompts):
        cluster_id = cluster_list[idx] if idx < len(cluster_list) else "none"
        prompt_lines.append(f"{idx}\tcluster:{cluster_id}\t{prompt}")
    with open(os.path.join(sample_folder_dir, "prompts.txt"), "w") as f:
        f.write("\n".join(prompt_lines))

    if not pil_images:
        return

    font = ImageFont.load_default()
    grid_cols, grid_rows = 4, 4
    grid_capacity = grid_cols * grid_rows
    tile_w, tile_h = pil_images[0].size
    for grid_idx, start in enumerate(range(0, num_imgs, grid_capacity)):
        indices = list(range(start, min(start + grid_capacity, num_imgs)))
        grid_image = Image.new("RGB", (grid_cols * tile_w, grid_rows * tile_h))
        draw = ImageDraw.Draw(grid_image)
        for pos, img_idx in enumerate(indices):
            row, col = divmod(pos, grid_cols)
            x_offset = col * tile_w
            y_offset = row * tile_h
            grid_image.paste(pil_images[img_idx], (x_offset, y_offset))
            cluster_id = cluster_list[img_idx] if img_idx < len(cluster_list) else "none"
            label_text = f"idx {img_idx} | cluster {cluster_id}"
            if hasattr(draw, "textbbox"):
                left, top, right, bottom = draw.textbbox(
                    (0, 0), label_text, font=font
                )
                text_w, text_h = right - left, bottom - top
            else:
                text_w, text_h = font.getsize(label_text)
            text_x = x_offset + 4
            text_y = y_offset + tile_h - text_h - 6
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2],
                fill=(0, 0, 0),
            )
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
        grid_path = os.path.join(
            sample_folder_dir, f"sample_grid_{grid_idx:02d}.png"
        )
        grid_image.save(grid_path)
        print(f"Grid image saved to {grid_path}")


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
    working_embeddings = embeddings.detach()
    if PCA:
        working_embeddings = working_embeddings.float()
        num_components = min(TOP_N_COMPONENTS, *working_embeddings.shape)
        if num_components > 0:
            _, _, v = torch.pca_lowrank(working_embeddings, q=num_components)
            working_embeddings = working_embeddings @ v[:, :num_components]
    centroids = working_embeddings[torch.randperm(num_points)[:num_clusters]].clone()
    assignments = torch.zeros(num_points, dtype=torch.long)
    for _ in range(max(num_iters, 1)):
        distances = torch.cdist(working_embeddings, centroids)
        new_assignments = distances.argmin(dim=1)
        if torch.equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for idx in range(num_clusters):
            mask = assignments == idx
            if mask.any():
                centroids[idx] = working_embeddings[mask].mean(dim=0)
            else:
                replacement_idx = torch.randint(0, num_points, ()).item()
                centroids[idx] = working_embeddings[replacement_idx]
    else:
        distances = torch.cdist(working_embeddings, centroids)
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

    # embeddings_np = embeddings.detach().cpu().numpy()
    embedding_tensor = embeddings.detach()  # stays torch.Tensor
    if PCA:
        embedding_tensor = embedding_tensor.float()
        num_components = min(TOP_N_COMPONENTS, *embedding_tensor.shape)
        if num_components > 0:
            _, _, v = torch.pca_lowrank(embedding_tensor, q=num_components)
            embedding_tensor = embedding_tensor @ v[:, :num_components]

    embeddings_np = embedding_tensor.cpu().numpy()


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
        raise ValueError("Batch size must be positive.")  # avoid invalid slicing loops
    for start in range(0, length, batch_size):  # iterate window starting offsets
        yield start, min(start + batch_size, length)  # clamp end index to total length


def _generate_cluster_stage_patches(
    model: HARTForT2I,
    label_B: torch.Tensor,
    context_position_ids: torch.Tensor,
    context_mask: torch.Tensor,
    stage_N: int,
    cfg: float,
    g_seed: Optional[int] = None,
) -> List[torch.Tensor]:
    if stage_N < 0 or stage_N >= len(model.patch_nums) - 1:
        raise ValueError(
            f"stage_N must be between 0 and {len(model.patch_nums) - 2}, got {stage_N}"
        )  # validate stage bounds against coarse hierarchy

    device = label_B.device  # remember device of centroid embeddings
    B = label_B.shape[0]  # number of clusters to process
    if B == 0:
        raise ValueError("label_B must contain at least one centroid.")  # cannot run without inputs

    if g_seed is None:
        rng = None  # disable deterministic sampling when no seed is set
    else:
        model.rng.manual_seed(g_seed)  # configure shared generator for reproducibility
        rng = model.rng  # reuse generator in sampling helper

    zero_pad = torch.full_like(label_B, fill_value=0.0)  # unconditional branch for CFG
    cond_input = torch.cat((label_B, zero_pad), dim=0)  # double batch for conditional/unconditional streams
    cond_BD = model.context_embed(model.context_norm(cond_input))  # embed text context into model hidden space

    context_position_ids = torch.cat(
        (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
        dim=0,
    )  # extend position ids for unconditional tokens
    b = context_mask.shape[0]  # original batch size before CFG duplication
    context_mask = torch.cat(
        (context_mask, torch.full_like(context_mask, fill_value=0)), dim=0
    )  # replicate mask structure
    context_mask[b:, 0] = 1  # keep BOS token unmasked in unconditional branch

    if model.pos_1LC is not None:
        lvl_pos = model.lvl_embed(model.lvl_1L) + model.pos_1LC  # fetch hierarchical positional encoding
    else:
        lvl_pos = model.lvl_embed(model.lvl_1L)  # fallback when no offset tensor

    if model.pos_start is not None:
        next_token_map = (
            cond_BD
            + model.pos_start.expand_as(cond_BD)
            + lvl_pos[:, : model.first_l]
        )  # incorporate learned start tokens plus positional term
    else:
        next_token_map = cond_BD + lvl_pos[:, : model.first_l]  # simple sum of context and positions

    target_hw = model.patch_nums[-1]  # side length of full-resolution latent grid
    f_hat = cond_BD.new_zeros(B, model.Cvae, target_hw, target_hw)  # running accumulator of decoded features
    cur_L = 0  # track how many positional tokens have been consumed
    cond_BD_or_gss = model.shared_ada_lin(cond_BD)  # pre-compute shared AdaLN conditioning

    patches: List[torch.Tensor] = []  # buffer for per-stage accumulated feature maps
    for block in model.blocks:  # iterate transformer blocks once to toggle cache
        block.attn.kv_caching(True)  # enable KV caching for faster auto-reg recurrence

    try:
        for si, pn in enumerate(model.patch_nums[:-1]):  # iterate autoregressive stages (exclude maskgit)
            ratio = (
                si / model.num_stages_minus_1 if model.num_stages_minus_1 > 0 else 0.0
            )  # normalised stage progress for CFG scaling
            if si > 0:
                cur_L += pn * pn  # advance positional offset by number of tokens at previous stage
            else:
                cur_L += model.context_token  # first stage consumes text tokens

            x = next_token_map  # transformer input tokens
            for block in model.blocks:
                x = block(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )  # run standard HART block conditioned on centroids

            logits_BlV = model.get_logits(x, cond_BD)  # project hidden states to VAE codebook logits
            t = cfg * ratio  # adjust CFG strength by stage
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]  # apply CFG mixing
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]  # only decode the first SOS token at stage 0

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=0.0,
                num_samples=1,
            )[:, :, 0]  # perform top-k sampling tuned for HART

            h_BChw = model.vae_quant_proxy[0].embedding(idx_Bl)  # convert token ids to embedding vectors
            h_BChw = h_BChw.transpose(1, 2).reshape(B, model.Cvae, pn, pn)  # reshape into spatial feature map

            f_hat, next_token_map = model.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(
                si, len(model.patch_nums), f_hat, h_BChw, patch_nums=model.patch_nums
            )  # update accumulated feature grid and next token map
            patches.append(f_hat.detach().clone().cpu())  # store CPU copy of accumulated features

            if si >= stage_N:
                break  # exit once we reach requested stage

            next_token_map = next_token_map.view(B, model.Cvae, -1).transpose(1, 2)  # flatten spatial map back to tokens
            lvl_slice = lvl_pos[:, cur_L : cur_L + model.patch_nums[si + 1] ** 2]  # positional slice for next stage tokens
            next_token_map = model.word_embed(next_token_map) + lvl_slice  # embed tokens and add positional offsets
            next_token_map = next_token_map.repeat(2, 1, 1)  # duplicate for conditional/unconditional CFG streams
    finally:
        for block in model.blocks:  # reset block state even on early exit
            block.attn.kv_caching(False)  # always disable caching afterwards

    return patches  # accumulated feature maps per stage up to N


def generate_cluster_centroid_patches(
    prompts: List[str],
    model: HARTForT2I,
    text_model,
    text_tokenizer,
    *,
    cluster_stage_N: int,
    output_path: str,
    device: torch.device,
    cfg: float,
    cluster_method: str = "kmeans",
    num_clusters: Optional[int] = None,
    cluster_truncate_tokens: Optional[int] = None,
    kmeans_iters: int = 20,
    max_token_length: int = 300,
    use_llm_system_prompt: bool = False,
    seed: Optional[int] = None,
    context_tensor: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    context_position_ids: Optional[torch.Tensor] = None,
    cluster_assignments: Optional[torch.Tensor] = None,
    cluster_context_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if not prompts:
        raise ValueError("prompts must contain at least one entry.")  # require data to cluster

    model.to(device)  # move HART to desired device
    model.eval()  # disable training-time layers

    with torch.inference_mode():  # no gradients needed anywhere below
        if (
            context_tensor is None
            or context_mask is None
            or context_position_ids is None
        ):  # optionally materialise fresh context tensors
            if text_model is None or text_tokenizer is None:
                raise ValueError(
                    "text_model and text_tokenizer must be provided when context tensors are not supplied."
                )  # need encoder to materialise context tensors
            (
                _,
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
            )  # encode every prompt

        context_tensor_cpu = context_tensor.detach().to("cpu", copy=True)  # freeze CPU copies for clustering
        context_mask_cpu = context_mask.detach().to("cpu", copy=True)  # same for masks
        context_position_ids_cpu = context_position_ids.detach().to("cpu", copy=True)  # same for position ids

        if cluster_assignments is None:
            if cluster_method == "kmeans":
                if num_clusters is None or num_clusters <= 0:
                    raise ValueError("num_clusters must be positive for kmeans.")  # enforce valid hyperparams
                if len(prompts) < num_clusters:
                    raise ValueError(
                        "num_clusters cannot exceed the number of prompts."
                    )  # without repeats clustering fails
                pooled_embeddings = _pool_prompt_embeddings(
                    context_tensor_cpu,
                    context_mask_cpu,
                    truncate_tokens=cluster_truncate_tokens,
                ).float()  # flatten prompt embeddings for clustering
                assignments, _ = _run_kmeans(
                    pooled_embeddings.detach(),
                    min(num_clusters, len(prompts)),
                    kmeans_iters,
                )  # perform lightweight k-means
                cluster_assignments_cpu = assignments  # capture assignment vector
            elif cluster_method == "hdbscan":
                pooled_embeddings = _pool_prompt_embeddings(
                    context_tensor_cpu,
                    context_mask_cpu,
                    truncate_tokens=cluster_truncate_tokens,
                ).float()  # produce embedding matrix for density clustering
                assignments, _ = _run_hdbscan(
                    pooled_embeddings.detach(),
                    min_cluster_size=max(2, num_clusters or 2),
                    min_samples=None,
                )  # fallback to HDBSCAN heuristics
                cluster_assignments_cpu = assignments  # use HDBSCAN labels
            else:
                cluster_assignments_cpu = torch.arange(
                    len(prompts), dtype=torch.long
                )  # treat each prompt as its own cluster
        else:
            cluster_assignments_cpu = cluster_assignments.detach().to("cpu").long()  # reuse provided assignments

        if cluster_assignments_cpu.numel() != len(prompts):
            raise ValueError(
                "cluster_assignments length must match the number of prompts."
            )  # guard mismatched mapping

        cluster_count = int(cluster_assignments_cpu.max().item()) + 1  # determine cluster cardinality
        cluster_members: List[List[int]] = [[] for _ in range(cluster_count)]  # allocate member lists
        for idx, cluster_id in enumerate(cluster_assignments_cpu.tolist()):
            cluster_members[cluster_id].append(idx)  # assign prompt index to cluster bucket

        if cluster_context_tensor is None:
            cluster_contexts = []  # build centroid embeddings from members
            for members in cluster_members:
                if members:
                    member_tensor = context_tensor_cpu[members].mean(dim=0)  # average contextual embeddings per cluster
                else:
                    member_tensor = context_tensor_cpu.mean(dim=0)  # fallback to global mean when cluster empty
                cluster_contexts.append(member_tensor)  # collect centroid embedding
            cluster_context_tensor_cpu = torch.stack(cluster_contexts, dim=0)  # stack into tensor (clusters, T, D)
        else:
            cluster_context_tensor_cpu = cluster_context_tensor.detach().to("cpu")  # use precomputed centroids

        representative_indices = [
            members[0] if members else 0 for members in cluster_members
        ]  # choose a representative prompt index per cluster
        cluster_context_mask_cpu = torch.stack(
            [context_mask_cpu[idx] for idx in representative_indices], dim=0
        )  # gather mask tensors for each representative prompt
        cluster_position_ids_cpu = torch.stack(
            [context_position_ids_cpu[idx] for idx in representative_indices], dim=0
        )  # gather position id tensors likewise

        target_dtype = model.word_embed.weight.dtype  # align dtype with transformer embeddings
        cluster_context_tensor_device = cluster_context_tensor_cpu.to(
            device=device, dtype=target_dtype
        )  # move centroid contexts to device/dtype
        cluster_context_mask_device = cluster_context_mask_cpu.to(device=device)  # move masks to device
        cluster_position_ids_device = cluster_position_ids_cpu.to(device=device)  # move position ids to device

        patches = _generate_cluster_stage_patches(
            model=model,
            label_B=cluster_context_tensor_device,
            context_position_ids=cluster_position_ids_device,
            context_mask=cluster_context_mask_device,
            stage_N=cluster_stage_N,
            cfg=cfg,
            g_seed=seed,
        )  # compute accumulated feature maps up to requested stage

        output_dir = os.path.dirname(output_path)  # locate parent directory of save path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)  # ensure directory exists before saving

        shared_patch_count = len(patches)  # note how many stages were captured
        metadata: Dict[str, Any] = {
            "created_at": datetime.datetime.utcnow().isoformat(),  # timestamp for audit trail
            "cluster_stage_N": cluster_stage_N,  # highest pre-generated stage index
            "shared_patch_count": shared_patch_count,  # number of stored stages
            "cluster_method": cluster_method,  # clustering strategy employed
            "num_clusters": cluster_count,  # resulting cluster count
            "num_prompts": len(prompts),  # prompts included in clustering set
            "max_token_length": max_token_length,  # tokenizer length constraint
            "use_llm_system_prompt": use_llm_system_prompt,  # whether system prompt was prepended
            "cluster_truncate_tokens": cluster_truncate_tokens,  # optional token truncation depth
            "kmeans_iters": kmeans_iters if cluster_method == "kmeans" else None,  # k-means iteration budget (if used)
            "cfg": cfg,  # classifier-free guidance scale
        }  # persist reproduction-critical metadata

        payload: Dict[str, Any] = {
            "metadata": metadata,  # configuration snapshot for record keeping
            "prompts": prompts,  # raw prompts included in clustering
            "cluster_assignments": cluster_assignments_cpu.tolist(),  # prompt -> cluster ids
            "cluster_members": cluster_members,  # inverse mapping cluster -> prompt indices
            "cluster_representative_indices": representative_indices,  # representative prompt per cluster
            "cluster_context_tensor": cluster_context_tensor_cpu,  # centroid text embeddings on CPU
            "cluster_context_mask": cluster_context_mask_cpu,  # matching attention masks
            "cluster_context_position_ids": cluster_position_ids_cpu,  # matching positional ids
            "cluster_centroid_patches": patches,  # accumulated VAE feature maps per stage
        }  # bundle all tensors and metadata for saving

        torch.save(payload, output_path)  # write dataset to disk

    return metadata  # surface metadata for logging or diagnostics


def load_cluster_centroid_patches(
    path: str, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cluster centroid file not found: {path}")  # fail fast on missing files

    data: Dict[str, Any] = torch.load(path, map_location="cpu")  # load payload without allocating GPU memory

    if device is not None:
        for key in (
            "cluster_context_tensor",
            "cluster_context_mask",
            "cluster_context_position_ids",
        ):  # iterate over tensor fields that benefit from device move
            tensor_value = data.get(key)  # fetch tensor backing each key
            if isinstance(tensor_value, torch.Tensor):
                data[key] = tensor_value.to(device)  # move tensor to requested device
        patches_value = data.get("cluster_centroid_patches")  # list of per-stage tensors
        if isinstance(patches_value, list):
            data["cluster_centroid_patches"] = [
                patch.to(device) if isinstance(patch, torch.Tensor) else patch
                for patch in patches_value
            ]  # move each stored patch independently

    return data  # caller receives tensor bundle ready for inference

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
                    assignments, _ = _run_kmeans(
                        pooled_embeddings.detach(),
                        min(args.num_clusters, len(prompts)),
                        args.kmeans_iters,
                    )
                    cluster_assignments = assignments
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

            if cluster_assignments is not None:
                cluster_assignments = cluster_assignments.to(dtype=torch.long)
                cluster_count = int(cluster_assignments.max().item()) + 1
                cluster_contexts = []
                for cluster_idx in range(cluster_count):
                    mask = cluster_assignments == cluster_idx
                    if mask.any():
                        cluster_contexts.append(context_tensor_all[mask].mean(dim=0))
                    else:
                        cluster_contexts.append(context_tensor_all.mean(dim=0))
                cluster_context_tensor = torch.stack(cluster_contexts, dim=0)

                assignment_list = cluster_assignments.tolist()
                for cluster_idx in range(cluster_count):
                    member_prompts = [
                        prompts[p_idx]
                        for p_idx, cluster_id in enumerate(assignment_list)
                        if cluster_id == cluster_idx
                    ]
                    print(f"Cluster {cluster_idx}: {len(member_prompts)} prompts")

            if args.save_cluster_centroid_patches:
                if args.cluster_centroid_stage is None:
                    raise ValueError(
                        "--cluster_centroid_stage must be provided when "
                        "--save_cluster_centroid_patches is set."
                    )  # user must declare how many stages to materialise
                if cluster_assignments is None and args.cluster_method != "none":
                    raise ValueError(
                        "Cluster centroid generation requires valid cluster assignments. "
                        "Adjust --num_clusters or choose --cluster_method none to proceed."
                    )  # refuse to generate patches without cluster labels
                centroid_output_path = (
                    args.cluster_centroid_path
                    if args.cluster_centroid_path
                    else os.path.join(
                        args.sample_folder_dir,
                        f"cluster_centroids_stage_{args.cluster_centroid_stage}.pt",
                    )
                )  # resolve save location (defaults to samples directory)
                active_model: HARTForT2I = (
                    ema_model if args.use_ema else model
                )  # use EMA weights when enabled
                centroid_metadata = generate_cluster_centroid_patches(
                    prompts=prompts,  # prompts from the current batch
                    model=active_model,  # sampler (EMA or base) used for centroid rollouts
                    text_model=text_model,  # text encoder for context regeneration when needed
                    text_tokenizer=text_tokenizer,  # tokenizer paired with text encoder
                    cluster_stage_N=args.cluster_centroid_stage,  # highest stage to pre-generate
                    output_path=centroid_output_path,  # location for serialized patches
                    device=device,  # device where generation executes
                    cfg=args.cfg,  # reuse CFG scale from sampling
                    cluster_method=args.cluster_method,  # keep clustering strategy consistent
                    num_clusters=args.num_clusters,  # total clusters requested on CLI
                    cluster_truncate_tokens=args.cluster_truncate_tokens,  # optional truncation of embeddings
                    kmeans_iters=args.kmeans_iters,  # iteration budget for k-means
                    max_token_length=args.max_token_length,  # tokenizer sequence length limit
                    use_llm_system_prompt=args.use_llm_system_prompt,  # reuse system prompt configuration
                    seed=args.seed,  # propagate RNG seed for reproducibility
                    context_tensor=context_tensor_all,  # reuse already encoded prompt embeddings
                    context_mask=context_mask_all,  # reuse prompt attention masks
                    context_position_ids=context_position_ids_all,  # reuse positional ids per prompt
                    cluster_assignments=cluster_assignments,  # pass cluster mapping from earlier step
                    cluster_context_tensor=cluster_context_tensor,  # supply centroid embeddings if available
                )  # run centroid patch generation pipeline
                print(
                    f"Saved {centroid_metadata['shared_patch_count']} centroid patches per cluster "
                    f"to {centroid_output_path}"
                )  # inform user where patches were stored

            infer_func = (
                ema_model.autoregressive_infer_cfg
                if args.use_ema
                else model.autoregressive_infer_cfg
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
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids_chunk,
                    context_mask=context_mask_chunk,
                    save_autoregressive_steps=args.save_autoregressive_steps,
                    sample_folder_dir=args.sample_folder_dir,
                    store_seperately=args.store_seperately,
                    prompts=prompts[start:end],
                    prompt_offset=start,
                )
                inference_time += time.time() - chunk_start
                outputs.append(output_chunk.detach().cpu())
            output_imgs = torch.cat(outputs, dim=0)
            

    total_time = time.time() - start_time
    print(
        f"Generate {len(prompts)} images in {total_time:2f}s "
        f"(batched inference {inference_time:2f}s)."
    )

    save_images(
        output_imgs.clone(),
        args.sample_folder_dir,
        args.store_seperately,
        prompts,
        cluster_assignments=cluster_assignments,
        save_individual=args.save_individual_images,
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
        default=5,
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
        "--save_individual_images",
        action="store_true",
        help="If set, save each generated image with prompt and cluster metadata in the filename.",
    )
    parser.add_argument(
        "--save_autoregressive_steps",
        help="Enable saving intermediate autoregressive stage images.",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--save_cluster_centroid_patches",
        help="Generate and persist cluster centroid patches before sampling.",
        action="store_true",
    )  # toggle pre-computation of centroid warm-start patches
    parser.add_argument(
        "--cluster_centroid_stage",
        type=int,
        default=None,
        help="0-indexed stage up to which centroid patches are generated.",
    )  # specify which stage's accumulated features to cache
    parser.add_argument(
        "--cluster_centroid_path",
        type=str,
        default=None,
        help=(
            "Destination path for centroid patches. "
            "Defaults to <sample_folder_dir>/cluster_centroids_stage_<N>.pt."
        ),
    )  # optional override for centroid save location
    args = parser.parse_args()

    main(args)
