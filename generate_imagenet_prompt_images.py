import argparse
import copy
import os
import re
import time
from typing import Dict, List, Optional, Tuple

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


def save_image_single(
    img_tensor: torch.Tensor,
    prompt: str,
    output_dir: str,
    resize_to: int,
    prompt_idx: int,
    sample_idx: int,
) -> str:
    """Save a single image with prompt_id in the filename."""
    # img_tensor shape: (3, H, W)
    img_np = img_tensor.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    os.makedirs(output_dir, exist_ok=True)
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC

    pil_img = Image.fromarray(np.transpose(img_np, (1, 2, 0)))
    if resize_to:
        pil_img = pil_img.resize((resize_to, resize_to), resample=resample)
    # Filename includes prompt_id for easy mapping to cluster file
    fname = f"prompt_{prompt_idx:04d}_sample_{sample_idx:02d}_{_sanitize_filename(prompt)[:50]}.png"
    pil_img.save(os.path.join(output_dir, fname))
    return fname


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


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def get_prompt_embeddings(
    prompts: List[str],
    text_model,
    text_tokenizer,
    max_token_length: int,
    use_llm_system_prompt: bool,
    device: torch.device,
) -> np.ndarray:
    """
    Encode all prompts and return embeddings for clustering.

    Returns:
        embeddings: numpy array of shape (num_prompts, embedding_dim)
                   We use mean pooling over the context_tensor for clustering.
    """
    all_embeddings = []

    # Process in batches to avoid OOM
    batch_size = 32
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]

        with torch.inference_mode():
            (
                _context_tokens,
                _context_mask,
                _context_position_ids,
                context_tensor,
            ) = encode_prompts(
                batch_prompts,
                text_model,
                text_tokenizer,
                max_token_length,
                llm_system_prompt,
                use_llm_system_prompt,
            )
            # Mean pool over sequence dimension to get a single vector per prompt
            # context_tensor shape: (batch, seq_len, hidden_dim)
            embeddings = context_tensor.mean(dim=1).cpu().numpy()  # (batch, hidden_dim)
            all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def cluster_prompts_kmeans(
    embeddings: np.ndarray,
    num_clusters: int,
) -> Tuple[np.ndarray, List[int]]:
    """
    Cluster prompt embeddings using KMeans.

    Args:
        embeddings: numpy array of shape (num_prompts, embedding_dim)
        num_clusters: number of clusters to create

    Returns:
        cluster_labels: array of cluster assignments for each prompt
        centroid_ids: list of prompt ids that are closest to each cluster center
    """
    from sklearn.cluster import KMeans

    print(f"[Clustering] Running KMeans with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_  # (num_clusters, embedding_dim)

    # For each cluster, find the prompt closest to the cluster center (centroid prompt)
    centroid_ids = []
    for cluster_idx in range(num_clusters):
        # Get indices of prompts in this cluster
        cluster_mask = cluster_labels == cluster_idx
        cluster_prompt_indices = np.where(cluster_mask)[0]

        if len(cluster_prompt_indices) == 0:
            # Empty cluster (shouldn't happen with KMeans, but just in case)
            centroid_ids.append(-1)
            continue

        # Find the prompt closest to the cluster center
        cluster_embeddings = embeddings[cluster_mask]
        center = cluster_centers[cluster_idx]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_in_cluster = np.argmin(distances)
        centroid_prompt_id = cluster_prompt_indices[closest_in_cluster]
        centroid_ids.append(int(centroid_prompt_id))

    print(f"[Clustering] KMeans complete. Found {num_clusters} clusters.")
    return cluster_labels, centroid_ids


def cluster_prompts_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> Tuple[np.ndarray, List[int]]:
    """
    Cluster prompt embeddings using HDBSCAN.

    HDBSCAN automatically determines the number of clusters.
    Noise points are assigned label -1.

    Args:
        embeddings: numpy array of shape (num_prompts, embedding_dim)
        min_cluster_size: minimum size of clusters
        min_samples: minimum samples in a neighborhood for core points

    Returns:
        cluster_labels: array of cluster assignments for each prompt (-1 for noise)
        centroid_ids: list of prompt ids that are closest to each cluster centroid
    """
    from hdbscan import HDBSCAN

    print(f"[Clustering] Running HDBSCAN with min_cluster_size={min_cluster_size}...")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # Get unique cluster labels (excluding noise label -1)
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)
    num_clusters = len(unique_labels)

    print(f"[Clustering] HDBSCAN found {num_clusters} clusters (plus noise points labeled -1).")

    # For each cluster, compute centroid and find closest prompt
    centroid_ids = []
    for cluster_idx in sorted(unique_labels):
        cluster_mask = cluster_labels == cluster_idx
        cluster_prompt_indices = np.where(cluster_mask)[0]

        # Compute cluster centroid as mean of cluster embeddings
        cluster_embeddings = embeddings[cluster_mask]
        center = cluster_embeddings.mean(axis=0)

        # Find prompt closest to centroid
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_in_cluster = np.argmin(distances)
        centroid_prompt_id = cluster_prompt_indices[closest_in_cluster]
        centroid_ids.append(int(centroid_prompt_id))

    # For noise points (-1), we'll assign them to their own "cluster"
    # Each noise point becomes its own centroid (no sharing)
    # This is handled in the mapping logic below

    return cluster_labels, centroid_ids


def build_prompt_to_centroid_mapping(
    cluster_labels: np.ndarray,
    centroid_ids: List[int],
) -> Dict[int, int]:
    """
    Build a mapping from each prompt_id to its centroid prompt_id.

    For HDBSCAN noise points (label=-1), each noise point maps to itself
    (i.e., no sharing of cached f_hat).

    Args:
        cluster_labels: cluster assignment for each prompt (-1 for noise in HDBSCAN)
        centroid_ids: list of centroid prompt ids for each cluster (indexed by cluster label)

    Returns:
        mapping: dict mapping prompt_id -> centroid_prompt_id
    """
    mapping = {}
    unique_labels = set(cluster_labels)

    # Build a lookup from cluster_label to centroid_id
    # centroid_ids is ordered by cluster index (0, 1, 2, ...)
    label_to_centroid = {}
    sorted_labels = sorted([l for l in unique_labels if l >= 0])
    for idx, label in enumerate(sorted_labels):
        if idx < len(centroid_ids):
            label_to_centroid[label] = centroid_ids[idx]

    for prompt_id, label in enumerate(cluster_labels):
        if label == -1:
            # Noise point (HDBSCAN): maps to itself (no shared computation)
            mapping[prompt_id] = prompt_id
        else:
            mapping[prompt_id] = label_to_centroid.get(label, prompt_id)

    return mapping


def write_cluster_mapping_file(
    mapping: Dict[int, int],
    cluster_labels: np.ndarray,
    output_path: str,
):
    """
    Write the prompt-to-centroid mapping to a .txt file.

    Format:
        prompt_id=123 -> centroid_id=7, cluster_id=2
        prompt_id=124 -> centroid_id=7, cluster_id=2
        ...
    """
    with open(output_path, "w") as f:
        f.write("# Prompt to Centroid Mapping\n")
        f.write("# Format: prompt_id -> centroid_id, cluster_id\n")
        f.write("# For HDBSCAN noise points (cluster_id=-1), prompt maps to itself.\n")
        f.write("#" + "=" * 60 + "\n\n")

        for prompt_id in sorted(mapping.keys()):
            centroid_id = mapping[prompt_id]
            cluster_id = int(cluster_labels[prompt_id])
            f.write(f"prompt_id={prompt_id} -> centroid_id={centroid_id}, cluster_id={cluster_id}\n")

    print(f"[Output] Cluster mapping written to: {output_path}")


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def run_inference_no_clustering(args):
    """Original inference without clustering (cluster_algo=3)."""
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

            if args.save_images:
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
                f"batch {start // args.batch_size + 1} processed ({len(batch_prompts)} images)."
            )

    total_time = time.time() - total_start
    print(
        f"Generated {len(prompts) * args.num_images_per_prompt} images in {total_time:.2f}s. "
        f"Outputs in {args.output_dir}"
    )

    if args.save_images and prompt_log:
        with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
            f.write("\n".join(prompt_log))


def run_inference_with_clustering(args):
    """
    Inference with clustering-based shared patch generation.

    Two-phase approach:
    - Phase 1: Generate f_hat and next_token_map for centroid prompts only
    - Phase 2: Generate full images using cached centroid data for shared patches
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device is required for HART inference.")

    set_seed(args.seed)
    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    # Load models
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

    infer_func = ema_model.autoregressive_infer_cfg if args.use_ema else model.autoregressive_infer_cfg

    # =========================================================================
    # STEP 1: Get embeddings for all prompts (used for clustering)
    # =========================================================================
    print("\n[Step 1] Computing prompt embeddings for clustering...")
    embeddings = get_prompt_embeddings(
        prompts,
        text_model,
        text_tokenizer,
        args.max_token_length,
        args.use_llm_system_prompt,
        device,
    )
    print(f"[Step 1] Embeddings shape: {embeddings.shape}")

    # =========================================================================
    # STEP 2: Cluster prompts
    # =========================================================================
    print("\n[Step 2] Clustering prompts...")
    if args.cluster_algo == 1:
        # KMeans clustering
        cluster_labels, centroid_ids = cluster_prompts_kmeans(
            embeddings,
            num_clusters=args.num_clusters,
        )
    elif args.cluster_algo == 2:
        # HDBSCAN clustering
        cluster_labels, centroid_ids = cluster_prompts_hdbscan(
            embeddings,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
        )
    else:
        raise ValueError(f"Invalid cluster_algo: {args.cluster_algo}")

    # Build prompt_id -> centroid_id mapping
    prompt_to_centroid = build_prompt_to_centroid_mapping(cluster_labels, centroid_ids)

    # Get unique centroid prompt ids (these are the prompts we'll run Phase 1 on)
    unique_centroid_ids = list(set(prompt_to_centroid.values()))
    print(f"[Step 2] Number of unique centroids: {len(unique_centroid_ids)}")

    # Write mapping file
    mapping_file_path = os.path.join(args.output_dir, "cluster_mapping.txt")
    write_cluster_mapping_file(prompt_to_centroid, cluster_labels, mapping_file_path)

    # =========================================================================
    # STEP 3 (Phase 1): Generate and cache f_hat/next_token_map for centroids
    # =========================================================================
    print("\n[Step 3 - Phase 1] Generating shared patches for centroid prompts...")

    # Cache structure: centroid_prompt_id -> {"f_hat": tensor, "next_token_map": tensor}
    # Note: We cache per sample_idx since seeds differ
    centroid_cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    # The alpha parameter determines up to which stage we cache
    # This comes from the model's shared_hart logic (typically alpha=3 means first 4 stages)
    alpha = args.alpha

    total_start = time.time()

    for sample_idx in range(args.num_images_per_prompt):
        seed = args.seed + sample_idx
        print(f"\n[Phase 1] Sample {sample_idx + 1}/{args.num_images_per_prompt}")

        for centroid_id in unique_centroid_ids:
            centroid_prompt = prompts[centroid_id]

            # Encode the centroid prompt
            (
                _context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                [centroid_prompt],
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            # Create cache directory for this centroid's f_hat
            cache_dir = os.path.join(args.output_dir, "centroid_cache", f"sample_{sample_idx:02d}")
            os.makedirs(cache_dir, exist_ok=True)
            fhat_save_path = os.path.join(cache_dir, f"centroid_{centroid_id:04d}")

            with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Run inference with save_fhat=True to cache the shared state
                # This generates the image but importantly saves f_hat at stage=alpha
                _ = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    save_fhat=True,  # Enable saving f_hat
                    save_fhat_path=fhat_save_path,
                    alpha=alpha,
                    is_shared_hart=False,  # We're generating, not using shared state
                )

            # Store cache path for later retrieval
            if centroid_id not in centroid_cache:
                centroid_cache[centroid_id] = {}
            centroid_cache[centroid_id][sample_idx] = {
                "cache_path": fhat_save_path,
            }

            print(f"  [Phase 1] Cached centroid prompt_id={centroid_id}")

    print(f"\n[Phase 1] Complete. Cached {len(unique_centroid_ids)} centroid states.")

    # =========================================================================
    # STEP 4 (Phase 2): Generate full images using cached centroid data
    # =========================================================================
    print("\n[Step 4 - Phase 2] Generating images for all prompts using cached shared patches...")

    prompt_log = []

    for sample_idx in range(args.num_images_per_prompt):
        seed = args.seed + sample_idx
        print(f"\n[Phase 2] Sample {sample_idx + 1}/{args.num_images_per_prompt}")

        for prompt_id, prompt in enumerate(prompts):
            # Look up which centroid this prompt belongs to
            centroid_id = prompt_to_centroid[prompt_id]

            # Encode this prompt
            (
                _context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                [prompt],
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            # Load cached shared state from centroid
            cache_path = centroid_cache[centroid_id][sample_idx]["cache_path"]
            fhat_file = os.path.join(cache_path, f"fhat_kv_stage_{alpha}.pt")

            if os.path.exists(fhat_file):
                # Load the cached f_hat state
                shared_state = torch.load(fhat_file, map_location=device)
            else:
                # If cache doesn't exist (shouldn't happen), generate without sharing
                print(f"  [Warning] Cache not found for centroid {centroid_id}, generating without sharing")
                shared_state = None

            with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Generate image using cached shared_state
                # When is_shared_hart=True and shared_state is provided,
                # infer_cfg will skip stages 0..alpha and use the cached f_hat
                output_img = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    save_fhat=False,
                    alpha=alpha,
                    is_shared_hart=(shared_state is not None),
                    shared_state=shared_state,
                )

            # Save the generated image with prompt_id in filename (if save_images is enabled)
            if args.save_images:
                fname = save_image_single(
                    output_img[0],  # Remove batch dimension
                    prompt,
                    args.output_dir,
                    args.resize_to,
                    prompt_id,
                    sample_idx,
                )
                prompt_log.append(f"{fname}\tprompt_id={prompt_id}\tcentroid_id={centroid_id}\t{prompt}")

            if (prompt_id + 1) % 50 == 0 or prompt_id == len(prompts) - 1:
                print(f"  [Phase 2] Processed {prompt_id + 1}/{len(prompts)} prompts")

    total_time = time.time() - total_start
    print(
        f"\n[Complete] Generated {len(prompts) * args.num_images_per_prompt} images in {total_time:.2f}s. "
        f"Outputs in {args.output_dir}"
    )

    # Write prompt log with centroid info (only if images were saved)
    if args.save_images and prompt_log:
        log_file_path = os.path.join(args.output_dir, "prompts_with_clusters.txt")
        with open(log_file_path, "w") as f:
            f.write("# filename\tprompt_id\tcentroid_id\tprompt_text\n")
            f.write("\n".join(prompt_log))
        print(f"[Output] Prompt log with cluster info written to: {log_file_path}")


def run_inference(args):
    """Main entry point that dispatches based on cluster_algo."""
    # Start total pipeline timer
    pipeline_start = time.time()

    if args.cluster_algo == 3:
        # No clustering - use original behavior
        print("[Mode] No clustering (cluster_algo=3) - using original inference flow")
        run_inference_no_clustering(args)
    elif args.cluster_algo in [1, 2]:
        # Clustering enabled
        algo_name = "KMeans" if args.cluster_algo == 1 else "HDBSCAN"
        print(f"[Mode] Clustering enabled (cluster_algo={args.cluster_algo}, algorithm={algo_name})")
        run_inference_with_clustering(args)
    else:
        raise ValueError(f"Invalid cluster_algo: {args.cluster_algo}. Must be 1 (KMeans), 2 (HDBSCAN), or 3 (None).")

    # Report total pipeline time (includes model loading, clustering, inference, saving)
    pipeline_total_time = time.time() - pipeline_start
    print(f"\n[Total Pipeline Time] {pipeline_total_time:.2f}s")


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
    parser.add_argument("--batch_size", type=int, default=4, help="Number of prompts per batch.")
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
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=False,
        help="Save generated images to disk. Default is False (only generate, don't save).",
    )

    # ==========================================================================
    # NEW: Clustering arguments
    # ==========================================================================
    parser.add_argument(
        "--cluster_algo",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help=(
            "Clustering algorithm for shared patch generation: "
            "1 = KMeans, 2 = HDBSCAN, 3 = No clustering (default, original behavior)"
        ),
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=50,
        help="Number of clusters for KMeans (only used when cluster_algo=1).",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN (only used when cluster_algo=2).",
    )
    parser.add_argument(
        "--hdbscan_min_samples",
        type=int,
        default=3,
        help="Minimum samples for HDBSCAN core points (only used when cluster_algo=2).",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=3,
        help=(
            "The stage index up to which f_hat is cached and shared. "
            "Stages 0..alpha use the centroid's cached f_hat; stages alpha+1.. use the prompt's own context. "
            "Default is 3 (first 4 stages shared)."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)
