import argparse
import copy
import os
import time
import pickle
import json
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity

from hart.utils import encode_prompts, llm_system_prompt
from hart.modules.models.transformer import HARTForT2I

DEFAULT_DATASET_NAME = "/scratch/ar6316/adv_proj/hart-v2/MJHQ-30K/"


def _resolve_prompt_column(dataset, requested: Optional[str]) -> str:
    if requested:
        if requested not in dataset.column_names:
            raise ValueError(
                f"Column '{requested}' not found. Available columns: {dataset.column_names}."
            )
        return requested

    default_column = "label"
    if default_column in dataset.column_names:
        return default_column

    raise ValueError(
        "Unable to infer prompt column. Specify with --prompt_column."
    )


def _prepare_batch(
    dataset,
    start: int,
    batch_size: int,
    prompt_column: str,
    prompt_feature,
) -> Tuple[List[str], int]:
    end = min(start + batch_size, len(dataset))
    raw = dataset[start:end]

    prompts_raw = list(raw[prompt_column])
    if prompt_feature is not None and hasattr(prompt_feature, "int2str"):
        prompts = [prompt_feature.int2str(int(p)) for p in prompts_raw]
    else:
        prompts = [str(p) for p in prompts_raw]

    actual = len(prompts)
    if actual == 0:
        raise ValueError(
            "Encountered empty batch while iterating over the dataset."
        )

    if actual < batch_size:
        pad_prompt = prompts[-1]
        prompts.extend([pad_prompt] * (batch_size - actual))

    return prompts, actual


def generate_cluster_centroids(
    model,
    text_model,
    text_tokenizer,
    prompts: List[str],
    num_clusters: int,
    stage_N: int,
    device: torch.device,
    output_path: str,
    max_samples: int = 1000,
    batch_size: int = 1,
) -> None:
    """
    Generate cluster centroids by clustering prompts and generating intermediate features.
    
    Args:
        model: HART model
        text_model: Text encoder model
        text_tokenizer: Text tokenizer
        prompts: List of text prompts
        num_clusters: Number of clusters to create
        stage_N: Stage number to generate patches up to
        device: Device to run on
        output_path: Path to save the centroids
        max_samples: Maximum number of samples to use for clustering
        batch_size: Batch size for generation
    """
    print(f"Generating {num_clusters} cluster centroids for stage {stage_N}...")
    
    # Limit prompts for efficiency
    prompts = prompts[:max_samples]
    
    # Encode all prompts
    print("Encoding prompts...")
    prompt_embeddings = []
    embeddings_to_prompts = {}
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding"):
        batch_prompts = prompts[i:i + batch_size]
        _, _, _, context_tensor = encode_prompts(
            batch_prompts,
            text_model,
            text_tokenizer,
            300,
            llm_system_prompt,
            True,
        )
        
        # Get the mean embedding
        with torch.no_grad():
            embeddings = context_tensor.mean(dim=1).cpu().numpy()
            prompt_embeddings.append(embeddings)
            
            for j, prompt in enumerate(batch_prompts):
                idx = i + j
                if idx < len(embeddings):
                    embeddings_to_prompts[len(embeddings_to_prompts)] = prompt
    
    prompt_embeddings = np.concatenate(prompt_embeddings, axis=0)
    
    # Cluster the prompts
    print(f"Clustering {len(prompt_embeddings)} prompts into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(prompt_embeddings)
    
    # Generate patches for each cluster
    print("Generating patches for each cluster...")
    cluster_patches = {}
    
    # Store cluster assignment for each cluster
    cluster_to_prompts = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_to_prompts[label].append(prompts[idx])
    
    for cluster_id in tqdm(range(num_clusters), desc="Generating cluster patches"):
        cluster_prompts = cluster_to_prompts[cluster_id]
        
        if not cluster_prompts:
            continue
        
        # Use the first prompt in each cluster as representative
        # You could also use the centroid prompt or aggregate results
        representative_prompt = cluster_prompts[0]
        
        # Generate patches up to stage_N
        _, _, _, context_tensor = encode_prompts(
            [representative_prompt],
            text_model,
            text_tokenizer,
            300,
            llm_system_prompt,
            True,
        )
        
        # Generate intermediate features
        patches = []
        with torch.inference_mode(), torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):
            # Create a custom function to capture intermediate features
            patches = _generate_intermediate_patches(
                model, context_tensor, stage_N, device
            )
        
        cluster_patches[cluster_id] = {
            'patches': patches,
            'representative_prompt': representative_prompt,
            'cluster_prompts': cluster_prompts[:10],  # Store first 10 for reference
            'cluster_centroid': kmeans.cluster_centers_[cluster_id].tolist(),
        }
    
    # Save to file
    print(f"Saving cluster centroids to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_data = {
        'cluster_patches': cluster_patches,
        'kmeans_model': kmeans,
        'embeddings_to_prompts': embeddings_to_prompts,
        'stage_N': stage_N,
        'num_clusters': num_clusters,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Also save human-readable info
    info_path = output_path.replace('.pkl', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(
            {
                'num_clusters': num_clusters,
                'stage_N': stage_N,
                'cluster_sizes': {
                    str(i): len(prompts) for i, prompts in cluster_to_prompts.items()
                },
                'representative_prompts': {
                    str(i): data['representative_prompt']
                    for i, data in cluster_patches.items()
                },
            },
            f,
            indent=2,
        )
    
    print(f"Cluster centroids saved to {output_path}")


def _generate_intermediate_patches(model, context_tensor, stage_N, device):
    """
    Helper function to generate intermediate patches up to stage_N.
    
    TODO: This is a placeholder implementation. In a complete implementation,
    you would need to:
    1. Run inference up to stage_N
    2. Capture the accumulated f_hat feature maps at each stage
    3. Store them as a list of patches
    
    The patches should be the feature maps (B, Cvae, H, W) representing
    the accumulated features at each stage.
    
    This might require modifying the HART model to expose intermediate states
    or creating a custom inference loop that tracks f_hat at each stage.
    """
    patches = []
    B = context_tensor.shape[0]
    
    # TODO: Implement actual patch generation
    # This should capture f_hat at stages 0 through stage_N
    # Each patch should be shape (B, Cvae, patch_size, patch_size)
    
    print(f"Warning: Intermediate patch generation needs implementation")
    
    return patches


def load_cluster_centroids(
    centroid_path: str,
    prompt: str,
    text_model,
    text_tokenizer,
    device: torch.device,
) -> Tuple[Optional[List[torch.Tensor]], Optional[int], Optional[int]]:
    """
    Find which cluster a prompt belongs to and load the corresponding patches.
    
    Args:
        centroid_path: Path to the saved cluster centroids
        prompt: Text prompt
        text_model: Text encoder model
        text_tokenizer: Text tokenizer
        device: Device to run on
    
    Returns:
        Tuple of (cluster_patches, cluster_stage_N, cluster_assignment)
    """
    if not os.path.exists(centroid_path):
        print(f"Warning: Cluster centroids not found at {centroid_path}")
        return None, None, None
    
    with open(centroid_path, 'rb') as f:
        data = pickle.load(f)
    
    kmeans = data['kmeans_model']
    cluster_patches = data['cluster_patches']
    stage_N = data['stage_N']
    
    # Encode the prompt
    _, _, _, context_tensor = encode_prompts(
        [prompt],
        text_model,
        text_tokenizer,
        300,
        llm_system_prompt,
        True,
    )
    
    with torch.no_grad():
        prompt_embedding = context_tensor.mean(dim=1).cpu().numpy()
    
    # Find the closest cluster
    cluster_centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(cluster_centroids - prompt_embedding, axis=1)
    cluster_id = np.argmin(distances)
    
    if cluster_id not in cluster_patches:
        print(f"Warning: Cluster {cluster_id} not found in patches")
        return None, None, None
    
    patches = cluster_patches[cluster_id]['patches']
    return patches, stage_N, cluster_id


def _generate_centroids_mode(args: argparse.Namespace, device: torch.device) -> None:
    """Generate cluster centroids from the dataset."""
    # Load models
    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_state = torch.load(
            os.path.join(args.model_path, "ema_model.bin"), map_location=device
        )
        ema_model.load_state_dict(ema_state)
        model = ema_model
    
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()
    
    # Load dataset
    try:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    except ValueError as err:
        if "Unknown split" not in str(err):
            raise
        dataset_dict = load_dataset(args.dataset_name)
        fallback_split = list(dataset_dict.keys())[0]
        dataset = dataset_dict[fallback_split]
    
    if args.dataset_limit and args.dataset_limit > 0:
        limit = min(args.dataset_limit, len(dataset))
        dataset = dataset.select(range(limit))
    
    prompt_column = _resolve_prompt_column(dataset, args.prompt_column)
    prompts = [str(item[prompt_column]) for item in dataset]
    
    # Generate centroids
    output_path = args.centroid_output_path or f"./cluster_centroids/cluster_centroids_stage_{args.centroid_stage_N}.pkl"
    
    generate_cluster_centroids(
        model=model,
        text_model=text_model,
        text_tokenizer=text_tokenizer,
        prompts=prompts,
        num_clusters=args.num_clusters,
        stage_N=args.centroid_stage_N,
        device=device,
        output_path=output_path,
        max_samples=args.centroid_max_samples,
        batch_size=args.batch_size,
    )




def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    set_seed(args.seed)
    
    # Handle cluster centroid generation mode
    if args.generate_centroids:
        _generate_centroids_mode(args, device)
        return

    try:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    except ValueError as err:
        if "Unknown split" not in str(err):
            raise
        dataset_dict = load_dataset(args.dataset_name)
        available_splits = list(dataset_dict.keys())
        if not available_splits:
            raise ValueError(
                f"Dataset '{args.dataset_name}' does not expose any splits."
            ) from err
        fallback_split = available_splits[0]
        print(
            f"Split '{args.dataset_split}' not found. Falling back to '{fallback_split}'."
        )
        dataset = dataset_dict[fallback_split]
        args.dataset_split = fallback_split
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset '{args.dataset_name}' split '{args.dataset_split}' is empty."
        )

    if args.dataset_limit and args.dataset_limit > 0:
        limit = min(args.dataset_limit, len(dataset))
        if limit < len(dataset):
            dataset = dataset.select(range(limit))
    else:
        limit = len(dataset)

    limit = len(dataset)

    prompt_column = _resolve_prompt_column(dataset, args.prompt_column)
    features = getattr(dataset, "features", None)
    prompt_feature = (
        features.get(prompt_column)
        if features is not None and hasattr(features, "get")
        else None
    )

    model = AutoModel.from_pretrained(args.model_path).to(device)
    model.eval()

    infer_model = model
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_state = torch.load(
            os.path.join(args.model_path, "ema_model.bin"), map_location=device
        )
        ema_model.load_state_dict(ema_state)
        infer_model = ema_model

    infer_func = infer_model.autoregressive_infer_cfg

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    with torch.inference_mode(), torch.autocast(
        "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
    ):
        warmup_prompts, _ = _prepare_batch(
            dataset,
            start=0,
            batch_size=args.batch_size,
            prompt_column=prompt_column,
            prompt_feature=prompt_feature,
        )
        (
            context_tokens,
            context_mask,
            context_position_ids,
            context_tensor,
        ) = encode_prompts(
            warmup_prompts,
            text_model,
            text_tokenizer,
            args.max_token_length,
            llm_system_prompt,
            args.use_llm_system_prompt,
        )

        for _ in tqdm(range(args.warmup_iter), desc="Warmup", leave=False):
            infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=args.cfg,
                g_seed=args.seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )
        torch.cuda.synchronize()

        total_images = 0
        profiled_batches = 0
        max_batches = (
            args.profile_iter if args.profile_iter is not None and args.profile_iter > 0 else None
        )
        torch.cuda.synchronize()
        start_time = time.time()

        for batch_start in tqdm(
            range(0, len(dataset), args.batch_size),
            desc="Profiling",
        ):
            if max_batches is not None and profiled_batches >= max_batches:
                break

            batch_prompts, actual = _prepare_batch(
                dataset,
                start=batch_start,
                batch_size=args.batch_size,
                prompt_column=prompt_column,
                prompt_feature=prompt_feature,
            )

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

            # Load cluster centroids if available
            cluster_patches = None
            cluster_stage_N = None
            if args.centroid_path:
                # For batch inference, use the first prompt for clustering
                # In a full implementation, you might want to handle each prompt separately
                first_prompt = batch_prompts[0]
                cluster_patches, cluster_stage_N, cluster_id = load_cluster_centroids(
                    args.centroid_path,
                    first_prompt,
                    text_model,
                    text_tokenizer,
                    device,
                )
                if args.verbose:
                    print(f"Prompt assigned to cluster {cluster_id}")
            
            infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=args.cfg,
                g_seed=args.seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
                cluster_centroid_patches=cluster_patches if cluster_patches else None,
                cluster_stage_N=cluster_stage_N if cluster_patches else None,
            )
            total_images += actual
            profiled_batches += 1

        torch.cuda.synchronize()
        total_time = time.time() - start_time

    if total_images == 0:
        raise RuntimeError("No images were generated during profiling.")

    if max_batches is not None and total_images < limit:
        print(
            f"Stopped early after {profiled_batches} batches due to --profile_iter={max_batches}."
        )

    average_time = total_time / total_images
    print(
        f"Generated {total_images} images from {limit} prompts on split '{args.dataset_split}' "
        f"in {total_time:.2f}s (average {average_time:.4f}s per image)"
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="./hart-0.7b-1024px/llm",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="./Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Generation batch size", default=1
    )
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
    parser.add_argument("--warmup_iter", type=int, default=50)
    parser.add_argument(
        "--profile_iter",
        type=int,
        default=-1,
        help="Optional cap on batches to profile; set <=0 to cover the entire dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset identifier to profile.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to load (falls back to first available if missing).",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=1000,
        help="Number of samples to profile (set <=0 to use the entire split).",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default=None,
        help="Optional prompt column override (defaults to 'label').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    
    # Cluster centroid arguments
    parser.add_argument(
        "--generate_centroids",
        action="store_true",
        help="Generate cluster centroids instead of running inference.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=10,
        help="Number of clusters to create.",
    )
    parser.add_argument(
        "--centroid_stage_N",
        type=int,
        default=2,
        help="Stage N up to which to generate cluster centroids.",
    )
    parser.add_argument(
        "--centroid_output_path",
        type=str,
        default="./cluster_centroids/cluster_centroids_stage_{args.centroid_stage_N}.pkl",
        help="Path to save cluster centroids.",
    )
    parser.add_argument(
        "--centroid_max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to use for clustering.",
    )
    parser.add_argument(
        "--centroid_path",
        type=str,
        default=None,
        help="Path to load cluster centroids from for inference.",
    )
    
    args = parser.parse_args()

    main(args)
