import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hart.modules.models.transformer import HARTForT2I
from hart.utils import encode_prompts, llm_system_prompt

DEFAULT_DATASET_NAME = "playgroundai/MJHQ-30K"


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


def _pool_prompt_embeddings(
    context_tensor: torch.Tensor,
    context_mask: torch.Tensor,
    truncate_tokens: Optional[int] = None,
) -> torch.Tensor:
    del context_mask
    if truncate_tokens is not None:
        if truncate_tokens <= 0:
            raise ValueError("truncate_tokens must be positive when provided.")
        if context_tensor.size(1) < truncate_tokens:
            raise ValueError(
                f"context_tensor must have at least {truncate_tokens} tokens to truncate."
            )
        context_tensor = context_tensor[:, :truncate_tokens, :]
    return context_tensor.reshape(context_tensor.size(0), -1)


def _run_kmeans(
    embeddings: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
) -> torch.Tensor:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")
    if embeddings.size(0) < num_clusters:
        raise ValueError(
            "num_clusters cannot exceed the number of prompt embeddings."
        )

    centroids = embeddings[torch.randperm(embeddings.size(0))[:num_clusters]].clone()
    assignments = torch.zeros(embeddings.size(0), dtype=torch.long)
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
                replacement_idx = torch.randint(0, embeddings.size(0), ()).item()
                centroids[idx] = embeddings[replacement_idx]
    else:
        distances = torch.cdist(embeddings, centroids)
        assignments = distances.argmin(dim=1)

    return assignments


def _run_hdbscan(
    embeddings: torch.Tensor,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN clustering requires the `hdbscan` package. "
            "Install it with `pip install hdbscan`."
        ) from exc

    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive.")
    if min_samples is None:
        min_samples = min_cluster_size

    embeddings_np = embeddings.detach().cpu().numpy()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
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


def _build_cluster_metadata(
    context_tensor: torch.Tensor,
    context_mask: torch.Tensor,
    method: str,
    *,
    num_clusters: Optional[int],
    truncate_tokens: Optional[int],
    kmeans_iters: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: Optional[int],
    log_prefix: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    method = method.lower()
    if method == "none":
        return None, None, 0

    context_tensor_cpu = context_tensor.detach().float().cpu()
    context_mask_cpu = context_mask.detach().cpu()
    pooled_embeddings = _pool_prompt_embeddings(
        context_tensor_cpu,
        context_mask_cpu,
        truncate_tokens=truncate_tokens,
    ).float()

    try:
        if method == "kmeans":
            if num_clusters is None or num_clusters <= 0:
                raise ValueError("num_clusters must be positive for k-means.")
            effective_clusters = min(int(num_clusters), pooled_embeddings.size(0))
            if effective_clusters <= 0:
                raise ValueError("Not enough prompts to form clusters.")
            assignments = _run_kmeans(
                pooled_embeddings.detach(),
                effective_clusters,
                num_iters=kmeans_iters,
            )
            cluster_count = effective_clusters
        elif method == "hdbscan":
            assignments, cluster_count = _run_hdbscan(
                pooled_embeddings.detach(),
                min_cluster_size=hdbscan_min_cluster_size,
                min_samples=hdbscan_min_samples,
            )
        else:
            raise ValueError(f"Unsupported clustering method '{method}'.")
    except Exception as exc:  # noqa: BLE001
        print(f"{log_prefix}Skipping clustering: {exc}")
        return None, None, 0

    if assignments.numel() == 0 or cluster_count <= 0:
        print(f"{log_prefix}Skipping clustering: no valid clusters found.")
        return None, None, 0

    assignments = assignments.to(dtype=torch.long)
    cluster_contexts = []
    for cluster_idx in range(assignments.max().item() + 1):
        mask = assignments == cluster_idx
        if mask.any():
            cluster_contexts.append(context_tensor_cpu[mask].mean(dim=0))
        else:
            cluster_contexts.append(context_tensor_cpu.mean(dim=0))
    cluster_context_tensor = torch.stack(cluster_contexts, dim=0)

    cluster_count = assignments.max().item() + 1

    return assignments, cluster_context_tensor, cluster_count


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    set_seed(args.seed)

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
        ema_path = os.path.join(args.model_path, "ema_model.bin")
        if os.path.exists(ema_path):
            ema_state = torch.load(ema_path, map_location=device)
            ema_model.load_state_dict(ema_state)
            infer_model = ema_model
        else:
            print(
                f"EMA weights not found at '{ema_path}'. Continuing with base model."
            )

    infer_func = infer_model.autoregressive_infer_cfg

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    with torch.inference_mode(), torch.autocast(
        "cuda",
        enabled=True,
        dtype=torch.float16,
        cache_enabled=True,
    ):
        warmup_prompts, _ = _prepare_batch(
            dataset=dataset,
            start=0,
            batch_size=args.batch_size,
            prompt_column=prompt_column,
            prompt_feature=prompt_feature,
        )
        (
            _,
            warmup_context_mask,
            warmup_context_position_ids,
            warmup_context_tensor,
        ) = encode_prompts(
            warmup_prompts,
            text_model,
            text_tokenizer,
            args.max_token_length,
            llm_system_prompt,
            args.use_llm_system_prompt,
        )

        (
            warmup_assignments,
            warmup_cluster_context,
            warmup_cluster_count,
        ) = _build_cluster_metadata(
            warmup_context_tensor,
            warmup_context_mask,
            args.cluster_method,
            num_clusters=args.num_clusters,
            truncate_tokens=args.cluster_truncate_tokens,
            kmeans_iters=args.kmeans_iters,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            log_prefix="[Warmup] ",
        )

        if warmup_cluster_count:
            print(f"[Warmup] Generated {warmup_cluster_count} clusters.")

        warmup_context_tensor = warmup_context_tensor.to(device)
        warmup_context_mask = warmup_context_mask.to(device)
        warmup_context_position_ids = warmup_context_position_ids.to(device)

        for _ in tqdm(range(args.warmup_iter), desc="Warmup", leave=False):
            infer_func(
                B=warmup_context_tensor.size(0),
                label_B=warmup_context_tensor,
                cluster_assignments=warmup_assignments,
                cluster_context_tensor=warmup_cluster_context,
                cluster_warmup_steps=args.cluster_warmup_steps,
                cfg=args.cfg,
                g_seed=args.seed,
                more_smooth=args.more_smooth,
                context_position_ids=warmup_context_position_ids,
                context_mask=warmup_context_mask,
            )

        torch.cuda.synchronize()

        total_images = 0
        profiled_batches = 0
        max_batches = (
            args.profile_iter
            if args.profile_iter is not None and args.profile_iter > 0
            else None
        )
        total_clusters = 0

        torch.cuda.synchronize()
        start_time = time.time()

        for batch_start in tqdm(
            range(0, len(dataset), args.batch_size),
            desc="Profiling",
        ):
            if max_batches is not None and profiled_batches >= max_batches:
                break

            batch_prompts, actual = _prepare_batch(
                dataset=dataset,
                start=batch_start,
                batch_size=args.batch_size,
                prompt_column=prompt_column,
                prompt_feature=prompt_feature,
            )

            (
                _,
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

            (
                cluster_assignments,
                cluster_context_tensor,
                cluster_count,
            ) = _build_cluster_metadata(
                context_tensor,
                context_mask,
                args.cluster_method,
                num_clusters=args.num_clusters,
                truncate_tokens=args.cluster_truncate_tokens,
                kmeans_iters=args.kmeans_iters,
                hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
                hdbscan_min_samples=args.hdbscan_min_samples,
                log_prefix=f"[Batch {profiled_batches}] ",
            )

            total_clusters += cluster_count

            context_tensor = context_tensor.to(device)
            context_mask = context_mask.to(device)
            context_position_ids = context_position_ids.to(device)

            infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cluster_assignments=cluster_assignments,
                cluster_context_tensor=cluster_context_tensor,
                cluster_warmup_steps=args.cluster_warmup_steps,
                cfg=args.cfg,
                g_seed=args.seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
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
        f"in {total_time:.2f}s (average {average_time:.4f}s per image; total clusters {total_clusters})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="pretrained_models/HART-1024",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Generation batch size.",
        default=8,
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
    parser.add_argument("--warmup_iter", type=int, default=25)
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
        default=1024,
        help="Number of samples to profile (set <=0 to use the entire split).",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default=None,
        help="Optional prompt column override (defaults to 'label').",
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=["none", "kmeans", "hdbscan"],
        default="kmeans",
        help="Clustering method to apply to prompt embeddings.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=4,
        help="Number of clusters to form when using k-means.",
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
    args = parser.parse_args()

    main(args)
