import argparse
import copy
import os
import time
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

from hart.utils import encode_prompts, llm_system_prompt
from hart.modules.models.transformer import HARTForT2I

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

            infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
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
        f"in {total_time:.2f}s (average {average_time:.4f}s per image)"
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
    args = parser.parse_args()

    main(args)
