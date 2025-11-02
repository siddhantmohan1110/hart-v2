#!/usr/bin/env python3
"""Utilities for counting long prompts inside the MJHQ-30K dataset."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

from datasets import Dataset, IterableDataset, load_dataset
from datasets.exceptions import DatasetNotFoundError

PROMPT_KEYS = ("prompt", "text", "caption", "prompts")


def _resolve_dataset(args: argparse.Namespace) -> Dataset | IterableDataset:
    """Load MJHQ-30K from HF Hub or a user-specified set of files."""
    if args.data_files:
        data_files = args.data_files
        if os.path.isdir(data_files):
            # Allow pointing at a directory that already stores split files.
            data_files = {
                split: os.path.join(data_files, f"{split}.jsonl")
                for split in ("train", "validation", "test")
                if os.path.exists(os.path.join(data_files, f"{split}.jsonl"))
            }
            if not data_files:
                raise FileNotFoundError(
                    f"No split JSONL files found under {args.data_files!r}."
                )
        return load_dataset("json", data_files=data_files, split=args.split or None)

    split = args.split or "test"
    try:
        return load_dataset("playgroundai/MJHQ-30K", split=split)
    except DatasetNotFoundError as err:
        raise RuntimeError(
            "Unable to locate 'playgroundai/MJHQ-30K' on the Hugging Face Hub. "
            "Pass a local dataset via --data-files (directory, glob, or JSONL file)."
        ) from err
    except ValueError as err:
        if "Unknown split" in str(err):
            raise RuntimeError(
                f"Split {split!r} is unavailable on 'playgroundai/MJHQ-30K'. "
                "Choose the listed split (e.g. --split test) or provide local data."
            ) from err
        raise
    except Exception as err:  # noqa: BLE001 - provide helpful hint for download issues
        raise RuntimeError(
            "Failed to download MJHQ-30K from the Hugging Face Hub. "
            "Check network access or provide a local dataset with --data-files."
        ) from err


def iter_prompts(dataset: Iterable[dict]) -> Iterable[str]:
    for example in dataset:
        for key in PROMPT_KEYS:
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                yield value
                break


def count_long_prompts(prompts: Iterable[str], threshold: int) -> int:
    return sum(1 for prompt in prompts if len(prompt.split()) > threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count how many MJHQ-30K prompts contain more than a given "
            "number of whitespace-separated terms."
        )
    )
    parser.add_argument(
        "--data-files",
        type=str,
        default=None,
        help=(
            "Optional path or glob pointing at a local JSON/JSONL dataset. "
            "When omitted, the script streams the dataset from the HF Hub."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to inspect (defaults to 'test').",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Count prompts whose term count is strictly greater than this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = _resolve_dataset(args)
    prompts = iter_prompts(dataset)
    count = count_long_prompts(prompts, args.threshold)
    print(f"Number of prompts with more than {args.threshold} terms: {count}")


if __name__ == "__main__":
    main()
