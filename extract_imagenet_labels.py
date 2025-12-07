"""Extract image-to-label pairs from an ImageNet-style validation JSON file.

The script expects a JSON object where keys are image filenames (e.g.
`ILSVRC2012_val_00048980.JPEG`) and values are the human-readable labels.
It writes the pairs as CSV with two columns: image,label.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_labels(path: Path) -> Dict[str, str]:
    """Load the validation labels JSON and ensure it is a mapping."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return {str(k): str(v) for k, v in data.items()}


def write_csv(rows: Iterable[Tuple[str, str]], out_path: Path | None) -> None:
    """Write image-label pairs to CSV."""
    if out_path is None or str(out_path) == "-":
        output = sys.stdout
        close_output = False
    else:
        output = out_path.open("w", newline="", encoding="utf-8")
        close_output = True

    try:
        writer = csv.writer(output)
        writer.writerow(["image", "label"])
        for image, label in rows:
            writer.writerow([image, label])
    finally:
        if close_output:
            output.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ImageNet validation labels from JSON to CSV."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to JSON file mapping image filenames to labels.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("-"),
        help="Output CSV path (default: stdout).",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort rows by image filename before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.json_path)
    items = sorted(labels.items()) if args.sort else labels.items()
    write_csv(items, args.out)


if __name__ == "__main__":
    main()
