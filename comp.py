import argparse
from collections.abc import Mapping, Sequence
from typing import Any

import torch

DEFAULT_FIRST = "./fhat_images/x_before_logits_2_with_sharing.pt"
DEFAULT_SECOND = "./fhat_images/x_before_logits_2.pt"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor


def _iter_tensors(obj: Any, path: str = "root"):
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from _iter_tensors(value, f"{path}.{key}")
    elif _is_sequence(obj):
        for idx, value in enumerate(obj):
            yield from _iter_tensors(value, f"{path}[{idx}]")


def _tensor_stats(tensor: torch.Tensor) -> tuple[tuple[int, ...], float, float]:
    tensor = _cpu_tensor(tensor).float()
    if tensor.numel() == 0:
        return tuple(tensor.shape), float("nan"), float("nan")
    mean = tensor.mean().item()
    var = tensor.var(unbiased=False).item()
    return tuple(tensor.shape), mean, var


def _describe_object(label: str, obj: Any) -> None:
    tensors = list(_iter_tensors(obj))
    if not tensors:
        print(f"{label}: no tensors found (type {type(obj).__name__})")
        return

    if len(tensors) == 1 and tensors[0][0] == "root":
        shape, mean, var = _tensor_stats(tensors[0][1])
        print(f"{label}: shape={shape}, mean={mean:.6g}, var={var:.6g}")
        return

    print(f"{label}: contains {len(tensors)} tensor(s). Showing first 5:")
    for path, tensor in tensors[:5]:
        shape, mean, var = _tensor_stats(tensor)
        print(f"  - {path}: shape={shape}, mean={mean:.6g}, var={var:.6g}")
    remaining = len(tensors) - 5
    if remaining > 0:
        print(f"  ... {remaining} more tensor(s) omitted.")


def _compare(obj_a: Any, obj_b: Any, path: str = "root") -> tuple[bool, str]:
    if isinstance(obj_a, torch.Tensor) and isinstance(obj_b, torch.Tensor):
        obj_a = _cpu_tensor(obj_a)
        obj_b = _cpu_tensor(obj_b)
        if torch.equal(obj_a, obj_b):
            return True, ""
        return False, f"Tensor mismatch at {path}"

    if isinstance(obj_a, Mapping) and isinstance(obj_b, Mapping):
        if obj_a.keys() != obj_b.keys():
            missing_a = obj_a.keys() - obj_b.keys()
            missing_b = obj_b.keys() - obj_a.keys()
            return (
                False,
                f"Key mismatch at {path}: "
                f"only in first {sorted(missing_a)}; only in second {sorted(missing_b)}",
            )
        for key in obj_a:
            ok, reason = _compare(obj_a[key], obj_b[key], f"{path}.{key}")
            if not ok:
                return ok, reason
        return True, ""

    if _is_sequence(obj_a) and _is_sequence(obj_b):
        if len(obj_a) != len(obj_b):
            return False, f"Sequence length mismatch at {path}: {len(obj_a)} != {len(obj_b)}"
        for idx, (item_a, item_b) in enumerate(zip(obj_a, obj_b)):
            ok, reason = _compare(item_a, item_b, f"{path}[{idx}]")
            if not ok:
                return ok, reason
        return True, ""

    if obj_a == obj_b:
        return True, ""
    return False, f"Value mismatch at {path}: {obj_a!r} != {obj_b!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two serialized PyTorch objects (.pt files).")
    parser.add_argument(
        "first",
        nargs="?",
        default=DEFAULT_FIRST,
        help=f"Path to the first .pt file (default: {DEFAULT_FIRST})",
    )
    parser.add_argument(
        "second",
        nargs="?",
        default=DEFAULT_SECOND,
        help=f"Path to the second .pt file (default: {DEFAULT_SECOND})",
    )
    args = parser.parse_args()

    first_obj = torch.load(args.first, map_location="cpu")
    second_obj = torch.load(args.second, map_location="cpu")

    _describe_object("First file", first_obj)
    _describe_object("Second file", second_obj)

    equal, reason = _compare(first_obj, second_obj)

    if equal:
        print("Files are identical.")
    else:
        print("Files differ.")
        print(reason)


if __name__ == "__main__":
    main()
