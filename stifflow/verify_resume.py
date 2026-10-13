"""Compare continuous and interrupted/resumed Lightning runs on SHHPC."""

import argparse
import json
import os
from pathlib import Path

from .preflight import fingerprint
from .smoke import external_directory


def compare_states(left, right, path="checkpoint"):
    import torch

    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise TypeError(f"Resume type differs: {path}")
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7, msg=path)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"Resume keys differ: {path}")
        for key in left:
            compare_states(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise ValueError(f"Resume sequence differs: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            compare_states(a, b, f"{path}[{index}]")
    elif left != right:
        raise ValueError(f"Resume value differs: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Full checkpoint comparisons belong in an SHHPC allocation")
    if args.output.exists():
        parser.error("Existing acceptance reports are never overwritten")
    import torch

    left = torch.load(args.continuous, map_location="cpu", weights_only=False)
    right = torch.load(args.resumed, map_location="cpu", weights_only=False)
    keys = (
        "global_step",
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "training_signature",
        "stifflow_rng",
        "torch_rng",
        "gradient_queue",
    )
    if "cuda_rng" in left or "cuda_rng" in right:
        keys += ("cuda_rng",)
    if "consumed_samples" in left or "consumed_samples" in right:
        keys += ("consumed_samples",)
    for key in keys:
        compare_states(left[key], right[key], key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "passed": True,
                "continuous": fingerprint(args.continuous),
                "resumed": fingerprint(args.resumed),
                "global_step": left["global_step"],
                "checks": keys,
                "float_rtol": 1e-6,
                "float_atol": 1e-7,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
