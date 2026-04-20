"""Bounded CPU-only pilot training helpers for tiny real artifacts."""

from __future__ import annotations

import math
from typing import Any


class TinyPilotTorchUnavailableError(RuntimeError):
    """Raised when PyTorch is unavailable for the bounded pilot."""


MICRO_PILOT_HARD_MAX_SAMPLES = 6


def run_tiny_velocity_pilot_training(
    samples: list[dict[str, Any]],
    seed: int,
    max_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Run a strictly bounded CPU training loop on tiny real flow samples.

    This is intentionally small and non-benchmark: only a bounded tiny real
    subset is optimized, while validation and test losses are recorded as
    interface diagnostics.
    """

    if not samples:
        raise ValueError("Cannot run tiny pilot training on an empty sample list")
    if max_steps <= 1:
        raise ValueError("Tiny pilot training requires more than one optimizer step")
    if max_steps > 8:
        raise ValueError("Tiny pilot training is capped at eight optimizer steps")

    split_counts = _split_counts(samples)
    if len(samples) > MICRO_PILOT_HARD_MAX_SAMPLES:
        raise ValueError(f"Tiny pilot training is capped at {MICRO_PILOT_HARD_MAX_SAMPLES} samples")
    if any(split_counts[split_name] <= 0 for split_name in ["train", "val", "test"]):
        raise ValueError(f"Tiny pilot requires nonempty train/val/test splits, got {split_counts}")

    torch = _load_torch()
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    features_by_split: dict[str, Any] = {}
    targets_by_split: dict[str, Any] = {}
    per_sample_slices_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in ["train", "val", "test"]:
        split_samples = [sample for sample in samples if sample.get("split") == split_name]
        features, targets, slices = _batch_tensors(torch, split_samples)
        features_by_split[split_name] = features
        targets_by_split[split_name] = targets
        per_sample_slices_by_split[split_name] = slices

    model = torch.nn.Linear(4, 3)
    model.to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()

    forward_pass_count = 0
    initial_split_losses, forward_pass_count = _split_losses(
        torch=torch,
        model=model,
        loss_fn=loss_fn,
        features_by_split=features_by_split,
        targets_by_split=targets_by_split,
        forward_pass_count=forward_pass_count,
    )

    step_records = []
    for step_index in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        train_prediction = model(features_by_split["train"])
        forward_pass_count += 1
        train_loss = loss_fn(train_prediction, targets_by_split["train"])
        train_loss.backward()
        grad_norm = _grad_norm(model.parameters())
        optimizer.step()

        split_losses, forward_pass_count = _split_losses(
            torch=torch,
            model=model,
            loss_fn=loss_fn,
            features_by_split=features_by_split,
            targets_by_split=targets_by_split,
            forward_pass_count=forward_pass_count,
        )
        step_records.append(
            {
                "step_index": step_index,
                "optimizer_step": step_index + 1,
                "train_loss_before_step": float(train_loss.detach().cpu().item()),
                "gradient_l2_norm": grad_norm,
                "learning_rate": float(learning_rate),
                "split_losses_after_step": split_losses,
            }
        )

    final_split_losses = step_records[-1]["split_losses_after_step"]
    per_sample = _final_per_sample_metrics(
        torch=torch,
        model=model,
        features_by_split=features_by_split,
        targets_by_split=targets_by_split,
        per_sample_slices_by_split=per_sample_slices_by_split,
    )
    forward_pass_count += len(["train", "val", "test"])

    return {
        "model_name": "single_linear_velocity_pilot",
        "backend": "torch_cpu",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_name": "SGD",
        "learning_rate": float(learning_rate),
        "sample_count": len(samples),
        "split_sample_counts": split_counts,
        "atom_count": int(sum(features_by_split[split].shape[0] for split in ["train", "val", "test"])),
        "train_atom_count": int(features_by_split["train"].shape[0]),
        "component_count": int(sum(targets_by_split[split].numel() for split in ["train", "val", "test"])),
        "max_steps": int(max_steps),
        "forward_pass_count": int(forward_pass_count),
        "backward_pass_count": int(max_steps),
        "optimizer_step_count": int(max_steps),
        "initial_split_losses": initial_split_losses,
        "final_split_losses": final_split_losses,
        "step_records": step_records,
        "per_sample": per_sample,
        "note": (
            "Bounded tiny real-data pilot training diagnostics only; nonempty train/validation/test "
            "splits, CPU-only, not benchmark metrics."
        ),
    }


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise TinyPilotTorchUnavailableError(
            "PyTorch is required for the bounded tiny real pilot. No pilot outputs were written."
        ) from exc
    return torch


def _split_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for sample in samples:
        split_name = sample.get("split")
        if split_name not in counts:
            raise ValueError(f"Unsupported tiny pilot split label: {split_name}")
        counts[split_name] += 1
    return counts


def _batch_tensors(torch: Any, samples: list[dict[str, Any]]) -> tuple[Any, Any, list[dict[str, Any]]]:
    feature_rows = []
    target_rows = []
    per_sample_slices = []
    cursor = 0

    for sample in samples:
        xt = sample["xt"]
        target_velocity = sample["target_velocity"]
        if len(xt) != len(target_velocity):
            raise ValueError(f"Atom-count mismatch in sample {sample['molecule_id']}")
        start = cursor
        for atom_xt, atom_velocity in zip(xt, target_velocity):
            feature_rows.append([float(atom_xt[0]), float(atom_xt[1]), float(atom_xt[2]), float(sample["t"])])
            target_rows.append([float(atom_velocity[0]), float(atom_velocity[1]), float(atom_velocity[2])])
            cursor += 1
        per_sample_slices.append(
            {
                "molecule_id": sample["molecule_id"],
                "split": sample.get("split"),
                "start": start,
                "stop": cursor,
            }
        )

    return (
        torch.tensor(feature_rows, dtype=torch.float32, device="cpu"),
        torch.tensor(target_rows, dtype=torch.float32, device="cpu"),
        per_sample_slices,
    )


def _split_losses(
    torch: Any,
    model: Any,
    loss_fn: Any,
    features_by_split: dict[str, Any],
    targets_by_split: dict[str, Any],
    forward_pass_count: int,
) -> tuple[dict[str, float], int]:
    losses = {}
    with torch.no_grad():
        for split_name in ["train", "val", "test"]:
            prediction = model(features_by_split[split_name])
            forward_pass_count += 1
            loss = loss_fn(prediction, targets_by_split[split_name])
            losses[split_name] = float(loss.detach().cpu().item())
    return losses, forward_pass_count


def _final_per_sample_metrics(
    torch: Any,
    model: Any,
    features_by_split: dict[str, Any],
    targets_by_split: dict[str, Any],
    per_sample_slices_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for split_name in ["train", "val", "test"]:
            prediction = model(features_by_split[split_name])
            target = targets_by_split[split_name]
            for item in per_sample_slices_by_split[split_name]:
                pred_slice = prediction[item["start"] : item["stop"]]
                target_slice = target[item["start"] : item["stop"]]
                squared_error = (pred_slice - target_slice) ** 2
                absolute_error = torch.abs(pred_slice - target_slice)
                rows.append(
                    {
                        "molecule_id": item["molecule_id"],
                        "split": item["split"],
                        "atom_count": int(pred_slice.shape[0]),
                        "component_count": int(pred_slice.numel()),
                        "velocity_mse_after_pilot": float(squared_error.mean().detach().cpu().item()),
                        "velocity_mae_after_pilot": float(absolute_error.mean().detach().cpu().item()),
                    }
                )
    return rows


def _grad_norm(parameters: Any) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        squared += float((grad * grad).sum().cpu().item())
    return math.sqrt(squared)
