"""Tiny CPU-only trainable smoke step helpers."""

from __future__ import annotations

import math
from typing import Any


class TorchUnavailableError(RuntimeError):
    """Raised when the single-step smoke test cannot import PyTorch."""


def run_single_step_velocity_training_smoke(
    samples: list[dict[str, Any]],
    seed: int,
    learning_rate: float = 1.0e-2,
) -> dict[str, Any]:
    """Run exactly one CPU forward/loss/backward/optimizer step.

    The model is intentionally tiny: a single linear velocity head over per-atom
    Cartesian features. It exists only to validate trainable plumbing on the
    bounded tiny real artifact; it is not a conformer-generation backbone.
    """

    if not samples:
        raise ValueError("Cannot run single-step training smoke on an empty sample list")
    torch = _load_torch()
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    features, targets, per_sample_slices = _batch_tensors(torch, samples)
    model = torch.nn.Linear(4, 3)
    model.to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()

    optimizer.zero_grad(set_to_none=True)
    prediction_before = model(features)
    loss_before = loss_fn(prediction_before, targets)
    loss_before.backward()
    grad_norm = _grad_norm(model.parameters())
    optimizer.step()

    with torch.no_grad():
        prediction_after = model(features)
        loss_after = loss_fn(prediction_after, targets)
        absolute_error_after = torch.abs(prediction_after - targets)

    return {
        "model_name": "single_linear_velocity_smoke",
        "backend": "torch_cpu",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "forward_pass_count": 2,
        "backward_pass_count": 1,
        "optimizer_step_count": 1,
        "optimizer_name": "SGD",
        "learning_rate": float(learning_rate),
        "sample_count": len(samples),
        "atom_count": int(features.shape[0]),
        "component_count": int(targets.numel()),
        "loss_before_step": float(loss_before.detach().cpu().item()),
        "loss_after_step": float(loss_after.detach().cpu().item()),
        "mean_velocity_mae_after_step": float(absolute_error_after.mean().detach().cpu().item()),
        "max_velocity_abs_error_after_step": float(absolute_error_after.max().detach().cpu().item()),
        "gradient_l2_norm": grad_norm,
        "per_sample": _per_sample_metrics(torch, prediction_after, targets, per_sample_slices),
        "note": "Exactly one CPU forward/loss/backward/optimizer step; not a training loop and not benchmark metrics.",
    }


def run_bounded_multistep_velocity_training_smoke(
    samples: list[dict[str, Any]],
    seed: int,
    max_steps: int = 3,
    learning_rate: float = 1.0e-2,
) -> dict[str, Any]:
    """Run a very short CPU-only optimizer sequence for smoke validation."""

    if not samples:
        raise ValueError("Cannot run multi-step training smoke on an empty sample list")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive for multi-step training smoke")
    if max_steps > 5:
        raise ValueError("multi-step training smoke is capped at five optimizer steps")

    torch = _load_torch()
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    features, targets, per_sample_slices = _batch_tensors(torch, samples)
    model = torch.nn.Linear(4, 3)
    model.to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()
    step_records = []

    for step_index in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = loss_fn(prediction, targets)
        loss.backward()
        grad_norm = _grad_norm(model.parameters())
        optimizer.step()
        step_records.append(
            {
                "step_index": step_index,
                "optimizer_step": step_index + 1,
                "loss_before_step": float(loss.detach().cpu().item()),
                "gradient_l2_norm": grad_norm,
                "learning_rate": float(learning_rate),
            }
        )

    with torch.no_grad():
        prediction_after = model(features)
        loss_after = loss_fn(prediction_after, targets)
        absolute_error_after = torch.abs(prediction_after - targets)

    return {
        "model_name": "single_linear_velocity_smoke",
        "backend": "torch_cpu",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "forward_pass_count": int(max_steps) + 1,
        "backward_pass_count": int(max_steps),
        "optimizer_step_count": int(max_steps),
        "optimizer_name": "SGD",
        "learning_rate": float(learning_rate),
        "sample_count": len(samples),
        "atom_count": int(features.shape[0]),
        "component_count": int(targets.numel()),
        "initial_loss": step_records[0]["loss_before_step"],
        "final_loss": float(loss_after.detach().cpu().item()),
        "mean_velocity_mae_after_final_step": float(absolute_error_after.mean().detach().cpu().item()),
        "max_velocity_abs_error_after_final_step": float(absolute_error_after.max().detach().cpu().item()),
        "step_records": step_records,
        "per_sample": _per_sample_metrics(torch, prediction_after, targets, per_sample_slices),
        "note": "Bounded CPU multi-step smoke metrics only; not a benchmark or larger training run.",
    }


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise TorchUnavailableError(
            "PyTorch is required for the single-step training smoke test. "
            "No single-step training smoke outputs were written."
        ) from exc
    return torch


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


def _grad_norm(parameters: Any) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        squared += float((grad * grad).sum().cpu().item())
    return math.sqrt(squared)


def _per_sample_metrics(
    torch: Any,
    prediction: Any,
    target: Any,
    per_sample_slices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in per_sample_slices:
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
                "velocity_mse_after_step": float(squared_error.mean().detach().cpu().item()),
                "velocity_mae_after_step": float(absolute_error.mean().detach().cpu().item()),
            }
        )
    return rows
