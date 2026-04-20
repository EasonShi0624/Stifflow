"""Tiny CPU checkpoint/reload/sampling smoke helpers."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from src.models.single_step_smoke import TorchUnavailableError


def run_checkpoint_sampling_smoke(
    samples: list[dict[str, Any]],
    checkpoint_path: str | Path,
    seed: int,
    max_steps: int = 3,
    learning_rate: float = 1.0e-2,
    sample_step_size: float = 0.1,
    sampling_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train briefly, save/load a disposable checkpoint, and sample once."""

    if not samples:
        raise ValueError("Cannot run checkpoint/sampling smoke on an empty sample list")
    if sampling_samples is not None and not sampling_samples:
        raise ValueError("Cannot run checkpoint/sampling smoke with an empty sampling sample list")
    if max_steps <= 0 or max_steps > 5:
        raise ValueError("checkpoint/sampling smoke max_steps must be between 1 and 5")
    if sample_step_size <= 0.0 or sample_step_size > 1.0:
        raise ValueError("sample_step_size must be in (0, 1]")
    sampling_samples = samples if sampling_samples is None else sampling_samples

    torch = _load_torch()
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    features, targets, per_sample_slices = _batch_tensors(torch, samples)
    model = torch.nn.Linear(4, 3)
    model.to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()
    train_steps = []

    for step_index in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = loss_fn(prediction, targets)
        loss.backward()
        grad_norm = _grad_norm(model.parameters())
        optimizer.step()
        train_steps.append(
            {
                "step_index": step_index,
                "optimizer_step": step_index + 1,
                "loss_before_step": float(loss.detach().cpu().item()),
                "gradient_l2_norm": grad_norm,
            }
        )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "result_kind": "disposable_tiny_real_checkpoint_sampling_smoke_checkpoint",
        "benchmark_result": False,
        "model_name": "single_linear_velocity_smoke",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "learning_rate": float(learning_rate),
        "sample_step_size": float(sample_step_size),
        "note": "Disposable smoke checkpoint only; not a reusable research checkpoint.",
    }
    torch.save(checkpoint_payload, checkpoint_path)

    reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded_model = torch.nn.Linear(4, 3)
    reloaded_model.load_state_dict(reloaded_payload["model_state_dict"])
    reloaded_model.to("cpu")
    reloaded_model.eval()

    with torch.no_grad():
        prediction_after_reload = reloaded_model(features)
        reloaded_loss = loss_fn(prediction_after_reload, targets)
        sampled = _sample_coordinates(torch, reloaded_model, sampling_samples, sample_step_size)

    return {
        "model_name": "single_linear_velocity_smoke",
        "backend": "torch_cpu",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in reloaded_model.parameters()),
        "forward_pass_count": int(max_steps) + 1 + len(sampling_samples),
        "backward_pass_count": int(max_steps),
        "optimizer_step_count": int(max_steps),
        "optimizer_name": "SGD",
        "learning_rate": float(learning_rate),
        "sample_step_size": float(sample_step_size),
        "training_sample_count": len(samples),
        "sample_count": len(sampling_samples),
        "atom_count": int(features.shape[0]),
        "component_count": int(targets.numel()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_written": True,
        "checkpoint_reloaded": True,
        "checkpoint_result_kind": reloaded_payload.get("result_kind"),
        "checkpoint_benchmark_result": reloaded_payload.get("benchmark_result"),
        "initial_loss": train_steps[0]["loss_before_step"],
        "reloaded_loss": float(reloaded_loss.detach().cpu().item()),
        "train_steps": train_steps,
        "sampled_molecules": sampled,
        "per_sample_training_loss": _per_sample_metrics(torch, prediction_after_reload, targets, per_sample_slices),
        "note": "Tiny checkpoint/reload/sampling smoke metrics only; not benchmark metrics.",
    }


def warm_up_checkpoint_sampling_runtime() -> dict[str, Any]:
    """Exercise the CPU backend before timing comparable bounded pilot runs."""

    torch = _load_torch()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    features = torch.zeros((2, 4), dtype=torch.float32, device="cpu")
    targets = torch.zeros((2, 3), dtype=torch.float32, device="cpu")
    model = torch.nn.Linear(4, 3)
    model.to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    loss_fn = torch.nn.MSELoss()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(features), targets)
    loss.backward()
    optimizer.step()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "checkpoint_sampling_warmup.pt"
        torch.save(
            {
                "result_kind": "disposable_checkpoint_sampling_runtime_warmup",
                "benchmark_result": False,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = torch.nn.Linear(4, 3)
    reloaded.load_state_dict(payload["model_state_dict"])
    reloaded.to("cpu")
    reloaded.eval()
    with torch.no_grad():
        _ = reloaded(features)
    return {
        "backend": "torch_cpu",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise TorchUnavailableError(
            "PyTorch is required for the checkpoint/sampling smoke test. "
            "No checkpoint/sampling smoke outputs were written."
        ) from exc
    return torch


def _batch_tensors(torch: Any, samples: list[dict[str, Any]]) -> tuple[Any, Any, list[dict[str, Any]]]:
    feature_rows = []
    target_rows = []
    per_sample_slices = []
    cursor = 0
    for sample in samples:
        start = cursor
        for atom_xt, atom_velocity in zip(sample["xt"], sample["target_velocity"]):
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


def _sample_coordinates(
    torch: Any,
    model: Any,
    samples: list[dict[str, Any]],
    sample_step_size: float,
) -> list[dict[str, Any]]:
    sampled = []
    for sample in samples:
        xt = torch.tensor(
            [[float(value) for value in xyz] for xyz in sample["xt"]],
            dtype=torch.float32,
            device="cpu",
        )
        t_column = torch.full((xt.shape[0], 1), float(sample["t"]), dtype=torch.float32, device="cpu")
        prediction = model(torch.cat([xt, t_column], dim=1))
        sampled_coords = xt + float(sample_step_size) * prediction
        target = torch.tensor(
            [[float(value) for value in xyz] for xyz in sample["x1"]],
            dtype=torch.float32,
            device="cpu",
        )
        rmsd_to_target = _torch_rmsd(torch, sampled_coords, target)
        sampled.append(
            {
                "molecule_id": sample["molecule_id"],
                "split": sample.get("split"),
                "atom_count": int(sampled_coords.shape[0]),
                "sample_step_count": 1,
                "sample_step_size": float(sample_step_size),
                "sampled_rmsd_to_target": rmsd_to_target,
                "predicted_velocity_norm": float(torch.sqrt((prediction * prediction).sum()).detach().cpu().item()),
            }
        )
    return sampled


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
        rows.append(
            {
                "molecule_id": item["molecule_id"],
                "split": item["split"],
                "velocity_mse_after_reload": float(squared_error.mean().detach().cpu().item()),
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


def _torch_rmsd(torch: Any, coords_a: Any, coords_b: Any) -> float:
    if coords_a.shape != coords_b.shape:
        raise ValueError("RMSD requires matching coordinate shapes")
    diff = coords_a - coords_b
    return float(torch.sqrt((diff * diff).sum() / coords_a.shape[0]).detach().cpu().item())
