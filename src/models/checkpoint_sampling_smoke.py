"""Tiny CPU checkpoint/reload/sampling smoke helpers."""

from __future__ import annotations

import copy
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
    return _run_checkpoint_sampling(
        samples=samples,
        checkpoint_path=checkpoint_path,
        seed=seed,
        max_steps=max_steps,
        hard_max_steps=5,
        learning_rate=learning_rate,
        sample_step_size=sample_step_size,
        sampling_samples=sampling_samples,
        early_stop=None,
    )


def run_bounded_checkpoint_sampling_pilot(
    samples: list[dict[str, Any]],
    checkpoint_path: str | Path,
    seed: int,
    max_steps: int,
    hard_max_steps: int,
    learning_rate: float = 1.0e-2,
    sample_step_size: float = 0.1,
    sampling_samples: list[dict[str, Any]] | None = None,
    early_stop: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded longer-horizon checkpoint/reload/sampling pilot on CPU."""

    if not samples:
        raise ValueError("Cannot run bounded checkpoint/sampling pilot on an empty sample list")
    if sampling_samples is not None and not sampling_samples:
        raise ValueError("Cannot run bounded checkpoint/sampling pilot with an empty sampling sample list")
    if hard_max_steps <= 1 or hard_max_steps > 200:
        raise ValueError("bounded checkpoint/sampling pilot hard_max_steps must be between 2 and 200")
    if max_steps <= 1 or max_steps > hard_max_steps:
        raise ValueError("bounded checkpoint/sampling pilot max_steps must be between 2 and hard_max_steps")
    return _run_checkpoint_sampling(
        samples=samples,
        checkpoint_path=checkpoint_path,
        seed=seed,
        max_steps=max_steps,
        hard_max_steps=hard_max_steps,
        learning_rate=learning_rate,
        sample_step_size=sample_step_size,
        sampling_samples=sampling_samples,
        early_stop=early_stop,
    )


def run_validation_checkpoint_training(
    samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    checkpoint_path: str | Path,
    latest_checkpoint_path: str | Path,
    seed: int,
    max_steps: int,
    hard_max_steps: int,
    learning_rate: float = 1.0e-2,
    sample_step_size: float = 0.1,
    sampling_samples: list[dict[str, Any]] | None = None,
    early_stop: dict[str, Any] | None = None,
    checkpoint_result_kind: str = "reusable_validation_selected_checkpoint_not_benchmark",
    checkpoint_scope: str = "reusable_validation_selected_checkpoint",
    device: str = "cpu",
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Train, select a reusable checkpoint by validation loss, reload it, and sample once."""

    if not samples:
        raise ValueError("Cannot run validation-selected training on an empty train sample list")
    if not validation_samples:
        raise ValueError("Cannot run validation-selected training on an empty validation sample list")
    if sampling_samples is not None and not sampling_samples:
        raise ValueError("Cannot run validation-selected training with an empty sampling sample list")
    if hard_max_steps <= 200 or hard_max_steps > 5000:
        raise ValueError("formal validation-selected training hard_max_steps must be in (200, 5000]")
    if max_steps <= 200 or max_steps > hard_max_steps:
        raise ValueError("formal validation-selected training max_steps must be in (200, hard_max_steps]")
    if sample_step_size <= 0.0 or sample_step_size > 1.0:
        raise ValueError("sample_step_size must be in (0, 1]")
    early_stop_settings = _normalize_validation_early_stop(early_stop, max_steps=max_steps)
    sampling_samples = samples if sampling_samples is None else sampling_samples

    torch = _load_torch()
    device = _resolve_device(torch, device=device, require_cuda=require_cuda)
    torch.manual_seed(int(seed))
    if device == "cpu":
        torch.set_num_threads(1)

    train_features, train_targets, train_slices = _batch_tensors(torch, samples, device=device)
    validation_features, validation_targets, validation_slices = _batch_tensors(
        torch,
        validation_samples,
        device=device,
    )
    model = torch.nn.Linear(4, 3)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()

    train_steps = []
    best_validation_loss = math.inf
    best_step = 0
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    stale_steps = 0
    stopped_early = False
    stop_reason = "hard_cap_reached"

    for step_index in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        train_prediction = model(train_features)
        train_loss = loss_fn(train_prediction, train_targets)
        train_loss.backward()
        grad_norm = _grad_norm(model.parameters())
        optimizer.step()
        train_loss_value = float(train_loss.detach().cpu().item())
        with torch.no_grad():
            validation_loss = loss_fn(model(validation_features), validation_targets)
            validation_loss_value = float(validation_loss.detach().cpu().item())
        train_steps.append(
            {
                "step_index": step_index,
                "optimizer_step": step_index + 1,
                "train_loss_before_step": train_loss_value,
                "validation_loss_after_step": validation_loss_value,
                "gradient_l2_norm": grad_norm,
                "checkpoint_selected": False,
            }
        )
        if validation_loss_value < best_validation_loss - early_stop_settings["min_delta"]:
            best_validation_loss = validation_loss_value
            best_step = step_index + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            train_steps[-1]["checkpoint_selected"] = True
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= early_stop_settings["patience"]:
            stopped_early = True
            stop_reason = "validation_patience_exhausted"
            break

    executed_steps = len(train_steps)
    latest_checkpoint_path = Path(latest_checkpoint_path)
    best_checkpoint_path = Path(checkpoint_path)
    for path in [latest_checkpoint_path, best_checkpoint_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    latest_payload = {
        "result_kind": checkpoint_result_kind,
        "benchmark_result": False,
        "checkpoint_scope": checkpoint_scope,
        "checkpoint_role": "latest",
        "model_name": "single_linear_velocity_smoke",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "hard_max_steps": int(hard_max_steps),
        "executed_train_steps": int(executed_steps),
        "learning_rate": float(learning_rate),
        "sample_step_size": float(sample_step_size),
        "early_stop": early_stop_settings,
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "checkpoint_selection_metric": "validation_loss_after_step",
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "note": "Reusable formal-training checkpoint; bounded non-benchmark output.",
    }
    best_payload = {
        **latest_payload,
        "checkpoint_role": "best_validation",
        "model_state_dict": best_model_state,
        "optimizer_state_dict": best_optimizer_state,
        "selected_checkpoint_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
    }
    torch.save(latest_payload, latest_checkpoint_path)
    torch.save(best_payload, best_checkpoint_path)

    reloaded_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    reloaded_model = torch.nn.Linear(4, 3)
    reloaded_model.load_state_dict(reloaded_payload["model_state_dict"])
    reloaded_model.to(device)
    reloaded_model.eval()

    with torch.no_grad():
        train_prediction_after_reload = reloaded_model(train_features)
        validation_prediction_after_reload = reloaded_model(validation_features)
        reloaded_train_loss = loss_fn(train_prediction_after_reload, train_targets)
        reloaded_validation_loss = loss_fn(validation_prediction_after_reload, validation_targets)
        sampled = _sample_coordinates(torch, reloaded_model, sampling_samples, sample_step_size, device=device)

    return {
        "model_name": "single_linear_velocity_smoke",
        "backend": "torch_cuda" if device.startswith("cuda") else "torch_cpu",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_required": bool(require_cuda),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "trainable_parameter_count": sum(parameter.numel() for parameter in reloaded_model.parameters()),
        "forward_pass_count": executed_steps * 2 + 2 + len(sampling_samples),
        "backward_pass_count": executed_steps,
        "optimizer_step_count": executed_steps,
        "optimizer_name": "SGD",
        "learning_rate": float(learning_rate),
        "sample_step_size": float(sample_step_size),
        "training_sample_count": len(samples),
        "validation_sample_count": len(validation_samples),
        "sample_count": len(sampling_samples),
        "atom_count": int(train_features.shape[0]),
        "validation_atom_count": int(validation_features.shape[0]),
        "component_count": int(train_targets.numel()),
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_written": True,
        "checkpoint_reloaded": True,
        "checkpoint_reloaded_path": str(best_checkpoint_path),
        "checkpoint_result_kind": reloaded_payload.get("result_kind"),
        "checkpoint_benchmark_result": reloaded_payload.get("benchmark_result"),
        "checkpoint_scope": checkpoint_scope,
        "checkpoint_reusable": True,
        "checkpoint_selected_by_validation": True,
        "checkpoint_selection_metric": "validation_loss_after_step",
        "selected_checkpoint_step": int(best_step),
        "configured_max_steps": int(max_steps),
        "hard_max_steps": int(hard_max_steps),
        "executed_train_steps": executed_steps,
        "early_stop_enabled": True,
        "early_stop_patience": early_stop_settings["patience"],
        "early_stop_min_delta": early_stop_settings["min_delta"],
        "early_stop_monitor": early_stop_settings["monitor"],
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "initial_train_loss": train_steps[0]["train_loss_before_step"],
        "initial_validation_loss": train_steps[0]["validation_loss_after_step"],
        "final_train_loss": train_steps[-1]["train_loss_before_step"],
        "final_validation_loss": train_steps[-1]["validation_loss_after_step"],
        "best_validation_loss": float(best_validation_loss),
        "best_validation_loss_step": int(best_step),
        "reloaded_train_loss": float(reloaded_train_loss.detach().cpu().item()),
        "reloaded_validation_loss": float(reloaded_validation_loss.detach().cpu().item()),
        "reloaded_loss": float(reloaded_validation_loss.detach().cpu().item()),
        "train_steps": train_steps,
        "sampled_molecules": sampled,
        "per_sample_training_loss": _per_sample_metrics(
            torch,
            train_prediction_after_reload,
            train_targets,
            train_slices,
        ),
        "per_sample_validation_loss": _per_sample_metrics(
            torch,
            validation_prediction_after_reload,
            validation_targets,
            validation_slices,
        ),
        "note": "Validation-selected formal-training metrics only; bounded non-benchmark output.",
    }


def _run_checkpoint_sampling(
    samples: list[dict[str, Any]],
    checkpoint_path: str | Path,
    seed: int,
    max_steps: int,
    hard_max_steps: int,
    learning_rate: float,
    sample_step_size: float,
    sampling_samples: list[dict[str, Any]] | None,
    early_stop: dict[str, Any] | None,
) -> dict[str, Any]:
    if sample_step_size <= 0.0 or sample_step_size > 1.0:
        raise ValueError("sample_step_size must be in (0, 1]")
    sampling_samples = samples if sampling_samples is None else sampling_samples
    early_stop_settings = _normalize_early_stop(early_stop, max_steps=max_steps)

    torch = _load_torch()
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    features, targets, per_sample_slices = _batch_tensors(torch, samples)
    model = torch.nn.Linear(4, 3)
    model.to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()
    train_steps = []
    best_loss = math.inf
    best_step = 0
    stale_steps = 0
    stopped_early = False
    stop_reason = "hard_cap_reached"

    for step_index in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = loss_fn(prediction, targets)
        loss.backward()
        grad_norm = _grad_norm(model.parameters())
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        train_steps.append(
            {
                "step_index": step_index,
                "optimizer_step": step_index + 1,
                "loss_before_step": loss_value,
                "gradient_l2_norm": grad_norm,
            }
        )
        if early_stop_settings is None:
            continue
        if loss_value < best_loss - early_stop_settings["min_delta"]:
            best_loss = loss_value
            best_step = step_index + 1
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= early_stop_settings["patience"]:
            stopped_early = True
            stop_reason = "patience_exhausted"
            break

    executed_steps = len(train_steps)
    if early_stop_settings is None:
        best_loss = train_steps[-1]["loss_before_step"]
        best_step = executed_steps
    elif best_step == 0:
        best_loss = train_steps[-1]["loss_before_step"]
        best_step = executed_steps

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
        "hard_max_steps": int(hard_max_steps),
        "executed_train_steps": int(executed_steps),
        "learning_rate": float(learning_rate),
        "sample_step_size": float(sample_step_size),
        "early_stop": early_stop_settings,
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
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
        "forward_pass_count": executed_steps + 1 + len(sampling_samples),
        "backward_pass_count": executed_steps,
        "optimizer_step_count": executed_steps,
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
        "configured_max_steps": int(max_steps),
        "hard_max_steps": int(hard_max_steps),
        "executed_train_steps": executed_steps,
        "early_stop_enabled": early_stop_settings is not None,
        "early_stop_patience": None if early_stop_settings is None else early_stop_settings["patience"],
        "early_stop_min_delta": None if early_stop_settings is None else early_stop_settings["min_delta"],
        "early_stop_monitor": None if early_stop_settings is None else early_stop_settings["monitor"],
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "best_train_loss": float(best_loss),
        "best_train_loss_step": int(best_step),
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


def _normalize_early_stop(early_stop: dict[str, Any] | None, max_steps: int) -> dict[str, Any] | None:
    if early_stop is None:
        return None
    if early_stop.get("enabled") is False:
        return None
    patience = int(early_stop["patience"])
    min_delta = float(early_stop.get("min_delta", 0.0))
    monitor = str(early_stop.get("monitor", "train_loss_before_step"))
    if monitor != "train_loss_before_step":
        raise ValueError("bounded checkpoint/sampling early_stop monitor must be train_loss_before_step")
    if patience < 1 or patience >= max_steps:
        raise ValueError("bounded checkpoint/sampling early_stop patience must be in [1, max_steps)")
    if min_delta < 0.0:
        raise ValueError("bounded checkpoint/sampling early_stop min_delta must be non-negative")
    return {
        "enabled": True,
        "patience": patience,
        "min_delta": min_delta,
        "monitor": monitor,
    }


def _normalize_validation_early_stop(early_stop: dict[str, Any] | None, max_steps: int) -> dict[str, Any]:
    if early_stop is None or early_stop.get("enabled") is not True:
        raise ValueError("formal validation-selected training requires early_stop.enabled=true")
    patience = int(early_stop["patience"])
    min_delta = float(early_stop.get("min_delta", 0.0))
    monitor = str(early_stop.get("monitor", "validation_loss_after_step"))
    if monitor != "validation_loss_after_step":
        raise ValueError("formal validation-selected training early_stop monitor must be validation_loss_after_step")
    if patience < 1 or patience >= max_steps:
        raise ValueError("formal validation-selected training patience must be in [1, max_steps)")
    if min_delta < 0.0:
        raise ValueError("formal validation-selected training min_delta must be non-negative")
    return {
        "enabled": True,
        "patience": patience,
        "min_delta": min_delta,
        "monitor": monitor,
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


def _resolve_device(torch: Any, device: str, require_cuda: bool = False) -> str:
    requested = str(device)
    if requested == "auto":
        requested = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if requested.startswith("cuda") and not bool(torch.cuda.is_available()):
        if require_cuda:
            raise RuntimeError("CUDA was requested for formal full-database training, but torch.cuda is unavailable")
        return "cpu"
    if require_cuda and not requested.startswith("cuda"):
        raise RuntimeError("require_cuda=true requires device='cuda' or device='auto' with CUDA available")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError(f"Unsupported torch device for checkpoint/sampling training: {requested}")
    return requested


def _batch_tensors(torch: Any, samples: list[dict[str, Any]], device: str = "cpu") -> tuple[Any, Any, list[dict[str, Any]]]:
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
        torch.tensor(feature_rows, dtype=torch.float32, device=device),
        torch.tensor(target_rows, dtype=torch.float32, device=device),
        per_sample_slices,
    )


def _sample_coordinates(
    torch: Any,
    model: Any,
    samples: list[dict[str, Any]],
    sample_step_size: float,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    sampled = []
    for sample in samples:
        xt = torch.tensor(
            [[float(value) for value in xyz] for xyz in sample["xt"]],
            dtype=torch.float32,
            device=device,
        )
        t_column = torch.full((xt.shape[0], 1), float(sample["t"]), dtype=torch.float32, device=device)
        prediction = model(torch.cat([xt, t_column], dim=1))
        sampled_coords = xt + float(sample_step_size) * prediction
        target = torch.tensor(
            [[float(value) for value in xyz] for xyz in sample["x1"]],
            dtype=torch.float32,
            device=device,
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
