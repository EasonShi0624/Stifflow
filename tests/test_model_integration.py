"""Small CPU backbone checks; explicitly skipped while ET-Flow cannot import."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

try:
    from stifflow.model import StifFlow
except (ImportError, OSError) as exc:
    pytest.skip(
        f"ET-Flow binary dependency unavailable: {exc}", allow_module_level=True
    )


def test_dense_neighbors_equivariance_and_finite_gradient():
    torch.manual_seed(42)
    model = StifFlow(
        hidden_channels=16,
        num_layers=1,
        num_rbf=8,
        num_heads=4,
        clip_during_norm=True,
        cutoff_upper=10.0,
        lr_scheduler_type=None,
    )
    n = 40  # Deliberately exceeds upstream's 32-neighbor cap.
    rng = np.random.default_rng(42)
    x = torch.tensor(rng.normal(size=(n, 3)), dtype=torch.float32)
    z = torch.full((n,), 6, dtype=torch.long)
    nodes = torch.zeros(n, 10)
    batch = torch.zeros(n, dtype=torch.long)
    bonds = torch.tensor([[i, j] for i in range(n - 1) for j in [i + 1]]).T
    bonds = torch.cat([bonds, bonds.flip(0)], dim=1)
    attrs = torch.ones(bonds.shape[1], 3)
    t = torch.tensor([[0.4]])
    y = model(z, t, x, bonds, attrs, nodes, batch)
    R = torch.tensor(
        Rotation.from_rotvec([0.3, 0.4, -0.7]).as_matrix(), dtype=torch.float32
    )
    perm = torch.randperm(n)
    inverse = torch.argsort(perm)
    transformed = model(
        z[perm], t, x[perm] @ R.T, inverse[bonds], attrs, nodes[perm], batch
    )
    torch.testing.assert_close(transformed, y[perm] @ R.T, atol=2e-4, rtol=2e-4)
    loss = y.square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("method", ["precision", "alternative_a"])
def test_optimizer_scheduler_rng_resume(method):
    import io

    from etflow.models.loss import batchwise_mse_loss
    from torch_geometric.data import Batch

    from stifflow.model import graph_data
    from stifflow.source import prepare_source
    from stifflow.targets import draw_target, path_module

    torch.manual_seed(42)
    graph = {
        "atomic_numbers": [6, 1, 1],
        "bonds": [[0, 1], [1, 0], [0, 2], [2, 0]],
        "node_attr": np.zeros((3, 10)),
        "bond_attr": np.ones((4, 3)),
    }
    source = prepare_source(graph)
    label = path_module(method).prepare_label(source, np.eye(6), 0.592485)
    x1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    batch = Batch.from_data_list([graph_data(graph)])
    options = {
        "method": method,
        "hidden_channels": 16,
        "num_layers": 1,
        "num_rbf": 8,
        "num_heads": 4,
        "clip_during_norm": True,
        "lr_scheduler_type": "CosineAnnealingWarmupRestarts",
        "first_cycle_steps": 10,
        "warmup_steps": 0,
        "max_lr": 1e-4,
    }

    def configured():
        model = StifFlow(**options)
        config = model.configure_optimizers()
        return model, config["optimizer"], config["lr_scheduler"]["scheduler"]

    def step(model, optimizer, scheduler):
        xt, ut, t = draw_target(x1, source, label, method, model.target_rng)
        optimizer.zero_grad()
        pred = model(
            batch.atomic_numbers,
            torch.tensor([[t]]),
            torch.tensor(xt, dtype=torch.float32),
            batch.edge_index,
            batch.edge_attr,
            batch.node_attr,
            batch.batch,
        )
        loss = batchwise_mse_loss(
            pred, torch.tensor(ut, dtype=torch.float32), batch.batch, reduce="sum"
        )
        loss.backward()
        model.on_after_backward()
        model.configure_gradient_clipping(optimizer, None, None)
        optimizer.step()
        scheduler.step()
        model.on_before_zero_grad(optimizer)
        return loss.detach()

    model, optimizer, scheduler = configured()
    step(model, optimizer, scheduler)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    model.on_save_checkpoint(checkpoint)
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    expected_loss = step(model, optimizer, scheduler)
    restored, opt2, sched2 = configured()
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    restored.load_state_dict(saved["state_dict"], strict=True)
    opt2.load_state_dict(saved["optimizer"])
    sched2.load_state_dict(saved["scheduler"])
    restored.on_load_checkpoint(saved)
    actual_loss = step(restored, opt2, sched2)
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-7, rtol=1e-7)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    assert opt2.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_graph_only_sampler_without_label_access():
    from stifflow.model import generate

    torch.manual_seed(42)
    model = StifFlow(
        hidden_channels=16,
        num_layers=1,
        num_rbf=8,
        num_heads=4,
        clip_during_norm=True,
        lr_scheduler_type=None,
    )
    graph = {
        "atomic_numbers": [6, 1, 1],
        "bonds": [[0, 1], [1, 0], [0, 2], [2, 0]],
        "node_attr": np.zeros((3, 10)),
        "bond_attr": np.ones((4, 3)),
        "K": object(),
        "target_positions": object(),
    }
    positions, metrics = generate(
        model, graph, sampler_config={"samples": 1, "steps": 2}
    )
    assert positions.shape == (1, 3, 3) and np.isfinite(positions).all()
    np.testing.assert_allclose(positions.mean(axis=1), 0, atol=1e-6)
    assert metrics["forward_calls"] == metrics["nfe_per_conformer"] == 2


def test_batched_finite_checks_do_not_skip_empty_or_late_bad_tensors():
    from stifflow.model import require_finite

    require_finite([], "gradient")
    require_finite([torch.zeros(2), torch.ones(3)], "gradient")
    for what, value in (("gradient", float("nan")), ("parameter", float("inf"))):
        with pytest.raises(FloatingPointError, match=what):
            require_finite([torch.zeros(2), torch.tensor([value])], what)
