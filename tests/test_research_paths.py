"""Small float64 scientific checks, without training or molecular benchmarks."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from stifflow.geometry import centered
from stifflow.precision import factor, prepare_label
from stifflow.research_clock import PathClock, fit_clock, precision_energy
from stifflow.research_paths import (
    DELAYED_METHODS,
    RESEARCH_METHODS,
    label_method,
    relative_rotation,
    sample_target,
)
from stifflow.source import prepare_source
from stifflow.targets import draw_target, precision_target
from stifflow_alt_a.path import factor as a_factor
from stifflow_alt_a.path import prepare_label as a_label


def delayed_kwargs(method):
    return {"stiffness_start": 0.5} if method in DELAYED_METHODS else {}


@pytest.fixture
def example():
    rng = np.random.default_rng(27)
    source = prepare_source(
        {"atomic_numbers": [6, 6, 6, 1], "bonds": [[0, 1], [1, 2], [0, 3]]}
    )
    d = source.S.shape[0]
    M = rng.normal(size=(d, d))
    K = M @ M.T + 0.2 * np.eye(d)
    x1, z = centered(rng.normal(size=(4, 3))), rng.normal(size=d)
    return (
        source,
        a_label(source, K, 0.592485),
        prepare_label(source, K, 0.592485),
        x1,
        z,
    )


@pytest.mark.parametrize("method", RESEARCH_METHODS[:-1])
def test_path_derivative_endpoints_and_centering(example, method):
    source, label, _, x1, z = example
    R = Rotation.from_rotvec([0.2, -0.7, 1.0]).as_matrix()

    def path(t):
        return sample_target(
            x1, t, z, source, label, method, R, **delayed_kwargs(method)
        )

    x0, _ = path(0.0)
    assert_allclose(path(1.0)[0], x1, atol=1e-12)
    raw = (source.Q @ source.S @ z).reshape(-1, 3)
    if method == "aligned_linear":
        assert_allclose(x0 @ x0.T, raw @ raw.T, atol=1e-12)
        assert np.linalg.norm(x1 - x0) <= np.linalg.norm(x1 - raw)
        assert np.linalg.det(x0[:3]) * np.linalg.det(raw[:3]) >= 0
    else:
        assert_allclose(x0, raw, atol=1e-12)
    for t in (0.02, 0.37, 0.9):
        x, u = path(t)
        assert_allclose(u, (path(t + 1e-6)[0] - path(t - 1e-6)[0]) / 2e-6, atol=2e-8)
        assert_allclose(x.mean(axis=0), 0, atol=1e-13)
        assert_allclose(u.mean(axis=0), 0, atol=1e-13)


def test_rotation_null_preserves_entire_noise_energy_not_orientation(example):
    source, label, _, x1, _ = example
    R = Rotation.from_rotvec([0.2, -0.7, 1.0]).as_matrix()
    W = source.Q.T @ np.kron(np.eye(len(x1)), R) @ source.Q
    L = label["L_K"]
    null = {"L_K": W @ L @ W.T}
    assert np.linalg.norm(null["L_K"] - L) > 0.1
    assert_allclose(W @ source.S, source.S @ W, atol=1e-13)
    assert_allclose(
        np.linalg.eigvalsh(null["L_K"] @ null["L_K"].T),
        np.linalg.eigvalsh(L @ L.T),
        atol=1e-12,
    )
    for t in (0.0, 0.1, 0.4, 0.9, 1.0):
        A, Ad = a_factor(t, source, label)
        N, Nd = a_factor(t, source, null)
        assert_allclose(
            [np.linalg.norm(A), np.linalg.norm(Ad)],
            [np.linalg.norm(N), np.linalg.norm(Nd)],
            atol=1e-12,
        )
        z = np.arange(len(L), dtype=float)
        x, u = sample_target(np.zeros_like(x1), t, z, source, label, "a_rot_null", R)
        assert_allclose(x.ravel(), source.Q @ N @ z, atol=1e-12)
        assert_allclose(u.ravel(), source.Q @ Nd @ z, atol=1e-12)
    scale = np.linalg.norm(L) / np.linalg.norm(source.S)
    assert_allclose(
        np.linalg.norm(scale * source.S) ** 2, np.trace(L @ L.T), atol=1e-12
    )


@pytest.mark.parametrize("method", RESEARCH_METHODS)
def test_common_rng_stream_and_replay(example, method):
    source, alt, prec, x1, _ = example
    label = prec if label_method(method) == "precision" else alt
    clock = (
        PathClock(
            {"schema_version": 1, "split": "train", "t": [0.0, 1.0], "q": [1.0, 1.0]}
        )
        if method == "precision_clock"
        else None
    )
    rng, baseline_rng = np.random.default_rng(42), np.random.default_rng(42)
    first = draw_target(x1, source, label, method, rng, clock, **delayed_kwargs(method))
    saved = copy.deepcopy(rng.bit_generator.state)
    relative_rotation(rng)
    assert saved == rng.bit_generator.state
    original = draw_target(x1, source, prec, "precision", baseline_rng)
    assert rng.bit_generator.state == baseline_rng.bit_generator.state
    assert first[2] == original[2]
    if clock:
        assert_allclose(first[0], original[0], atol=1e-13)
        assert_allclose(first[1], original[1], atol=1e-13)
    replay = np.random.default_rng()
    replay.bit_generator.state = saved
    for a, b in zip(
        draw_target(x1, source, label, method, rng, clock, **delayed_kwargs(method)),
        draw_target(x1, source, label, method, replay, clock, **delayed_kwargs(method)),
    ):
        assert_allclose(a, b, atol=0)


def test_clock_chain_rule_and_exact_energy(example):
    source, _, label, x1, z = example
    spec = {
        "schema_version": 1,
        "split": "train",
        "t": [0.0, 0.2, 1.0],
        "q": [0.5, 1.5, 0.5],
    }
    clock = PathClock(spec)
    assert clock.inverse(0.0) == (0.0, 0.5)
    assert_allclose(clock.inverse(1.0), (1.0, 0.5), atol=1e-13)
    for tau in (0.01, 0.13, 0.45, 0.99):
        t, q = clock.inverse(tau)
        u = precision_target(x1, t, z, source, label)[1] / q
        xp = precision_target(x1, clock.inverse(tau + 1e-6)[0], z, source, label)[0]
        xm = precision_target(x1, clock.inverse(tau - 1e-6)[0], z, source, label)[0]
        assert_allclose(u, (xp - xm) / 2e-6, atol=2e-7)
    times = np.linspace(0, 1, 19)
    expected = [
        (6 * t * (1 - t)) ** 2 * np.square(x1).sum()
        + np.square(factor(t, source, label)[1]).sum()
        for t in times
    ]
    assert_allclose(precision_energy(x1, source, label, times), expected, rtol=1e-13)
    for changed in (
        {"split": "val"},
        {"q": [0.5, 1.5, -0.75]},
        {"t": [0.0, 0.0, 1.0]},
        {"q": [1.0, 2.0, 3.0]},
    ):
        with pytest.raises(ValueError):
            PathClock(spec | changed)
    with pytest.raises(ValueError, match="Only precision_clock"):
        draw_target(x1, source, label, "control", np.random.default_rng(42), clock)


def test_clock_fit_is_deterministic_and_reduces_sample_action(example):
    source, _, label, x1, _ = example

    class Records:
        offsets = np.array([0, 2])

        def sample(self, i, j):
            return None, {"x1": x1, "source": source, "label": label, "row_id": j}

    spec = fit_clock(Records(), rows=2)
    assert spec == fit_clock(Records(), rows=2)
    clock = PathClock(spec)
    t, q = clock.t, clock.q
    energy = np.array(spec["mean_target_energy"])
    assert np.trapz(energy / q, t) < np.trapz(energy, t)
    mapped = [clock.inverse(tau)[0] for tau in np.linspace(0, 1, 501)]
    assert np.all(np.diff(mapped) > 0)


def test_new_paths_rotate_and_permute_together(example):
    from stifflow.source import Source

    source, label, _, x1, z = example
    R = Rotation.from_rotvec([0.3, 0.2, -0.5]).as_matrix()
    relative = Rotation.from_rotvec([0.4, -0.1, 0.2]).as_matrix()
    perm = np.array([3, 1, 0, 2])
    W = source.Q.T @ np.kron(np.eye(4)[perm], R) @ source.Q
    transformed = Source(source.Q, W @ source.S @ W.T, W @ source.C0 @ W.T)
    other = {"L_K": W @ label["L_K"] @ W.T}
    for method in RESEARCH_METHODS[:-1]:
        x, u = sample_target(
            x1, 0.73, z, source, label, method, relative, **delayed_kwargs(method)
        )
        xx, uu = sample_target(
            x1[perm] @ R.T,
            0.73,
            W @ z,
            transformed,
            other,
            method,
            R @ relative @ R.T,
            **delayed_kwargs(method),
        )
        assert_allclose(xx, x[perm] @ R.T, atol=1e-12)
        assert_allclose(uu, u[perm] @ R.T, atol=1e-12)


def test_solver_reference_convergence_and_first_step():
    import torch
    from stifflow.research_solver import integrate

    initial = torch.ones((3, 3), dtype=torch.float64)
    field = lambda x, t: -x
    euler, meta = integrate(field, initial, 50)
    intervened, intervention = integrate(field, initial, 50, exact_first=True)
    assert_allclose(euler, intervened, atol=0)
    assert meta["forward_calls"] == 50 and intervention["forward_calls"] == 49
    rk4, meta = integrate(field, initial, 20, "rk4")
    assert_allclose(rk4, initial * np.exp(-1), atol=3e-8, rtol=0)
    assert meta["forward_calls"] == 80
    with pytest.raises(ValueError):
        integrate(field, initial, 20, "rk4", True)


def test_path_statistics_same_rows_and_scale_units(example):
    from stifflow.research_path_statistics import summarize_paths

    source, alt, prec, x1, _ = example

    class Records:
        offsets = np.array([0, 2])

        def sample(self, i, j):
            return None, {
                "x1": x1,
                "source": source,
                "label": alt if self.method == "alternative_a" else prec,
                "row_id": j,
            }

    records = Records()
    reports = []
    for method in ("precision", "alternative_a", "control"):
        records.method = method
        report = summarize_paths(records, method, rows=2)
        assert_allclose(
            np.square(report["coordinate_rms"]) * x1.size,
            report["mean_per_molecule_position_energy"],
            atol=1e-13,
        )
        reports.append(report)
    assert reports[0]["row_ids"] == reports[1]["row_ids"] == reports[2]["row_ids"]
    assert_allclose(
        [r["coordinate_rms"][0] for r in reports], reports[0]["coordinate_rms"][0]
    )


@pytest.mark.parametrize("method", RESEARCH_METHODS)
def test_model_checkpoint_embeds_clock_and_keeps_backbone_identical(method):
    import torch
    from stifflow.model import StifFlow

    options = {
        "hidden_channels": 16,
        "num_layers": 1,
        "num_rbf": 8,
        "num_heads": 4,
        "clip_during_norm": True,
        "lr_scheduler_type": None,
    }
    torch.manual_seed(42)
    baseline = StifFlow(method="control", **options)
    specification = {
        "schema_version": 1,
        "split": "train",
        "t": [0.0, 1.0],
        "q": [1.0, 1.0],
    }
    torch.manual_seed(42)
    model = StifFlow(
        method=method,
        path_clock=specification if method == "precision_clock" else None,
        **delayed_kwargs(method),
        **({"locality_report_sha256": "a" * 64} if method in DELAYED_METHODS else {}),
        **options,
    )
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    reloaded = StifFlow(**dict(model.hparams))
    reloaded.load_state_dict(model.state_dict(), strict=True)
    if method == "precision_clock":
        assert reloaded.path_clock.inverse(0.37) == (0.37, 1.0)
    else:
        assert "path_clock" not in model.hparams and reloaded.path_clock is None
    if method in DELAYED_METHODS:
        assert reloaded.stiffness_start == 0.5
        assert reloaded.hparams["locality_report_sha256"] == "a" * 64
    else:
        assert "stiffness_start" not in model.hparams


def test_solver_probe_toy_field_uses_graph_only_common_noise():
    import torch
    from stifflow.research_solver import diagnose

    class LinearField(torch.nn.Module):
        method = "control"

        def __init__(self):
            super().__init__()
            self.source_config = {"sigma0": 1.0, "rho": 1.0}
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, **kwargs):
            assert set(kwargs) == {
                "z",
                "t",
                "pos",
                "bond_index",
                "edge_attr",
                "node_attr",
                "batch",
            }
            return -kwargs["pos"]

    graph = {
        "atomic_numbers": [6, 1, 1],
        "bonds": [[0, 1], [1, 0], [0, 2], [2, 0]],
        "node_attr": np.zeros((3, 10)),
        "bond_attr": np.ones((4, 3)),
        "K": object(),
        "x1": object(),
    }
    arrays, report = diagnose(LinearField(), graph, 1, reference_steps=100)
    assert report["reference_converged_at_1e-3_angstrom"]
    assert report["initial_field_relative_squared_error"] == [0.0]
    assert_allclose(arrays["euler50"], arrays["exact_first50"], atol=0)
    assert max(report["endpoint_displacement_rms_per_conformer"]["euler100"]) < max(
        report["endpoint_displacement_rms_per_conformer"]["euler50"]
    )
