"""P0 solver probes; a refined RK4 result is a numerical reference, not truth."""

import numpy as np


def integrate(field, initial, steps, solver="euler", exact_first=False):
    if not isinstance(steps, int) or steps < 1 or solver not in ("euler", "rk4"):
        raise ValueError("Expected positive steps and euler/rk4")
    if exact_first and solver != "euler":
        raise ValueError("Exact first-step intervention is defined only for Euler")
    x, h, calls, trace = initial.clone(), 1 / steps, 0, []
    for i in range(steps):
        t = i / steps
        k1 = -x if exact_first and i == 0 else field(x, t)
        calls += int(not (exact_first and i == 0))
        if i % max(1, steps // 50) == 0:
            trace.append(
                {
                    "t": t,
                    "coordinate_rms": float(x.square().mean().sqrt()),
                    "velocity_rms": float(k1.square().mean().sqrt()),
                }
            )
        if solver == "euler":
            x = x + h * k1
        else:
            k2 = field(x + 0.5 * h * k1, t + 0.5 * h)
            k3 = field(x + 0.5 * h * k2, t + 0.5 * h)
            k4 = field(x + h * k3, (i + 1) / steps)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            calls += 3
    if not x.isfinite().all():
        raise FloatingPointError("Nonfinite solver trajectory")
    trace.append({"t": 1.0, "coordinate_rms": float(x.square().mean().sqrt())})
    return x, {"forward_calls": calls, "trace": trace}


def diagnose(model, graph, samples, seed=42, reference_steps=200):
    import torch
    from torch_geometric.data import Batch

    from .model import graph_data
    from .source import prepare_source

    if model.method not in ("precision", "alternative_a", "control"):
        raise ValueError("The v*(x,0)=-x diagnostic is restricted to original paths")
    if (
        not isinstance(samples, int)
        or samples < 1
        or not isinstance(reference_steps, int)
        or reference_steps < 100
    ):
        raise ValueError("Invalid diagnostic sample/reference step budget")
    source, data = prepare_source(graph, model.source_config), graph_data(graph)
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    outputs = {
        name: []
        for name in ("euler50", "euler100", "exact_first50", "rk4", "rk4_refined")
    }
    details, endpoint_error = [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, samples, 32):
            size = min(32, samples - offset)
            batch = Batch.from_data_list([data] * size).to(device)
            initial = (
                rng.standard_normal((size, source.S.shape[0])) @ source.S.T @ source.Q.T
            )
            x0 = torch.as_tensor(
                initial.reshape(-1, 3), dtype=torch.float32, device=device
            )

            def field(x, t, batch=batch, size=size):
                return model(
                    z=batch.atomic_numbers,
                    t=x.new_full((size, 1), t),
                    pos=x,
                    bond_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    node_attr=batch.node_attr,
                    batch=batch.batch,
                )

            error = (field(x0, 0.0) + x0).reshape(size, -1).square().sum(dim=1)
            denominator = x0.reshape(size, -1).square().sum(dim=1)
            endpoint_error.extend((error / denominator).cpu().tolist())
            for name, solver, steps, exact in (
                ("euler50", "euler", 50, False),
                ("euler100", "euler", 100, False),
                ("exact_first50", "euler", 50, True),
                ("rk4", "rk4", reference_steps, False),
                ("rk4_refined", "rk4", 2 * reference_steps, False),
            ):
                value, info = integrate(field, x0, steps, solver, exact)
                outputs[name].append(value.reshape(size, -1, 3).cpu().numpy())
                details.append(
                    {"offset": offset, "samples": size, "variant": name, **info}
                )
    arrays = {name: np.concatenate(values) for name, values in outputs.items()}
    reference = arrays["rk4_refined"]
    differences = {
        name: np.sqrt(np.square(values - reference).sum(axis=2).mean(axis=1)).tolist()
        for name, values in arrays.items()
        if name != "rk4_refined"
    }
    return arrays, {
        "initial_field_relative_squared_error": endpoint_error,
        "initial_field_forward_calls": (samples + 31) // 32,
        "endpoint_displacement_rms_per_conformer": differences,
        "reference_converged_at_1e-3_angstrom": bool(max(differences["rk4"]) <= 1e-3),
        "reference_steps": reference_steps,
        "details": details,
        "caveat": "Float32 network and finite-step RK4. Refinement agreement is required; it does not prove model accuracy.",
    }
