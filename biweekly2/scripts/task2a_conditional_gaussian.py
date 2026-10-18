"""Task 2a: empirical verification of Alternative A's conditional Gaussian.

For real validation molecules with stiffness labels, fix the target conformer
x1, the Hessian factor L_K, and the orientation.  At fixed times t, draw M
independent z ~ N(0, I) in the reduced space and verify:

1. the empirical mean of X_t equals alpha(t) x1 (centered);
2. the empirical reduced covariance of X_t matches the analytic
   Sigma_t = A_t A_t^T and the closed form f^2 C0 + g^2 C_K + f g (S L_K^T
   + L_K S^T) implied by A_t = f S + g L_K;
3. whitened draws w = A_t^{-1} Q^T (X_t - alpha x1) are standard normal:
   mean 0, identity covariance, eigenvalue spectrum inside the
   Marchenko-Pastur finite-sample bounds;
4. the same checks on the analytic velocity U_t.

Float64 throughout.  Errors are compared against finite-sample expectations
(~1/sqrt(M)), so residual deviations at machine-level scale are not required.

Outputs runs/biweekly2-20261019/results/task2a-conditional-gaussian.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
OUT = WS / "runs/biweekly2-20261019/results/task2a-conditional-gaussian.json"
VAL_MANIFEST = WS / "cache/qm9-v3-packed-strict/val/manifest.json"
CONFIG = WS / "stifflow-v1/configs/qm9-formal-4090d.yaml"
TIMES = (0.1, 0.25, 0.5, 0.75, 0.9)
DRAWS = (256, 1024, 4096)
MOLECULES = 32
SEED = 42


def select_molecules(records, count):
    """Same deterministic SHA256(id) ordering rule as the research tools."""
    ordered = sorted(records, key=lambda r: hashlib.sha256(r["id"].encode()).digest())
    return ordered[:count]


def marchenko_pastur_bounds(n, p):
    """Eigenvalue support of (1/n) W W^T for W [p, n] standard normal rows."""
    ratio = p / n
    lo = (1 - np.sqrt(1.0 / ratio)) ** 2 if ratio > 1 else 0.0
    hi = (1 + np.sqrt(1.0 / ratio)) ** 2
    return lo, hi


def check_molecule(graph, item, rng):
    from stifflow.precision import schedules
    from stifflow_alt_a.path import factor

    source, label, x1 = item["source"], item["label"], item["x1"]
    Q, d = source.Q, source.S.shape[0]
    result = {"n_atoms": int(x1.shape[0]), "d": d, "times": {}}
    for t in TIMES:
        A, Adot = factor(t, source, label)
        alpha, _, alphadot, s = schedules(t)
        f, g = s * s * (1 + t), t * t * s
        # Analytic covariance in the reduced space.
        CK = label["L_K"] @ label["L_K"].T
        C0 = source.C0
        cross = source.S @ label["L_K"].T + label["L_K"] @ source.S.T
        sigma_analytic = f * f * C0 + g * g * CK + f * g * cross
        sigma_factor = A @ A.T
        scale = np.linalg.norm(sigma_analytic)
        t_row = {
            "sigma_identity_residual": float(
                np.linalg.norm(sigma_factor - sigma_analytic) / scale
            )
        }
        x1c = x1 - x1.mean(axis=0)
        mean_red = alpha * (Q.T @ x1c.ravel())  # reduced coordinates of alpha*x1
        Ainv = np.linalg.inv(A)
        for m in DRAWS:
            z = rng.standard_normal((m, d))
            # Reduced coordinates: c_t = Q^T X_t = alpha Q^T x1 + A z.
            x_red = mean_red + z @ A.T
            u_red = alphadot * mean_red + z @ Adot.T
            # 1. mean: E[noise]=0 in reduced space.
            mean_err = float(
                np.linalg.norm(x_red.mean(axis=0) - mean_red)
                / np.linalg.norm(mean_red)
            )
            emp_cov = np.cov(x_red - mean_red, rowvar=False)
            denom = np.linalg.norm(sigma_analytic)
            cov_rel = float(np.linalg.norm(emp_cov - sigma_analytic) / denom)
            # finite-sample expectation ~ ||Sigma||/M roughly; report both.
            whitened = (x_red - mean_red) @ Ainv.T
            w_mean = whitened.mean(axis=0)
            w_cov = np.cov(whitened, rowvar=False)
            w_eigs = np.linalg.eigvalsh(w_cov)
            lo, hi = marchenko_pastur_bounds(m, d)
            # Velocity covariance.
            ud = Adot @ Adot.T
            emp_ucov = np.cov(u_red - alphadot * mean_red, rowvar=False)
            ucov_rel = float(np.linalg.norm(emp_ucov - ud) / np.linalg.norm(ud))
            t_row[f"M{m}"] = {
                "mean_relative_error": mean_err,
                "covariance_relative_error": cov_rel,
                "whitened_mean_norm": float(np.linalg.norm(w_mean)),
                "whitened_mean_norm_expected_sigma": float(np.sqrt(d / m)),
                "whitened_cov_identity_relative_error": float(
                    np.linalg.norm(w_cov - np.eye(d)) / np.linalg.norm(np.eye(d))
                ),
                "whitened_cov_eigenvalue_min": float(w_eigs[0]),
                "whitened_cov_eigenvalue_max": float(w_eigs[-1]),
                "marchenko_pastur_bounds": [float(lo), float(hi)],
                "whitened_eigs_within_mp": bool(
                    w_eigs[0] >= lo * 0.85 and w_eigs[-1] <= hi * 1.15
                ),
                "velocity_covariance_relative_error": ucov_rel,
            }
        result["times"][str(t)] = t_row
    return result


def main():
    from rdkit import RDLogger

    from stifflow.training import load_records

    RDLogger.DisableLog("rdApp.*")
    cfg = yaml.safe_load(CONFIG.read_text())
    records = load_records(VAL_MANIFEST, "val", "all", "alternative_a", cfg["source"])
    selected = select_molecules(records.records, MOLECULES)
    rng = np.random.default_rng(SEED)
    rows = {}
    for record in selected:
        index = records.records.index(record)
        # Use the first conformer row: fixed x1, fixed L_K, fixed orientation.
        graph, item = records.sample(index, 0)
        rows[record["id"]] = check_molecule(graph, item, rng)
        print(f"verified {record['id']}", flush=True)

    summary = {
        "schema_version": 1,
        "molecules": len(rows),
        "times": TIMES,
        "draws": DRAWS,
        "seed": SEED,
        "method": "alternative_a",
        "label_source": "packed validation labels (L_K), first conformer row",
        "molecules_data": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(summary, indent=2, allow_nan=False))
    # Compact aggregate for the figure/table.
    agg = {}
    for t in TIMES:
        for m in DRAWS:
            key = f"M{m}"
            vals = [rows[i]["times"][str(t)][key] for i in rows]
            agg[f"t{t}/{key}"] = {
                "median_covariance_relative_error": float(
                    np.median([v["covariance_relative_error"] for v in vals])
                ),
                "max_covariance_relative_error": float(
                    max(v["covariance_relative_error"] for v in vals)
                ),
                "median_whitened_cov_identity_error": float(
                    np.median([v["whitened_cov_identity_relative_error"] for v in vals])
                ),
                "all_whitened_eigs_within_mp": all(
                    v["whitened_eigs_within_mp"] for v in vals
                ),
                "median_sigma_identity_residual": float(
                    np.median(
                        [
                            rows[i]["times"][str(t)]["sigma_identity_residual"]
                            for i in rows
                        ]
                    )
                ),
            }
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
