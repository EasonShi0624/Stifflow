"""Read-only environment/provenance audit. No downloads or package changes."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
REPOS = (
    "ETFlow-reference",
    "stifflow-v1",
    "stifflow-v1-alternative-A",
    "stifflow-v1-benchmark",
)


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def optional_fingerprint(path):
    """Record absent optional baselines without blocking unrelated experiments."""
    path = Path(path).resolve()
    return fingerprint(path) if path.is_file() else {"path": str(path), "exists": False}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def provenance(root=WORKSPACE):
    repos = {}
    for name in REPOS:
        repo = root / name
        untracked = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
        repos[name] = {
            "path": str(repo.resolve()),
            "head": git(repo, "rev-parse", "HEAD").strip(),
            "branch": git(repo, "branch", "--show-current").strip(),
            "status": git(repo, "status", "--short"),
            "diff": git(repo, "diff", "HEAD", "--binary"),
            "untracked": [
                fingerprint(repo / p) for p in untracked if (repo / p).is_file()
            ],
        }
    files = [
        p
        for name in REPOS[1:3]
        for p in (root / name / "formulation").glob("*")
        if p.suffix in (".tex", ".pdf")
    ]
    pairs = {}
    for name in ("qm9-o3", "drugs-o3", "drugs-so3"):
        pairs[name] = {
            "config": optional_fingerprint(
                root / "ETFlow-reference/configs" / f"{name}.yaml"
            ),
            "checkpoint": optional_fingerprint(
                root / "dataset/ET-Flow_zenodo" / f"{name}.ckpt"
            ),
        }
    return {
        "repos": repos,
        "formulations": [fingerprint(p) for p in sorted(files)],
        "official_pairs": pairs,
    }


def probe(code, timeout=120):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"import/probe exceeded {timeout} seconds"}


def audit(gpu=False, warmup=False):
    report = {
        "python": sys.executable,
        "prefix": sys.prefix,
        "version": sys.version,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "modules": {},
        "conda_packages": [],
    }
    for name in (
        "torch",
        "torch_cluster",
        "torch_geometric",
        "etflow",
        "numpy",
        "scipy",
        "rdkit",
        "lightning",
    ):
        spec = importlib.util.find_spec(name)
        report["modules"][name] = None if spec is None else spec.origin
    for path in sorted(Path(sys.prefix, "conda-meta").glob("*.json")):
        item = json.loads(path.read_text())
        report["conda_packages"].append(
            {k: item.get(k) for k in ("name", "version", "build", "channel")}
        )
    if warmup:
        report["warmup"] = probe(
            "import torch, etflow; print(torch.__version__, etflow.__file__)",
            timeout=300,
        )
    report["probes"] = {
        "imports": probe(
            "import torch, torch_cluster, etflow; print(torch.__version__, torch.version.cuda); print(etflow.__file__)"
        ),
        "radius_graph_cpu": probe(
            "import torch; from torch_cluster import radius_graph; x=torch.tensor([[0.,0.,0.],[0.5,0.,0.]]); e=radius_graph(x,r=1.); assert e.shape == (2,2); print(e)"
        ),
    }
    if gpu:
        if not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError("--gpu is only allowed inside an allocated Slurm job")
        report["probes"]["gpu"] = probe(
            'import torch; from torch_cluster import radius_graph; x=torch.tensor([[0.,0.,0.],[0.5,0.,0.]],device="cuda"); assert radius_graph(x,r=1.).shape == (2,2); assert torch.isfinite(x @ x.T).all(); torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
        )
    expected = WORKSPACE / "ETFlow-reference/etflow/__init__.py"
    actual = report["modules"]["etflow"]
    report["etflow_path_matches"] = (
        actual is not None and Path(actual).resolve() == expected.resolve()
    )
    report["ok"] = (
        report["etflow_path_matches"]
        and all(p["ok"] for p in report["probes"].values())
        and (not warmup or report["warmup"]["ok"])
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    # Exclusive creation protects previous audit records.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        report = audit(args.gpu, args.warmup)
        report["provenance"] = provenance()
        json.dump(report, stream, indent=2)
    print(f"preflight ok={report['ok']}: {args.output}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
