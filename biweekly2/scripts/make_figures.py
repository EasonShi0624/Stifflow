"""Biweekly Report 2 figures.

Figure 1 (mathematical verification + mechanism), Figure 2 (generation
performance + physical quality).  Vector PDF + 300 dpi PNG, sized for a
3-page report (full text width ~ 6.5 in).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
BW = WS / "runs/biweekly2-20261019"
FIG = BW / "figures"
METHOD_COLORS = {
    "alternative_a": "#2e86ab",
    "control": "#7a8b99",
}
METHOD_LABELS = {
    "alternative_a": "StifFlow",
    "control": "ET-Flow-backbone control",
}


def load(name):
    return json.loads((BW / f"results/{name}.json").read_text())


def figure1(data):
    gaussian = data["task2a_gaussian"]
    equiv = data["task2b_equivariance"]
    signal = data["task4a_signal"]["path_differences"]
    paths = data["p0_paths"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))

    # (a) finite-sample convergence of empirical covariance to analytic Sigma
    ax = axes[0]
    draws = np.array([256, 1024, 4096])
    times_plot = [0.1, 0.5, 0.9]
    for t, color, marker in zip(times_plot, ["#2e86ab", "#f6a516", "#7a8b99"], ["o", "s", "^"]):
        ys, lo, hi = [], [], []
        for m in draws:
            vals = [
                gaussian["molecules_data"][i]["times"][str(t)][f"M{m}"][
                    "covariance_relative_error"
                ]
                for i in gaussian["molecules_data"]
            ]
            ys.append(np.median(vals))
            lo.append(np.percentile(vals, 10))
            hi.append(np.percentile(vals, 90))
        ax.plot(draws, ys, marker=marker, color=color, label=f"t = {t}")
        ax.fill_between(draws, lo, hi, alpha=0.15, color=color)
    ax.plot(draws, 1 / np.sqrt(draws), "k--", lw=1, label=r"$1/\sqrt{M}$ guide")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(draws)
    ax.set_xticklabels(["256", "1024", "4096"])
    ax.minorticks_off()
    ax.set_xlabel("draws $M$ per molecule")
    ax.set_ylabel(r"rel. error of empirical $\mathrm{Cov}[X_t]$")
    ax.set_title("(a) Conditional-Gaussian check", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.tick_params(labelsize=7)

    # (b) equivariance + sampling consistency of the trained A model
    ax = axes[1]
    methods_plot = ["alternative_a", "control"]
    rot_max = [
        max(
            e
            for row in equiv["methods"][m]["equivariance"]
            for e in row["rotation_equivariance_relative_error"]
        )
        for m in methods_plot
    ]
    perm_max = [
        max(
            e
            for row in equiv["methods"][m]["equivariance"]
            for e in row["permutation_equivariance_relative_error"]
        )
        for m in methods_plot
    ]
    cons_rot = [
        max(
            row["rotation_endpoint_rmsd_angstrom"]
            for row in equiv["methods"][m]["sampling_consistency"]
        )
        for m in methods_plot
    ]
    cons_perm = [
        max(
            row["permutation_endpoint_rmsd_angstrom"]
            for row in equiv["methods"][m]["sampling_consistency"]
        )
        for m in methods_plot
    ]
    x = np.arange(2)
    w = 0.5
    colors = [METHOD_COLORS[m] for m in methods_plot]
    ax.bar(x - w / 4, rot_max, w / 2, color=colors, label="rotation")
    ax.bar(
        x + w / 4,
        perm_max,
        w / 2,
        color=colors,
        alpha=0.5,
        hatch="//",
        label="permutation",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["StifFlow", "control"], fontsize=8)
    ax.set_ylabel("max velocity rel. error", fontsize=8)
    ax.set_title("(b) Equivariance, trained ckpts", fontsize=9)
    ax.axhline(2e-6, color="k", ls=":", lw=0.8, label="float32 noise")
    note = (
        "Euler-50 sampling from transformed noise\n"
        "max endpoint RMSD:  rot "
        + f"{max(cons_rot):.1e}"
        + " Å,  perm "
        + f"{max(cons_perm):.1e}"
        + " Å"
    )
    ax.set_xlabel(note, fontsize=6)
    ax.set_ylim(3e-7, 3e-5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    # (c) stiffness signal actually carried by A's supervision vs control
    ax = axes[2]
    times = np.array(signal["times"])
    vel_med = np.array(signal["velocity_relative_difference_median"], dtype=float)
    vel_p90 = np.array(signal["velocity_relative_difference_p90"], dtype=float)
    mask = (times > 0) & (times < 1)
    ax.plot(
        times[mask],
        vel_med[mask],
        color=METHOD_COLORS["alternative_a"],
        lw=1.5,
        label="median",
    )
    ax.plot(
        times[mask],
        vel_p90[mask],
        color=METHOD_COLORS["alternative_a"],
        lw=1,
        ls="--",
        alpha=0.7,
        label="90th pct",
    )
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\|u_S-u_c\|/\|u_c\|$, same $z$")
    ax.set_title("(c) Stiffness signal in StifFlow supervision", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7)

    # (d) target velocity energy along the path (128 train rows, shared z)
    ax = axes[3]
    legend_handles = []
    for method in ("alternative_a", "control"):
        p = paths[method]
        (line,) = ax.plot(
            p["times"],
            p["target_energy"],
            color=METHOD_COLORS[method],
            lw=1.4,
            label=METHOD_LABELS[method].split(" (")[0],
        )
        legend_handles.append(line)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"mean $\|U_t\|^2$")
    ax.set_yscale("log")
    ax.set_title("(d) Target velocity scale", fontsize=9)
    ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=1.8, rect=(0, 0.075, 1, 1))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.savefig(FIG / "figure1-verification-mechanism.pdf")
    fig.savefig(FIG / "figure1-verification-mechanism.png", dpi=300)
    plt.close(fig)


def figure2(data):
    test = data["test_metrics"]
    curve = data["learning_curve"]
    physical = data["task3_physical_quality"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))

    # (a) frozen test protocol: MAT-R and chirality with seed std
    ax = axes[0]
    methods = ["alternative_a", "control"]
    x = np.arange(2)
    w = 0.38
    matr = [test[m]["mean"]["mat_r"] for m in methods]
    matr_err = [test[m]["seed_std"]["mat_r"] for m in methods]
    chir = [test[m]["per_seed"]["42"]["chirality_macro"] for m in methods]
    chir_err = [
        np.std([test[m]["per_seed"][str(s)]["chirality_macro"] for s in (42, 43, 44)], ddof=1)
        for m in methods
    ]
    ax.bar(x - w / 2, matr, w, yerr=matr_err, color=[METHOD_COLORS[m] for m in methods])
    ax.set_xticks(x)
    ax.set_xticklabels(["StifFlow", "control"], fontsize=8)
    ax.set_ylabel("MAT-R (Å)", fontsize=8)
    ax.set_ylim(0, 0.20)
    ax.set_title("(a) Performance on Test set", fontsize=9)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, chir, w, yerr=chir_err, color=[METHOD_COLORS[m] for m in methods], alpha=0.45, hatch="//")
    ax2.set_ylabel("chirality fraction", fontsize=8, color="#666")
    ax2.set_ylim(0, 1.15)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)

    # (b) learning curve on the monitor subset
    ax = axes[1]
    for method in ("alternative_a", "control"):
        rows = sorted(curve[method], key=lambda r: r["step"])
        ax.plot(
            [r["step"] for r in rows],
            [r["mat_r"] for r in rows],
            marker="o",
            ms=3,
            lw=1.4,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method].split(" (")[0],
        )
    ax.set_xticks([25000, 100000, 250000])
    ax.set_xticklabels(["25k", "100k", "250k"])
    ax.minorticks_off()
    ax.set_xlabel("training updates")
    ax.set_ylabel("MAT-R (Å), 512-mol monitor")
    ax.set_title("(b) Validation learning curves", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.tick_params(labelsize=7)

    # (c) MMFF94 relaxable strain energy distributions
    ax = axes[2]
    strains = {}
    for method in ("alternative_a", "control"):
        strains[method] = np.array(
            [
                r["relaxable_strain_kcal_mol"]
                for r in physical["methods"][method]["per_conformer"]
                if r.get("status") == "ok"
            ]
        )
    strains["references"] = np.array(
        [
            r["relaxable_strain_kcal_mol"]
            for r in physical["references"]["per_conformer"]
            if r.get("status") == "ok"
        ]
    )
    order = ("references", "control", "alternative_a")
    parts = ax.violinplot(
        [np.clip(strains[k], 0, 40) for k in order],
        positions=range(3),
        showmedians=True,
        widths=0.8,
    )
    for pc, k in zip(parts["bodies"], order):
        color = METHOD_COLORS.get(k, "#b8b8b8")
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("k")
    med = {k: float(np.median(v)) for k, v in strains.items()}
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        [
            f"refs\n{med['references']:.2f}",
            f"control\n{med['control']:.2f}",
            f"StifFlow\n{med['alternative_a']:.2f}",
        ],
        fontsize=8,
    )
    ax.set_xlabel("median (kcal/mol)", fontsize=7, labelpad=2)
    ax.set_ylabel("MMFF94 strain\n(kcal/mol)", fontsize=8)
    ax.set_title("(c) Physical quality", fontsize=9, pad=6)
    ax.set_ylim(0, 44)
    ax.tick_params(labelsize=7)

    # (d) bond-length deviation vs chirality fraction
    ax = axes[3]
    for method in ("alternative_a", "control"):
        bond = np.array(
            [
                r["bond_deviation_angstrom"]
                for r in physical["methods"][method]["per_conformer"]
                if r.get("status") == "ok" and "bond_deviation_angstrom" in r
            ]
        )
        chir_m = test[method]["per_seed"]["42"]["chirality_macro"]
        chir_s = np.std(
            [test[method]["per_seed"][str(s)]["chirality_macro"] for s in (42, 43, 44)],
            ddof=1,
        )
        ax.errorbar(
            np.mean(bond),
            chir_m,
            xerr=2 * np.std(bond) / np.sqrt(len(bond)),
            yerr=chir_s,
            marker="o",
            ms=6,
            color=METHOD_COLORS[method],
            capsize=2,
        )
        offset = {
            "alternative_a": (8, 3),
            "control": (-5, -12),
        }[method]
        ax.annotate(
            "StifFlow" if method == "alternative_a" else "control",
            (np.mean(bond), chir_m),
            textcoords="offset points",
            xytext=offset,
            fontsize=6.5,
            color=METHOD_COLORS[method],
        )
    ax.set_xlabel("heavy bond-length dev. (Å)")
    ax.set_ylabel("chirality fraction (test)")
    ax.set_xscale("log")
    ax.set_xlim(3.0e-3, 7.0e-3)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks([4e-3, 5e-3, 6e-3])
    ax.set_xticklabels(["0.004", "0.005", "0.006"])
    ax.minorticks_off()
    ax.set_title("(d) Local geometry vs chirality", fontsize=9)
    ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "figure2-performance-quality.pdf")
    fig.savefig(FIG / "figure2-performance-quality.png", dpi=300)
    plt.close(fig)


def main():
    data = json.loads((BW / "results/figure-data.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    figure1(data)
    figure2(data)
    print(json.dumps({"figures": [str(p) for p in sorted(FIG.iterdir())]}))


if __name__ == "__main__":
    main()
