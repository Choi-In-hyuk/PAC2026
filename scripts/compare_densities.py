"""Compare car_door results across 1cm/3cm/5cm sampling densities.

Produces visualizations/density_comparison.png (bar charts) and
visualizations/cardoor_all_coverage.png (coverage per density).
"""
import os, csv, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_coverage import load_motion_targets, load_surface_points, R_TOOL


DENSITIES = [
    ("1cm",      647, "panda_20261008_094525_cardoor.csv"),
    ("adaptive", 252, "panda_20261008_131905_cardooradaptive.csv"),
    ("3cm",      99,  "panda_20261008_105345_cardoor3cm.csv"),
    ("5cm",      42,  "panda_20261008_105345_cardoor5cm.csv"),
]

# explicit motion-file map (we know what ran against which surface)
MOTION_MAP = {
    ("1cm", "tsp_iklink"):      "motions/panda_tsp_iklink_20261008_095152.csv",
    ("1cm", "hrchy_gtsp"):      "motions/panda_hrchy_gtsp_20261008_103422.csv",
    ("adaptive", "tsp_iklink"): "motions/panda_tsp_iklink_20261008_133213.csv",
    ("adaptive", "hrchy_gtsp"): "motions/panda_hrchy_gtsp_20261008_133751.csv",
    ("3cm", "tsp_iklink"):      "motions/panda_tsp_iklink_20261008_105652.csv",
    ("3cm", "hrchy_gtsp"):      "motions/panda_hrchy_gtsp_20261008_110204.csv",
    ("5cm", "tsp_iklink"):      "motions/panda_tsp_iklink_20261008_110309.csv",
    ("5cm", "hrchy_gtsp"):      "motions/panda_hrchy_gtsp_20261008_110817.csv",
}


def motion_for(density, solver_tag):
    return MOTION_MAP.get((density, solver_tag))


# ─── collected hand-coded results from the runs (we saved them in /tmp/*log) ───
RESULTS = {
    ("1cm", "cart"):       {"reconfig": 43, "jmov": 147.43},
    ("1cm", "hrchy"):      {"reconfig": 48, "jmov": 65.80},
    ("adaptive", "cart"):  {"reconfig": 44, "jmov": 63.32},
    ("adaptive", "hrchy"): {"reconfig": 39, "jmov": 32.52},
    ("3cm", "cart"):       {"reconfig": 19, "jmov": 28.81},
    ("3cm", "hrchy"):      {"reconfig": 13, "jmov": 20.15},
    ("5cm", "cart"):       {"reconfig": 10, "jmov": 12.87},
    ("5cm", "hrchy"):      {"reconfig": 8,  "jmov": 11.60},
}


def main():
    pts, _ = load_surface_points()

    # ─── compute coverage per density ───
    cov_rows = []
    for dens, N, surf in DENSITIES:
        for solver, tag in [("Cart-TSP", "tsp_iklink"), ("Hrchy-GTSP", "hrchy_gtsp")]:
            m = motion_for(dens, tag)
            if not m:
                print(f"[warn] no motion for {dens}/{solver}")
                continue
            tool = load_motion_targets(m)
            # cover 647 original surface points (fixed ground truth)
            d = np.linalg.norm(pts[:, None, :] - tool[None, :, :], axis=2)
            hits = (d < R_TOOL)
            visit = hits.sum(axis=1)
            covered = visit >= 1
            cov_rows.append({
                "density": dens, "N_waypoint": N, "solver": solver,
                "coverage": covered.mean() * 100,
                "mean_overlap": visit[covered].mean() if covered.sum() > 0 else 0,
                "max_overlap": int(visit.max()),
                "N_motion": len(tool),
            })
            print(f"[{dens} / {solver}] T={len(tool)}  cov={covered.mean()*100:.1f}%  "
                  f"mean_overlap={visit[covered].mean():.1f}x  max={visit.max()}")

    # ─── plot ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    dens_labels = [f"{d[0]}\n({d[1]}p)" for d in DENSITIES]
    x = np.arange(len(dens_labels))
    w = 0.35

    # panel 1: joint movement
    cart_j = [RESULTS[(d, "cart")]["jmov"] for d in [dd[0] for dd in DENSITIES]]
    hrchy_j = [RESULTS[(d, "hrchy")]["jmov"] for d in [dd[0] for dd in DENSITIES]]
    axes[0, 0].bar(x - w/2, cart_j, w, label="Cart-TSP", color="#888888")
    axes[0, 0].bar(x + w/2, hrchy_j, w, label="Hrchy-GTSP", color="#2ca02c")
    for xi, (c, h) in enumerate(zip(cart_j, hrchy_j)):
        axes[0, 0].text(xi - w/2, c + 2, f"{c:.1f}", ha="center", fontsize=8)
        axes[0, 0].text(xi + w/2, h + 2, f"{h:.1f}", ha="center", fontsize=8)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(dens_labels)
    axes[0, 0].set_ylabel("joint movement (rad)")
    axes[0, 0].set_title("Joint movement vs sampling density")
    axes[0, 0].legend(); axes[0, 0].grid(axis="y", alpha=0.3)

    # panel 2: reconfigurations
    cart_r = [RESULTS[(d, "cart")]["reconfig"] for d in [dd[0] for dd in DENSITIES]]
    hrchy_r = [RESULTS[(d, "hrchy")]["reconfig"] for d in [dd[0] for dd in DENSITIES]]
    axes[0, 1].bar(x - w/2, cart_r, w, label="Cart-TSP", color="#888888")
    axes[0, 1].bar(x + w/2, hrchy_r, w, label="Hrchy-GTSP", color="#2ca02c")
    for xi, (c, h) in enumerate(zip(cart_r, hrchy_r)):
        axes[0, 1].text(xi - w/2, c + 0.5, str(c), ha="center", fontsize=8)
        axes[0, 1].text(xi + w/2, h + 0.5, str(h), ha="center", fontsize=8)
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(dens_labels)
    axes[0, 1].set_ylabel("reconfigurations")
    axes[0, 1].set_title("Reconfigurations vs sampling density")
    axes[0, 1].legend(); axes[0, 1].grid(axis="y", alpha=0.3)

    # panel 3: coverage
    cov_cart = [next(r for r in cov_rows if r["density"] == d and r["solver"] == "Cart-TSP")["coverage"]
                for d in [dd[0] for dd in DENSITIES]]
    cov_hrchy = [next(r for r in cov_rows if r["density"] == d and r["solver"] == "Hrchy-GTSP")["coverage"]
                 for d in [dd[0] for dd in DENSITIES]]
    axes[1, 0].bar(x - w/2, cov_cart, w, label="Cart-TSP", color="#888888")
    axes[1, 0].bar(x + w/2, cov_hrchy, w, label="Hrchy-GTSP", color="#2ca02c")
    for xi, (c, h) in enumerate(zip(cov_cart, cov_hrchy)):
        axes[1, 0].text(xi - w/2, c + 1, f"{c:.1f}%", ha="center", fontsize=8)
        axes[1, 0].text(xi + w/2, h + 1, f"{h:.1f}%", ha="center", fontsize=8)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(dens_labels)
    axes[1, 0].set_ylabel("coverage ratio (%)")
    axes[1, 0].set_title("Coverage of GT surface (R_tool=5cm)")
    axes[1, 0].set_ylim(0, 110)
    axes[1, 0].legend(); axes[1, 0].grid(axis="y", alpha=0.3)

    # panel 4: mean overlap
    ovl_cart = [next(r for r in cov_rows if r["density"] == d and r["solver"] == "Cart-TSP")["mean_overlap"]
                for d in [dd[0] for dd in DENSITIES]]
    ovl_hrchy = [next(r for r in cov_rows if r["density"] == d and r["solver"] == "Hrchy-GTSP")["mean_overlap"]
                 for d in [dd[0] for dd in DENSITIES]]
    axes[1, 1].bar(x - w/2, ovl_cart, w, label="Cart-TSP", color="#888888")
    axes[1, 1].bar(x + w/2, ovl_hrchy, w, label="Hrchy-GTSP", color="#2ca02c")
    for xi, (c, h) in enumerate(zip(ovl_cart, ovl_hrchy)):
        axes[1, 1].text(xi - w/2, c + 0.3, f"{c:.1f}x", ha="center", fontsize=8)
        axes[1, 1].text(xi + w/2, h + 0.3, f"{h:.1f}x", ha="center", fontsize=8)
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(dens_labels)
    axes[1, 1].set_ylabel("mean overlap (x)")
    axes[1, 1].set_title("Mean overlap on GT surface (lower = less redundant)")
    axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"car_door: 1cm vs 3cm vs 5cm sampling density  (R_tool={R_TOOL*100:.0f}cm)",
                 fontsize=13)
    plt.tight_layout()
    out = "visualizations/density_comparison.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
