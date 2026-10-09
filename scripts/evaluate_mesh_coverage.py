"""Mesh-area-based coverage evaluation for car_door motion CSVs.

For each mesh face in the working area, check whether any trajectory waypoint's
tool pad covers it. Aggregate by face area (m²), not point count.

Usage:
  python scripts/evaluate_mesh_coverage.py
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_coverage import load_motion_targets, R_TOOL


DATA_DIR = "meshes/car_door"
OUT_DIR = "visualizations"

MOTION_MAP = {
    ("1cm", "Cart-TSP"):  "motions/panda_tsp_iklink_20261008_095152.csv",
    ("1cm", "Hrchy-GTSP"): "motions/panda_hrchy_gtsp_20261008_103422.csv",
    ("3cm", "Cart-TSP"):  "motions/panda_tsp_iklink_20261008_105652.csv",
    ("3cm", "Hrchy-GTSP"): "motions/panda_hrchy_gtsp_20261008_110204.csv",
    ("5cm", "Cart-TSP"):  "motions/panda_tsp_iklink_20261008_110309.csv",
    ("5cm", "Hrchy-GTSP"): "motions/panda_hrchy_gtsp_20261008_110817.csv",
}


def load_mesh():
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    V = M["vertices"]              # (4637, 3)
    F = M["faces"]                 # (7811, 3)
    status = M["face_status"]      # (7811,)
    region = M["face_region_ids"]  # (7811,)
    return V, F, status, region


def face_centers_and_areas(V, F):
    v0 = V[F[:, 0]]; v1 = V[F[:, 1]]; v2 = V[F[:, 2]]
    centers = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
    return centers, areas, normals


def coverage_by_area(tool_pos_B, centers_B, areas, R_tool):
    """For each face center, count visits by trajectory within R_tool.
    Returns per-face visit_count (int array).
    """
    # chunk to control memory
    N = len(centers_B)
    T = len(tool_pos_B)
    chunk = 1024
    visit = np.zeros(N, dtype=np.int32)
    for i in range(0, N, chunk):
        c = centers_B[i:i + chunk]        # (c, 3)
        d = np.linalg.norm(c[:, None, :] - tool_pos_B[None, :, :], axis=2)
        visit[i:i + chunk] = (d < R_tool).sum(axis=1)
    return visit


def main():
    # mesh in W (world) frame
    V_W, F, status, region = load_mesh()
    centers_W, areas, normals_W = face_centers_and_areas(V_W, F)

    # transform mesh to robot base frame
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW = T_B_W[:3, :3]; t_BW = T_B_W[:3, 3]
    centers_B = (R_BW @ centers_W.T).T + t_BW

    # working-face mask: status == 1 (FEASIBLE)
    work_mask = (status == 1)
    print(f"Mesh faces: total {len(F)}, working (status=1): {work_mask.sum()}")
    print(f"  Total area:    {areas.sum() * 1e4:.2f} cm²")
    print(f"  Working area:  {areas[work_mask].sum() * 1e4:.2f} cm²")

    work_centers = centers_B[work_mask]
    work_areas = areas[work_mask]
    total_work_area = work_areas.sum()

    results = []
    for (dens, solver), path in MOTION_MAP.items():
        if not os.path.exists(path):
            continue
        tool = load_motion_targets(path)
        visit = coverage_by_area(tool, work_centers, work_areas, R_TOOL)
        covered_mask = visit >= 1
        covered_area = work_areas[covered_mask].sum()
        cov_ratio = covered_area / total_work_area

        # overlap on covered faces (area-weighted mean)
        if covered_mask.sum() > 0:
            mean_overlap = (visit[covered_mask] * work_areas[covered_mask]).sum() / covered_area
            max_overlap = int(visit.max())
        else:
            mean_overlap = max_overlap = 0

        results.append({
            "density": dens, "solver": solver,
            "T": len(tool),
            "coverage_pct": cov_ratio * 100,
            "covered_cm2": covered_area * 1e4,
            "total_cm2": total_work_area * 1e4,
            "mean_overlap": mean_overlap,
            "max_overlap": max_overlap,
            "visit": visit,
        })

        print(f"\n[{dens} / {solver}]  T={len(tool)}  "
              f"coverage={cov_ratio*100:5.1f}%  "
              f"({covered_area*1e4:.1f}/{total_work_area*1e4:.1f} cm²)  "
              f"mean_overlap={mean_overlap:.2f}x  max={max_overlap}")

    # ───── visualize ─────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dens_order = ["1cm", "3cm", "5cm"]
    solvers = ["Cart-TSP", "Hrchy-GTSP"]
    x = np.arange(len(dens_order)); w = 0.35
    colors = {"Cart-TSP": "#888888", "Hrchy-GTSP": "#2ca02c"}

    def get(dens, solver, key):
        for r in results:
            if r["density"] == dens and r["solver"] == solver:
                return r[key]
        return 0

    # panel 1: coverage % by area
    for i, s in enumerate(solvers):
        vals = [get(d, s, "coverage_pct") for d in dens_order]
        axes[0, 0].bar(x + (i - 0.5) * w, vals, w, label=s, color=colors[s])
        for xi, v in zip(x, vals):
            axes[0, 0].text(xi + (i - 0.5) * w, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(dens_order)
    axes[0, 0].set_ylabel("area coverage (%)")
    axes[0, 0].set_title(f"Mesh-area coverage (R_tool={R_TOOL*100:.0f}cm)")
    axes[0, 0].set_ylim(0, 110); axes[0, 0].legend(); axes[0, 0].grid(axis="y", alpha=0.3)

    # panel 2: covered area in cm²
    for i, s in enumerate(solvers):
        vals = [get(d, s, "covered_cm2") for d in dens_order]
        axes[0, 1].bar(x + (i - 0.5) * w, vals, w, label=s, color=colors[s])
        for xi, v in zip(x, vals):
            axes[0, 1].text(xi + (i - 0.5) * w, v + 10, f"{v:.0f}", ha="center", fontsize=9)
    total = results[0]["total_cm2"] if results else 0
    axes[0, 1].axhline(total, color="red", linestyle="--", lw=1, label=f"total work area {total:.0f} cm²")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(dens_order)
    axes[0, 1].set_ylabel("covered area (cm²)")
    axes[0, 1].set_title("Absolute covered area")
    axes[0, 1].legend(); axes[0, 1].grid(axis="y", alpha=0.3)

    # panel 3: mean overlap (area-weighted)
    for i, s in enumerate(solvers):
        vals = [get(d, s, "mean_overlap") for d in dens_order]
        axes[1, 0].bar(x + (i - 0.5) * w, vals, w, label=s, color=colors[s])
        for xi, v in zip(x, vals):
            axes[1, 0].text(xi + (i - 0.5) * w, v + 0.5, f"{v:.1f}x", ha="center", fontsize=9)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(dens_order)
    axes[1, 0].set_ylabel("mean overlap (visits per covered face, area-weighted)")
    axes[1, 0].set_title("Mean overlap on working surface")
    axes[1, 0].legend(); axes[1, 0].grid(axis="y", alpha=0.3)

    # panel 4: visit-count histogram (3cm/Hrchy as example)
    for r, c, lbl in [
        (get("3cm", "Hrchy-GTSP", "visit"), "#2ca02c", "3cm Hrchy"),
        (get("5cm", "Hrchy-GTSP", "visit"), "#ff7f0e", "5cm Hrchy"),
        (get("1cm", "Hrchy-GTSP", "visit"), "#1f77b4", "1cm Hrchy"),
    ]:
        if hasattr(r, "__len__") and len(r) > 0:
            axes[1, 1].hist(r, bins=np.arange(0, max(r.max() + 2, 10)) - 0.5,
                            weights=work_areas * 1e4,  # weight by area
                            alpha=0.5, label=lbl, color=c)
    axes[1, 1].set_xlabel("visit count per face")
    axes[1, 1].set_ylabel("working area (cm²)")
    axes[1, 1].set_title("Visit-count distribution (area-weighted)")
    axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"car_door MESH-AREA coverage  (working area: {total:.0f} cm², R_tool={R_TOOL*100:.0f}cm)",
                 fontsize=13)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "mesh_coverage.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
