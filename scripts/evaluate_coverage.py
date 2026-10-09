"""Coverage / overlap / gap evaluation for car_door motion CSVs.

For each surface point (ground truth sampling of the car door surface),
count how many trajectory waypoints had the tool within R_tool of it.

Metrics (per motion):
  coverage_ratio  = frac of surface points visited at least once
  mean_overlap    = mean of (visit count across surface points covered)
  max_overlap     = max visits to any single surface point
  overlap_ratio   = frac of covered points visited >= 2 times
  gap_ratio       = 1 - coverage_ratio
  unif_cv         = coefficient of variation of visit counts (lower = more uniform)

Compares all three solvers side by side.

Usage:
  python scripts/evaluate_coverage.py           # process latest cardoor motions
"""
import os, csv, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = "meshes/car_door"
MOTIONS_DIR = "motions"
OUT_DIR = "visualizations"

R_TOOL = 0.05       # tool disc radius (m), from car_door metadata assumption
R_SOFT = 0.005      # thickness tolerance perpendicular to surface (m)


def load_surface_points():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    mask = P["accessibility"] & P["tool_pose_valid"]
    points_W = P["points"][mask]
    normals_W = P["normals"][mask]
    T_B_W = np.array(meta["T_B_W"])
    # transform to robot base frame to match motion CSV
    pts_B = (T_B_W[:3, :3] @ points_W.T).T + T_B_W[:3, 3]
    nrm_B = (T_B_W[:3, :3] @ normals_W.T).T
    return pts_B, nrm_B


def load_motion_targets(path):
    """Return TOOL CONTACT positions in B frame.

    Motion CSV stores panda_hand targets (position + quaternion). The tool pad
    contact point is at offset [0, 0, 0.11] in hand frame along +z (tool axis).
    """
    from scipy.spatial.transform import Rotation as R
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    px, py, pz = (hdr.index(f"target-POS_{c}") for c in "XYZ")
    rx, ry, rz, rw = (hdr.index(f"target-ROT_{c}") for c in "XYZW")

    hand_pos = data[:, [px, py, pz]]
    quat = data[:, [rx, ry, rz, rw]]          # xyzw
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    R_B_hand = R.from_quat(quat).as_matrix()   # (T, 3, 3)

    # tool contact point = hand_origin + R_B_hand @ [0, 0, 0.11]
    offset_hand = np.array([0.0, 0.0, 0.11])
    tool_pos = hand_pos + np.einsum("tij,j->ti", R_B_hand, offset_hand)
    return tool_pos


def evaluate(motion_csv, surface_pts, surface_nrm):
    """Compute coverage metrics."""
    traj = load_motion_targets(motion_csv)
    T = len(traj)
    N = len(surface_pts)

    # (N, T) euclidean distance matrix
    d = np.linalg.norm(surface_pts[:, None, :] - traj[None, :, :], axis=2)
    hits = d < R_TOOL                    # (N, T) boolean
    visit_count = hits.sum(axis=1)        # (N,) visits per surface point

    covered = visit_count >= 1
    cov_ratio = covered.mean()
    if covered.sum() > 0:
        mean_overlap = visit_count[covered].mean()
        max_overlap = int(visit_count.max())
        overlap_ratio = (visit_count[covered] >= 2).mean()
        unif_cv = visit_count[covered].std() / (mean_overlap + 1e-9)
    else:
        mean_overlap = max_overlap = overlap_ratio = unif_cv = 0

    return {
        "name": os.path.basename(motion_csv),
        "T": T,
        "N": N,
        "coverage_ratio": cov_ratio,
        "gap_ratio": 1 - cov_ratio,
        "mean_overlap": mean_overlap,
        "max_overlap": max_overlap,
        "overlap_ratio": overlap_ratio,
        "unif_cv": unif_cv,
        "visit_count": visit_count,
    }


def latest_motion(solver, task="cardoor"):
    files = sorted(glob.glob(f"{MOTIONS_DIR}/panda_{solver}_*.csv"))
    # filter by surface task — motion CSVs don't carry task name, so use any panda_* latest
    # Here we take motions newer than the car_door surface CSV
    surf_files = sorted(glob.glob(f"surfaces/panda_*{task}.csv"))
    if not surf_files:
        return None
    surf_mtime = os.path.getmtime(surf_files[-1])
    # also exclude motions newer than the next surface variant to prevent cross-contamination
    cands = [f for f in files if os.path.getmtime(f) >= surf_mtime]
    return cands[0] if cands else None


def main():
    pts, nrm = load_surface_points()
    print(f"Surface points: {len(pts)}  R_tool={R_TOOL*100}cm")

    results = []
    for solver_key, label in [
        ("tsp_iklink", "Cartesian-TSP"),
        ("hrchy_gtsp", "Hrchy-GTSP (ours)"),
        ("gtsp", "Joint-GTSP"),
    ]:
        m = latest_motion(solver_key)
        if m is None:
            print(f"[{label}] no motion yet")
            continue
        r = evaluate(m, pts, nrm)
        r["label"] = label
        results.append(r)
        print(f"\n[{label}]  file: {r['name']}")
        print(f"  waypoints T = {r['T']}, surface pts N = {r['N']}")
        print(f"  coverage ratio : {r['coverage_ratio']*100:5.1f}%  (gap {r['gap_ratio']*100:.1f}%)")
        print(f"  mean overlap   : {r['mean_overlap']:.2f}x  (max {r['max_overlap']})")
        print(f"  overlap_ratio  : {r['overlap_ratio']*100:.1f}%  (surface pts hit >=2 times)")
        print(f"  uniformity CV  : {r['unif_cv']:.2f}  (lower = more uniform)")

    if not results:
        print("No motions available; skip plot.")
        return

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [r["label"] for r in results]
    colors = ["#888888", "#2ca02c", "#d62728"]

    # 1) coverage + gap
    cov = [r["coverage_ratio"] * 100 for r in results]
    gap = [r["gap_ratio"] * 100 for r in results]
    x = np.arange(len(results))
    axes[0].bar(x, cov, color=colors[: len(results)])
    for xi, c in zip(x, cov):
        axes[0].text(xi, c + 1, f"{c:.1f}%", ha="center", fontsize=10)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("coverage ratio (%)")
    axes[0].set_title("Coverage (higher is better)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # 2) overlap
    axes[1].bar(x, [r["mean_overlap"] for r in results], color=colors[: len(results)])
    for xi, r in zip(x, results):
        axes[1].text(xi, r["mean_overlap"] + 0.05, f"{r['mean_overlap']:.2f}x\n(max {r['max_overlap']})",
                     ha="center", fontsize=9)
    axes[1].set_ylabel("mean overlap (visits per covered pt)")
    axes[1].set_title("Overlap  (closer to 1 = tighter)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # 3) visit count histogram overlay
    for r, c in zip(results, colors):
        axes[2].hist(r["visit_count"], bins=np.arange(r["max_overlap"] + 2) - 0.5,
                     alpha=0.5, label=r["label"], color=c)
    axes[2].set_xlabel("visits per surface point")
    axes[2].set_ylabel("number of surface points")
    axes[2].set_title("Visit count distribution")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle(f"car_door coverage evaluation (R_tool={R_TOOL*100:.0f}cm, {len(pts)} surface pts)",
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "cardoor_coverage.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
