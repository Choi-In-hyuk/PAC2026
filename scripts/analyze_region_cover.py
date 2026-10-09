"""Analysis visualization: where did region_cover fail to cover, and where
is overlap concentrated. Uses the region-sweep evaluation model.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "scripts")
from eval_contact_area import (DATA_DIR, R_TOOL, PAD_INDENT, load_mesh,
                                face_centers_and_areas, load_motion,
                                hand_to_C_transforms)
from eval_region_sweep import (load_source_waypoints, match_region,
                                accumulate_visits)


OUT_DIR = "visualizations"


def main():
    motion_path = sys.argv[1] if len(sys.argv) > 1 else "motions/panda_region_cover_20261008_162216.csv"

    V, F, status = load_mesh()
    centers_W, areas = face_centers_and_areas(V, F)
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW, t_BW = T_B_W[:3, :3], T_B_W[:3, 3]
    centers_B = (R_BW @ centers_W.T).T + t_BW
    V_B = (R_BW @ V.T).T + t_BW
    tris_B = V_B[F]

    feas = (status == 1)
    feas_centers_B = centers_B[feas]
    feas_areas = areas[feas]
    feas_tris = tris_B[feas]

    # evaluate with region-sweep
    hand_pos, R_B_hand = load_motion(motion_path)
    t_B_C, R_B_C = hand_to_C_transforms(hand_pos, R_B_hand)
    src_pos_B, src_region = load_source_waypoints()
    regions_wp = match_region(t_B_C, src_pos_B, src_region)
    visit = accumulate_visits(t_B_C, R_B_C, regions_wp, feas_centers_B)

    total = feas_areas.sum()
    cov_mask = visit >= 1
    cov = feas_areas[cov_mask].sum()
    ov = feas_areas[visit >= 2].sum()
    uncov_area = feas_areas[~cov_mask].sum()
    print(f"motion: {motion_path}")
    print(f"coverage: {cov/total*100:.2f}%, uncovered: {uncov_area*1e4:.2f} cm²")
    print(f"overlap:  {ov/cov*100:.2f}% of covered,  max visit = {int(visit.max())}")

    # bounding box for consistent views
    pad = 0.03
    xr = (V_B[:, 0].min() - pad, V_B[:, 0].max() + pad)
    yr = (V_B[:, 1].min() - pad, V_B[:, 1].max() + pad)
    zr = (V_B[:, 2].min() - pad, V_B[:, 2].max() + pad)

    def setup_ax(ax, title):
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
        ax.set_box_aspect((xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]))
        ax.view_init(elev=22, azim=-60)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12)

    # ---------- Figure 1: uncovered map ----------
    fig = plt.figure(figsize=(18, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # covered = light green, uncovered = red; non-feasible background = light gray
    face_colors_cov = np.where(cov_mask, 1, 0)  # 1 covered, 0 uncovered
    col_covered = (0.70, 0.90, 0.75)
    col_uncovered = (0.90, 0.20, 0.20)
    cols = [col_uncovered if v == 0 else col_covered for v in face_colors_cov]

    ax1.add_collection3d(Poly3DCollection(tris_B[~feas], facecolors=(0.92, 0.92, 0.92),
                                            edgecolors='none', alpha=0.3))
    ax1.add_collection3d(Poly3DCollection(feas_tris, facecolors=cols,
                                            edgecolors='none', alpha=0.95))
    setup_ax(ax1, f"Uncovered map (red = not covered)\n"
                   f"coverage {cov/total*100:.2f}%, uncovered {uncov_area*1e4:.1f} cm²")

    # ---------- Figure 2: overlap heatmap ----------
    # buckets: 0 uncovered (gray), 1 (green), 2 (yellow), 3 (orange), 4+ (red)
    bucket_colors = [(0.92, 0.92, 0.92), (0.30, 0.80, 0.35),
                     (0.98, 0.85, 0.15), (0.98, 0.55, 0.15),
                     (0.90, 0.20, 0.15)]
    def to_bucket(v):
        if v == 0: return 0
        if v == 1: return 1
        if v == 2: return 2
        if v == 3: return 3
        return 4

    cols2 = [bucket_colors[to_bucket(v)] for v in visit]
    ax2.add_collection3d(Poly3DCollection(tris_B[~feas], facecolors=(0.95, 0.95, 0.95),
                                            edgecolors='none', alpha=0.3))
    ax2.add_collection3d(Poly3DCollection(feas_tris, facecolors=cols2,
                                            edgecolors='none', alpha=0.95))
    setup_ax(ax2, f"Visit-count heatmap  (green=1, yellow=2, orange=3, red=4+)\n"
                   f"overlap/covered {ov/cov*100:.2f}%, max visit {int(visit.max())}")

    # legend for ax2
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=bucket_colors[1], label=f"1x ({feas_areas[visit==1].sum()*1e4:.0f} cm²)"),
        mpatches.Patch(color=bucket_colors[2], label=f"2x ({feas_areas[visit==2].sum()*1e4:.0f} cm²)"),
        mpatches.Patch(color=bucket_colors[3], label=f"3x ({feas_areas[visit==3].sum()*1e4:.0f} cm²)"),
        mpatches.Patch(color=bucket_colors[4], label=f"4+x ({feas_areas[visit>=4].sum()*1e4:.0f} cm²)"),
        mpatches.Patch(color=bucket_colors[0], label=f"0x uncovered ({uncov_area*1e4:.0f} cm²)"),
    ]
    ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.02, 0.95), fontsize=8)

    fig.suptitle(f"region_cover analysis — {os.path.basename(motion_path)}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out1 = os.path.join(OUT_DIR, "region_cover_analysis.png")
    plt.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out1}")

    # ---------- Figure 3: separate close-up views (top-down + front) ----------
    fig2 = plt.figure(figsize=(18, 10))
    for i, (elev, azim, name) in enumerate([(90, -90, "top-down"),
                                             (0, -90, "front"),
                                             (0, 0, "side"),
                                             (30, -120, "angled")]):
        ax = fig2.add_subplot(2, 2, i + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(tris_B[~feas], facecolors=(0.95, 0.95, 0.95),
                                              edgecolors='none', alpha=0.3))
        ax.add_collection3d(Poly3DCollection(feas_tris, facecolors=cols2,
                                              edgecolors='none', alpha=0.95))
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
        ax.set_box_aspect((xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(name, fontsize=11)
    fig2.suptitle(f"region_cover visit-count — multiple views", fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "region_cover_multiview.png")
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out2}")

    # ---------- Figure 4: per-region bar chart ----------
    fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 5))

    # per-region coverage + overlap breakdown
    face_region = np.load(os.path.join(DATA_DIR, "mesh.npz"))["face_region_ids"][feas]
    unique_r = np.unique(face_region)

    reg_cov_pct = []; reg_ov_pct = []; reg_names = []; reg_total = []
    for r in unique_r:
        m = face_region == r
        tot = feas_areas[m].sum()
        if tot < 1e-6:
            continue
        covr = feas_areas[m & cov_mask].sum()
        ovr = feas_areas[m & (visit >= 2)].sum()
        reg_cov_pct.append(covr / tot * 100)
        reg_ov_pct.append((ovr / covr * 100) if covr > 0 else 0)
        reg_total.append(tot * 1e4)
        reg_names.append(int(r))

    idx = np.argsort(reg_total)[::-1]
    x = np.arange(len(reg_names))
    ax_a.bar(x, [reg_cov_pct[i] for i in idx], color="#2ca02c")
    ax_a.set_xticks(x); ax_a.set_xticklabels([reg_names[i] for i in idx], rotation=90, fontsize=6)
    ax_a.set_ylabel("coverage % (per region)")
    ax_a.set_title("Per-region coverage")
    ax_a.axhline(100, color='k', ls='--', lw=0.5, alpha=0.5)
    ax_a.grid(axis='y', alpha=0.3)

    ax_b.bar(x, [reg_ov_pct[i] for i in idx], color="#d62728")
    ax_b.set_xticks(x); ax_b.set_xticklabels([reg_names[i] for i in idx], rotation=90, fontsize=6)
    ax_b.set_ylabel("overlap % of covered (per region)")
    ax_b.set_title("Per-region overlap")
    ax_b.grid(axis='y', alpha=0.3)

    fig3.suptitle("Per-region breakdown (regions sorted by area desc)", fontsize=12)
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "region_cover_per_region.png")
    plt.savefig(out3, dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"saved {out3}")

    # print worst regions
    print("\n커버 낮은 영역 top 10:")
    worst_cov_idx = np.argsort(reg_cov_pct)[:10]
    for i in worst_cov_idx:
        print(f"  region {reg_names[i]:4d}: cov {reg_cov_pct[i]:5.1f}%, "
              f"ov {reg_ov_pct[i]:5.1f}%, 면적 {reg_total[i]:6.1f} cm²")

    print("\n중복 심한 영역 top 10:")
    worst_ov_idx = np.argsort(reg_ov_pct)[::-1][:10]
    for i in worst_ov_idx:
        print(f"  region {reg_names[i]:4d}: ov {reg_ov_pct[i]:5.1f}%, "
              f"cov {reg_cov_pct[i]:5.1f}%, 면적 {reg_total[i]:6.1f} cm²")


if __name__ == "__main__":
    main()
