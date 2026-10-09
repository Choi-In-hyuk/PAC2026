"""Coverage-progress GIF: color each working face by visit count over time."""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import io

sys.path.insert(0, "scripts")
from eval_contact_area import (DATA_DIR, R_TOOL, PAD_INDENT, TOOL_OFFSET, HAND_YAW,
                                load_mesh, face_centers_and_areas, load_motion,
                                hand_to_C_transforms)


def main():
    motion_path = sys.argv[1] if len(sys.argv) > 1 else "motions/panda_region_cover_20261008_162216.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "visualizations/coverage_progress.gif"
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 20  # frames = T / stride

    V, F, status = load_mesh()
    centers_W, areas = face_centers_and_areas(V, F)
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW, t_BW = T_B_W[:3, :3], T_B_W[:3, 3]
    centers_B = (R_BW @ centers_W.T).T + t_BW
    # also transform vertices to B
    V_B = (R_BW @ V.T).T + t_BW

    work_mask = (status == 1)

    hand_pos, R_B_hand = load_motion(motion_path)
    T = len(hand_pos)
    print(f"motion: {motion_path}, waypoints: {T}")
    t_B_C, R_B_C = hand_to_C_transforms(hand_pos, R_B_hand)

    # per-waypoint face hits (restricted to feasible)
    feasible_idx = np.where(work_mask)[0]
    centers_feas = centers_B[feasible_idx]

    visit = np.zeros(len(centers_feas), dtype=np.int32)

    # precompute vertices per face for Poly3DCollection
    tris_B = V_B[F]  # (Nf, 3, 3)
    tris_feas = tris_B[feasible_idx]

    # cumulative visits built as we go; we draw snapshots every `stride`
    frames_idx = list(range(0, T, stride)) + [T - 1]

    # colormap: 0=gray, 1=green, 2=yellow, 3+=red
    cmap_cols = [(0.80, 0.80, 0.80), (0.20, 0.75, 0.30), (1.00, 0.85, 0.10),
                 (0.95, 0.55, 0.15), (0.85, 0.20, 0.10)]
    cmap = ListedColormap(cmap_cols)

    def color_for(v):
        return cmap_cols[min(int(v), 4)]

    frames = []
    pos_history_B = []

    cur = 0
    for next_i in frames_idx:
        # advance visits from cur to next_i (inclusive)
        for i in range(cur, next_i + 1):
            delta = centers_feas - t_B_C[i]
            p_C = (R_B_C[i].T @ delta.T).T
            xy2 = p_C[:, 0]**2 + p_C[:, 1]**2
            z = p_C[:, 2]
            hit = (xy2 <= R_TOOL*R_TOOL) & (np.abs(z) <= PAD_INDENT)
            visit[hit] += 1
            pos_history_B.append(hand_pos[i])
        cur = next_i + 1

        # draw
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        face_colors = [color_for(v) for v in visit]
        poly = Poly3DCollection(tris_feas, facecolors=face_colors, edgecolors='none',
                                linewidths=0.0, alpha=0.95)
        ax.add_collection3d(poly)

        # also plot non-feasible faces as thin outlines
        ax.add_collection3d(Poly3DCollection(tris_B[~work_mask],
                                              facecolors=(0.95, 0.95, 0.95),
                                              edgecolors='none', alpha=0.3))

        # trajectory path so far
        if len(pos_history_B) > 1:
            arr = np.array(pos_history_B)
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color='black', lw=0.6, alpha=0.4)
        ax.scatter(hand_pos[next_i, 0], hand_pos[next_i, 1], hand_pos[next_i, 2],
                   color='red', s=30)

        # bounds
        all_pts = np.vstack([V_B[:, :3], hand_pos])
        pad = 0.05
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()

        cov_area = areas[feasible_idx][visit >= 1].sum()
        ov_area  = areas[feasible_idx][visit >= 2].sum()
        total    = areas[feasible_idx].sum()
        cov_pct  = cov_area / total * 100
        ov_pct   = (ov_area / cov_area * 100) if cov_area > 0 else 0
        ax.set_title(f"waypoint {next_i+1}/{T}   cov {cov_pct:5.1f}%   ov/cov {ov_pct:5.1f}%",
                     fontsize=11)

        # save frame to in-memory PNG then to PIL
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=120, loop=0, optimize=True)
    print(f"saved {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
