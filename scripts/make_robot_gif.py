"""Robot animation GIF: end-effector + tool disc moving through waypoints."""
import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import io

sys.path.insert(0, "scripts")
from eval_contact_area import (DATA_DIR, R_TOOL, load_mesh, load_motion, hand_to_C_transforms)


def main():
    motion_path = sys.argv[1] if len(sys.argv) > 1 else "motions/panda_region_cover_20261008_162216.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "visualizations/robot_motion.gif"
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    V, F, status = load_mesh()
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW, t_BW = T_B_W[:3, :3], T_B_W[:3, 3]
    V_B = (R_BW @ V.T).T + t_BW
    tris_B = V_B[F]

    hand_pos, R_B_hand = load_motion(motion_path)
    T = len(hand_pos)
    t_B_C, R_B_C = hand_to_C_transforms(hand_pos, R_B_hand)
    print(f"motion: {motion_path}, waypoints: {T}")

    # pre-build tool disc polygon in C frame (circle on z=0 plane, radius R_TOOL)
    N_DISC = 32
    th = np.linspace(0, 2 * np.pi, N_DISC, endpoint=False)
    disc_C = np.stack([R_TOOL * np.cos(th), R_TOOL * np.sin(th), np.zeros_like(th)], axis=1)  # (N,3)

    frames_idx = list(range(0, T, stride)) + [T - 1]
    frames = []

    for next_i in frames_idx:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # car door mesh (feasible = light green, else gray)
        feas = (status == 1)
        ax.add_collection3d(Poly3DCollection(tris_B[feas], facecolors=(0.75, 0.90, 0.75),
                                              edgecolors='none', alpha=0.6))
        ax.add_collection3d(Poly3DCollection(tris_B[~feas], facecolors=(0.90, 0.90, 0.90),
                                              edgecolors='none', alpha=0.4))

        # robot base marker
        ax.scatter(0, 0, 0, color='black', s=60, marker='s')
        ax.text(0, 0, 0.03, 'base', fontsize=8)

        # trajectory so far (hand positions)
        traj = hand_pos[:next_i + 1]
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='steelblue', lw=0.6, alpha=0.5)

        # current hand position
        hp = hand_pos[next_i]
        ax.scatter(hp[0], hp[1], hp[2], color='red', s=50)

        # line from base to hand
        ax.plot([0, hp[0]], [0, hp[1]], [0, hp[2]], color='black', lw=1.2, alpha=0.7)

        # tool: segment from hand to tool contact (length 0.11m along hand +z... already stored in t_B_C)
        tc = t_B_C[next_i]
        ax.plot([hp[0], tc[0]], [hp[1], tc[1]], [hp[2], tc[2]], color='darkorange', lw=2.0)
        ax.scatter(tc[0], tc[1], tc[2], color='darkorange', s=20)

        # tool disc polygon in base frame
        disc_B = (R_B_C[next_i] @ disc_C.T).T + tc
        ax.add_collection3d(Poly3DCollection([disc_B], facecolors=(1.0, 0.6, 0.0),
                                              edgecolors='darkorange', linewidths=0.5, alpha=0.5))

        # bounds based on car door + hand positions
        all_pts = np.vstack([V_B, hand_pos, [[0, 0, 0]]])
        pad = 0.05
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()
        ax.set_title(f"waypoint {next_i+1}/{T}", fontsize=11)

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
