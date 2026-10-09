"""Convert the preprocessed car_door dataset into arm_coverage surface CSV.

Input:  meshes/car_door/surface_points.npz, metadata.json
Output: surfaces/panda_<ts>_cardoor.csv

Chain:
  tool pose (C frame, W world)  -->  panda_hand pose (W)  -->  panda_hand pose (B base)

Conventions:
  C frame: origin at tool pad contact center, z_C along tool axis toward flange (away from surface).
  T_L_C: panda_link8 (L) -> C,  rotation = diag(1,-1,-1), translation = [0,0,0.11]
  T_L_hand: panda_link8 -> panda_hand,  Rz(-45 deg), translation 0  (from URDF)

  So:
    T_C_L = inv(T_L_C) with rotation=diag(1,-1,-1), translation=[0,0,0.11] in C frame
    T_C_hand = T_C_L @ T_L_hand
    T_W_hand = T_W_C @ T_C_hand
    T_B_hand = T_B_W @ T_W_hand
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R


DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door")
OUT_DIR = "surfaces"
ROBOT_NAME = "panda"
TASK_NAME = os.environ.get("TASK_NAME", "cardoor")


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_T(R_mat, t_vec):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec
    return T


def main():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    T_B_W = np.array(meta["T_B_W"])
    T_L_C = np.array(meta["T_L_C"])

    # chain pieces
    T_C_L = np.linalg.inv(T_L_C)
    T_L_hand = build_T(rz(-np.pi / 4), np.zeros(3))  # from panda URDF
    T_C_hand = T_C_L @ T_L_hand

    # filter: only visit tool_pose_valid & accessibility rows
    mask = P["tool_pose_valid"] & P["accessibility"]
    t_W_C = P["tool_positions"][mask]       # (K,3)
    R_W_C = P["tool_rotations"][mask]        # (K,3,3)
    K = len(t_W_C)
    print(f"Points: total {len(P['points'])}, visitable {K}")

    # per-waypoint: compute hand pose in B frame
    rows = []
    for i in range(K):
        T_W_C_i = build_T(R_W_C[i], t_W_C[i])
        T_W_hand_i = T_W_C_i @ T_C_hand
        T_B_hand_i = T_B_W @ T_W_hand_i

        t_B = T_B_hand_i[:3, 3]
        R_B = T_B_hand_i[:3, :3]
        q = R.from_matrix(R_B).as_quat()  # xyzw
        rows.append((i, *t_B, *q))

    # save CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ROBOT_NAME}_{ts}_{TASK_NAME}.csv"
    out_path = os.path.join(OUT_DIR, fname)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time",
                    "target-POS_X", "target-POS_Y", "target-POS_Z",
                    "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W"])
        for row in rows:
            w.writerow(row)
    print(f"Saved {K} waypoints to {out_path}")

    # quick sanity: workspace bounds in B frame
    pos = np.array([r[1:4] for r in rows])
    print(f"B-frame position range:")
    print(f"  X: [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]")
    print(f"  Y: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
    print(f"  Z: [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")
    print(f"  distance from base origin: min={np.linalg.norm(pos, axis=1).min():.3f}, "
          f"max={np.linalg.norm(pos, axis=1).max():.3f}")


if __name__ == "__main__":
    main()
