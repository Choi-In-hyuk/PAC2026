"""Downsample car_door surface points to coarser spacing via Farthest Point Sampling.

From the dense 1cm sampling, pick a subset so that no two picked points are
closer than min_dist. Keeps original tool_positions and tool_rotations.

Usage:
  python scripts/downsample_cardoor.py 0.03   # ~3cm spacing
  python scripts/downsample_cardoor.py 0.05   # ~5cm spacing
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R

DATA_DIR = "meshes/car_door"
OUT_DIR = "surfaces"
ROBOT = "panda"


def fps_select(positions, min_dist):
    """Farthest Point Sampling. Returns indices of selected points."""
    N = len(positions)
    selected = [0]
    dist = np.linalg.norm(positions - positions[0], axis=1)
    while True:
        i = int(np.argmax(dist))
        if dist[i] < min_dist:
            break
        selected.append(i)
        new_d = np.linalg.norm(positions - positions[i], axis=1)
        dist = np.minimum(dist, new_d)
    return np.array(selected)


def main():
    min_dist = float(sys.argv[1])
    cm = int(round(min_dist * 100))
    task_tag = f"cardoor{cm}cm"

    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    mask = P["accessibility"] & P["tool_pose_valid"]
    pts_W = P["points"][mask]
    tool_pos_W = P["tool_positions"][mask]
    tool_rot_W = P["tool_rotations"][mask]

    sel = fps_select(pts_W, min_dist)
    print(f"Downsample {len(pts_W)} -> {len(sel)} points (min_dist = {cm}cm)")

    # Chain to robot base: hand pose from tool pose (see car_door_to_surface.py)
    T_B_W = np.array(meta["T_B_W"])
    T_L_C = np.array(meta["T_L_C"])
    T_C_L = np.linalg.inv(T_L_C)
    T_L_hand = np.eye(4)
    rz45 = R.from_euler('z', -np.pi/4).as_matrix()
    T_L_hand[:3, :3] = rz45
    T_C_hand = T_C_L @ T_L_hand

    rows = []
    for i, idx in enumerate(sel):
        T_W_C = np.eye(4)
        T_W_C[:3, :3] = tool_rot_W[idx]
        T_W_C[:3, 3] = tool_pos_W[idx]
        T_W_hand = T_W_C @ T_C_hand
        T_B_hand = T_B_W @ T_W_hand
        t = T_B_hand[:3, 3]
        q = R.from_matrix(T_B_hand[:3, :3]).as_quat()
        rows.append((i, *t, *q))

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{OUT_DIR}/{ROBOT}_{ts}_{task_tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time",
                    "target-POS_X", "target-POS_Y", "target-POS_Z",
                    "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W"])
        for r in rows:
            w.writerow(r)
    print(f"Saved {len(rows)} waypoints to {out}")


if __name__ == "__main__":
    main()
