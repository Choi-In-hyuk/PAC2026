"""NN baseline: all 2917 waypoints, per-region nearest-neighbor order.

Within each region: NN greedy from waypoint 0 (local index).
Across regions: region_id order.
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, hand_pose_from_C
from robot import Robot

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
TAG = os.environ.get("TAG", "nn_dir0")
ROBOT_NAME = "panda"
TOL = [0.03, 0.03, 0., 0., 0., 999.]


def nn_order(positions, start=0):
    n = len(positions)
    if n == 0: return []
    if n == 1: return [0]
    visited = [start]
    rem = set(range(n)) - {start}
    while rem:
        cur = positions[visited[-1]]
        best = min(rem, key=lambda i: np.linalg.norm(positions[i] - cur))
        visited.append(best); rem.remove(best)
    return visited


def main():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    mask = P["tool_pose_valid"] & P["accessibility"]
    orig = np.where(mask)[0]
    reg = P["region_ids"][orig]
    t_B_C, R_B_C = compute_tool_C_poses_in_B(P, meta, orig)
    t_B_hand, R_B_hand = hand_pose_from_C(t_B_C, R_B_C, meta)
    quats = Rot.from_matrix(R_B_hand).as_quat()
    print(f"Total waypoints: {len(orig)}")

    final_order = []
    unique_regs = sorted(set(int(r) for r in reg))
    for rid in unique_regs:
        in_reg = np.where(reg == rid)[0]
        if len(in_reg) == 0: continue
        pos = t_B_hand[in_reg]
        order = nn_order(pos, start=0)
        final_order.extend([int(in_reg[i]) for i in order])
    print(f"NN order built: {len(final_order)} waypoints")

    robot = Robot(ROBOT_NAME)
    rows = []
    last_ik = []
    failed = 0
    for i, li in enumerate(final_order):
        pose = list(np.concatenate([t_B_hand[li], quats[li]]))
        ik = robot.reach_with_relaxed_ik(pose, False, tolerances=TOL,
                                         start_config=last_ik, max_iter=20)
        if len(ik) == 0:
            for _ in range(3):
                ik = robot.reach_with_relaxed_ik(pose, False, tolerances=TOL,
                                                 start_config=[], max_iter=20)
                if len(ik) > 0: break
        if len(ik) == 0:
            ik = last_ik if last_ik else [0.0]*7
            failed += 1
        rows.append((float(i), ik, pose))
        last_ik = ik
        if (i+1) % 200 == 0:
            print(f"  IK {i+1}/{len(final_order)}  fail={failed}")
    print(f"IK done: ok={len(final_order)-failed}, failed={failed}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"motions/panda_{TAG}_{ts}.csv"
    joint_cols = [f"panda-panda_joint{i}" for i in range(1, 8)]
    fieldnames = ['time'] + joint_cols + ['target-POS_X','target-POS_Y','target-POS_Z',
                                           'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for t, ik, pose in rows:
            w.writerow(dict(zip(fieldnames, [t] + list(ik) + list(pose))))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
