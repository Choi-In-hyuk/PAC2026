"""Zigzag baseline: all 2917 waypoints, per-region zigzag by (line_id, line_order).

Within each region:
  - Group waypoints by line_id (-1 handled last).
  - Sort each line by line_order.
  - Alternate direction (snake): line 0 forward, line 1 reversed, ...
  - Lines with line_id=-1 (gap-fill/neighbor-search) appended at end in NN order.

Across regions: process in region_id order.

Writes motion CSV with target poses + IK (relaxed_ik with warm start).
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, hand_pose_from_C
from robot import Robot

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
TAG = os.environ.get("TAG", "zigzag_dir0")
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
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    mask = P["tool_pose_valid"] & P["accessibility"]
    orig = np.where(mask)[0]
    reg = P["region_ids"][orig]
    line_id = P["line_ids"][orig]
    line_order = P["line_order"][orig]
    t_B_C, R_B_C = compute_tool_C_poses_in_B(P, meta, orig)
    t_B_hand, R_B_hand = hand_pose_from_C(t_B_C, R_B_C, meta)
    quats = Rot.from_matrix(R_B_hand).as_quat()
    print(f"Total waypoints: {len(orig)}")

    final_order = []   # local indices into orig
    unique_regs = sorted(set(int(r) for r in reg))
    for rid in unique_regs:
        in_reg = np.where(reg == rid)[0]
        if len(in_reg) == 0: continue
        # separate line_id != -1 vs -1
        normal = in_reg[line_id[in_reg] >= 0]
        gaps   = in_reg[line_id[in_reg] <  0]

        # group normal by line_id
        lines = {}
        for i in normal:
            lid = int(line_id[i])
            lines.setdefault(lid, []).append(i)
        # sort each line by line_order
        for lid in lines:
            lines[lid].sort(key=lambda i: int(line_order[i]))

        # Snake order: line_id ascending, alternate direction
        sorted_lids = sorted(lines.keys())
        region_seq = []
        for k, lid in enumerate(sorted_lids):
            seg = lines[lid]
            if k % 2 == 1:
                seg = list(reversed(seg))
            region_seq.extend(seg)

        # gaps: append in NN order from last normal waypoint
        if len(gaps) > 0:
            gap_pos = t_B_hand[gaps]
            if region_seq:
                last = t_B_hand[region_seq[-1]]
                start = int(np.argmin(np.linalg.norm(gap_pos - last, axis=1)))
            else:
                start = 0
            order = nn_order(gap_pos, start=start)
            region_seq.extend([int(gaps[i]) for i in order])

        final_order.extend(region_seq)

    print(f"Zigzag order built: {len(final_order)} waypoints")

    # IK with warm start
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

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"motions/panda_{TAG}_{ts}.csv"
    joint_cols = [f"panda-panda_joint{i}" for i in range(1, 8)]
    fieldnames = ['time'] + joint_cols + ['target-POS_X','target-POS_Y','target-POS_Z',
                                           'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for t, ik, pose in rows:
            row = [t] + list(ik) + list(pose)
            w.writerow(dict(zip(fieldnames, row)))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
