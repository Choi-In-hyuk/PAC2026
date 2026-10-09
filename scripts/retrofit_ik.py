"""Fill in joint values for an existing motion CSV using relaxed_ik.

Usage:
    python scripts/retrofit_ik.py <input_motion.csv> [<output_motion.csv>]

Reads target-POS/ROT, solves IK sequentially (warm start), writes CSV with joints.
"""
import sys, os, csv
import numpy as np

sys.path.insert(0, "scripts")
from robot import Robot

ROBOT_NAME = "panda"
TOL = [0.03, 0.03, 0., 0., 0., 999.]


def main():
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".csv", "_withIK.csv")

    with open(inp) as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys())
    print(f"Loaded {len(rows)} rows from {inp}")

    poses = []
    for r in rows:
        poses.append([float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                      float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])])

    robot = Robot(ROBOT_NAME)
    joint_cols = [c for c in header if c.startswith(f"{ROBOT_NAME}-")]
    n_j = len(joint_cols)
    print(f"Robot: {ROBOT_NAME}, {n_j} joints")

    joints = []
    last_ik = []
    failed = 0
    for i, pose in enumerate(poses):
        ik = robot.reach_with_relaxed_ik(pose, False, tolerances=TOL,
                                         start_config=last_ik, max_iter=20)
        if len(ik) == 0:
            for _ in range(3):
                ik = robot.reach_with_relaxed_ik(pose, False, tolerances=TOL,
                                                 start_config=[], max_iter=20)
                if len(ik) > 0: break
        if len(ik) == 0:
            # keep previous joints as fallback (so motion still has entry)
            ik = last_ik if last_ik else [0.0]*n_j
            failed += 1
        joints.append(ik)
        last_ik = ik
        if (i+1) % 100 == 0:
            print(f"  IK {i+1}/{len(poses)}  fail={failed}")
    print(f"IK done: ok={len(poses)-failed}, failed={failed}")

    # write
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for i, (r, j) in enumerate(zip(rows, joints)):
            new = {**r}
            for k, jk in enumerate(joint_cols):
                new[jk] = j[k]
            w.writerow(new)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
