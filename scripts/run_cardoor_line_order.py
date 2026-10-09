"""Order-preserving IK runner for car_door.

Sorts waypoints by (line_ids, line_order) and solves IK sequentially using
the previous solution as warm start. Produces a motion CSV comparable to
hrchy_gtsp output but with the line-based process order kept intact.
"""
import os, sys, csv, json, glob
import numpy as np
from datetime import datetime
sys.path.insert(0, "scripts")

from robot import Robot
from scipy.spatial.transform import Rotation as R


DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door")
MOTIONS_DIR = "motions"
ROBOT_NAME = "panda"
TAG = os.environ.get("TAG", "line_order")


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_T(R_mat, t_vec):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec
    return T


def load_ordered_hand_poses():
    """Return list of (pose_7, line_id, line_order) sorted by line order.

    Each pose_7 is [x, y, z, qx, qy, qz, qw] in robot BASE frame.
    """
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    T_B_W = np.array(meta["T_B_W"])
    T_L_C = np.array(meta["T_L_C"])
    T_C_L = np.linalg.inv(T_L_C)
    T_L_hand = build_T(rz(-np.pi / 4), np.zeros(3))
    T_C_hand = T_C_L @ T_L_hand

    mask = P["tool_pose_valid"] & P["accessibility"]
    idx = np.where(mask)[0]

    t_W_C = P["tool_positions"][idx]
    R_W_C = P["tool_rotations"][idx]
    line_ids = P["line_ids"][idx]
    line_order = P["line_order"][idx]

    # Sort: lines first (gap-fill points with line_id=-1 go last),
    # within a line sort by line_order.
    LARGE = 10**9
    sort_line = np.where(line_ids < 0, LARGE, line_ids)
    sort_key = np.lexsort((line_order, sort_line))

    out = []
    for k in sort_key:
        T_W_C_i = build_T(R_W_C[k], t_W_C[k])
        T_W_hand_i = T_W_C_i @ T_C_hand
        T_B_hand_i = T_B_W @ T_W_hand_i
        t_B = T_B_hand_i[:3, 3]
        q = R.from_matrix(T_B_hand_i[:3, :3]).as_quat()
        pose7 = [*t_B, *q]
        out.append((pose7, int(line_ids[k]), int(line_order[k])))

    return out


def main():
    poses_meta = load_ordered_hand_poses()
    print(f"Ordered waypoints: {len(poses_meta)}")

    robot = Robot(ROBOT_NAME)
    tolerances = [0.03, 0.03, 0., 0., 0., 999.]

    joints = []
    kept_poses = []
    kept_meta = []
    last_ik = []
    failed = 0

    for i, (pose, lid, lo) in enumerate(poses_meta):
        ik = robot.reach_with_relaxed_ik(pose, False, tolerances=tolerances,
                                         start_config=last_ik, max_iter=20)
        if len(ik) == 0:
            # retry with random seed a few times
            for _ in range(3):
                ik = robot.reach_with_relaxed_ik(pose, False, tolerances=tolerances,
                                                 start_config=[], max_iter=20)
                if len(ik) > 0:
                    break
        if len(ik) == 0:
            failed += 1
            continue
        joints.append(ik)
        kept_poses.append(pose)
        kept_meta.append((lid, lo))
        last_ik = ik

        if (i + 1) % 100 == 0:
            print(f"  solved {i+1}/{len(poses_meta)} (fail={failed})")

    print(f"IK done: ok={len(joints)}, failed={failed}")

    # joint movement + reconfig count (simple, same as solver_base definition style)
    ds = 0.0
    reconfigs = 0
    for i in range(len(joints) - 1):
        d = float(np.max(np.abs(np.array(joints[i]) - np.array(joints[i + 1]))))
        posi_d = float(np.linalg.norm(np.array(kept_poses[i][:3]) - np.array(kept_poses[i + 1][:3])))
        if d > 1.0 or posi_d > 0.2:
            reconfigs += 1
        else:
            ds += float(np.linalg.norm(np.array(joints[i]) - np.array(joints[i + 1])))
    print(f"joint movement: {ds:.3f},  reconfigs: {reconfigs}")

    # save CSV in the same format solver_base.save_motion uses
    os.makedirs(MOTIONS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ROBOT_NAME}_{TAG}_{ts}.csv"
    out_path = os.path.join(MOTIONS_DIR, fname)
    joint_names = [f"{ROBOT_NAME}-{jn}" for jn in robot.rk._joint_names]
    fieldnames = (['time'] + joint_names +
                  ['target-POS_X', 'target-POS_Y', 'target-POS_Z',
                   'target-ROT_X', 'target-ROT_Y', 'target-ROT_Z', 'target-ROT_W'])
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (ik, pose) in enumerate(zip(joints, kept_poses)):
            row = dict(zip(fieldnames, [float(i), *ik, *pose]))
            w.writerow(row)
    print(f"Motion saved to: {out_path}")


if __name__ == "__main__":
    main()
