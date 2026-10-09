"""Run author's hrchy_gtsp on each region separately.

Flow:
  1. Load input motion CSV (selected waypoints).
  2. For each waypoint, recover its region_id from the source npz.
  3. For each region: write region surface CSV → call hrchy_gtsp solver → read motion back.
  4. Concat per-region motions in region_id order.
"""
import os, sys, csv, glob, json, subprocess
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
MOTION_IN = os.environ.get("MOTION_IN", "")
TAG = os.environ.get("TAG", "per_region_hrchy")
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", 30))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 20))

MOTIONS_DIR = "motions"
SURFACES_DIR = "surfaces"
ROBOT_NAME = "panda"


def load_motion_waypoints(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    poses = []
    for r in rows:
        poses.append([float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                      float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])])
    return np.array(poses)


def load_source_waypoints():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    mask = P["tool_pose_valid"] & P["accessibility"]
    t_W_C = P["tool_positions"][mask]
    R_BW = T_B_W[:3, :3]; t_BW = T_B_W[:3, 3]
    t_B_C = (R_BW @ t_W_C.T).T + t_BW
    return t_B_C, P["region_ids"][mask]


def nearest_region_for_motion(motion_poses, src_pos_B, src_region):
    from eval_contact_area import hand_to_C_transforms
    hand_pos = motion_poses[:, :3]
    quats = motion_poses[:, 3:7]
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    R_B_hand = Rot.from_quat(quats).as_matrix()
    t_B_C_motion, _ = hand_to_C_transforms(hand_pos, R_B_hand)
    d = np.linalg.norm(t_B_C_motion[:, None, :] - src_pos_B[None, :, :], axis=2)
    nn = np.argmin(d, axis=1)
    return src_region[nn]


def run_hrchy_on_region(region_poses, region_id, time_limit, num_samples):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    surf_path = f"{SURFACES_DIR}/panda_{ts}_region{region_id}.csv"
    with open(surf_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time','target-POS_X','target-POS_Y','target-POS_Z',
                    'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W'])
        for i, p in enumerate(region_poses):
            w.writerow([i, *p])

    for name in ("joint_gtsp_high_level.tour", "joint_gtsp_low_level.tour"):
        f = f"GLKH/{name}"
        if os.path.exists(f): os.remove(f)

    env = os.environ.copy()
    env["SURFACE_PATTERN"] = surf_path
    env["TIME_LIMIT"] = str(time_limit)
    env["NUM_SAMPLES"] = str(num_samples)

    hard_timeout = time_limit * 3 + 120

    print(f"  [region {region_id}] {len(region_poses)} wp → hrchy (time_limit={time_limit}s, hard={hard_timeout}s)")
    try:
        subprocess.run(
            ["python", "scripts/run_cardoor_hrchy_only.py"],
            env=env, timeout=hard_timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")

    produced = sorted(glob.glob(f"{MOTIONS_DIR}/panda_hrchy_gtsp_*.csv"), key=os.path.getmtime)
    if not produced:
        print(f"    no motion produced")
        return []
    latest = produced[-1]
    out_poses = load_motion_waypoints(latest)
    order = []
    taken = set()
    for p in out_poses:
        d = np.linalg.norm(region_poses[:, :3] - p[:3], axis=1)
        for idx in np.argsort(d):
            if idx not in taken:
                order.append(int(idx)); taken.add(int(idx)); break
    os.remove(latest)
    print(f"    done: {len(order)} ordered")
    return order


def main():
    os.makedirs(MOTIONS_DIR, exist_ok=True)
    os.makedirs(SURFACES_DIR, exist_ok=True)

    if not MOTION_IN or not os.path.exists(MOTION_IN):
        print(f"MOTION_IN missing: {MOTION_IN}")
        return
    print(f"input motion: {MOTION_IN}")
    print(f"DATA_DIR: {DATA_DIR}, TIME_LIMIT per region: {TIME_LIMIT}s")

    motion_poses = load_motion_waypoints(MOTION_IN)
    print(f"selected waypoints: {len(motion_poses)}")

    src_pos_B, src_region = load_source_waypoints()
    region_of_motion = nearest_region_for_motion(motion_poses, src_pos_B, src_region)
    unique_regions = np.unique(region_of_motion)
    print(f"regions in selected: {len(unique_regions)}")

    all_order_poses = []
    for rid in unique_regions:
        idx_in_motion = np.where(region_of_motion == rid)[0]
        region_poses = motion_poses[idx_in_motion]
        if len(region_poses) == 0: continue
        order = run_hrchy_on_region(region_poses, int(rid), TIME_LIMIT, NUM_SAMPLES)
        if not order:
            # if hrchy produced nothing, append in incoming order (last resort, no re-ordering)
            order = list(range(len(region_poses)))
        ordered_abs = idx_in_motion[order]
        for i in ordered_abs:
            all_order_poses.append(motion_poses[i])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{MOTIONS_DIR}/panda_{TAG}_{ts}.csv"
    joint_cols = [f"{ROBOT_NAME}-panda_joint{i}" for i in range(1, 8)]
    fieldnames = ['time'] + joint_cols + ['target-POS_X','target-POS_Y','target-POS_Z',
                                           'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for i, p in enumerate(all_order_poses):
            row = [float(i)] + [0.0]*7 + list(p)
            w.writerow(dict(zip(fieldnames, row)))
    print(f"\nSaved: {out}  ({len(all_order_poses)} waypoints)")


if __name__ == "__main__":
    main()
