"""Region-aware adaptive FPS downsampling of car_door surface points.

Density policy (algorithm-friendly):
  - Big flat regions   (> 0.03 m²)   → min_dist 5 cm, dense enough for curvature
  - Medium regions     (0.005 ~ 0.03) → min_dist 2.5 cm
  - Small/detail       (< 0.005 m²)   → keep all points (critical detail)
  - Boundary points    (boundary_flags) → add if dropped by FPS (prevent gaps)
"""
import os, csv, json
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R


DATA_DIR = "meshes/car_door"
OUT_DIR = "surfaces"
ROBOT = "panda"
TASK = "cardooradaptive"


def fps_select(positions, min_dist, start_idx=0):
    if len(positions) == 0:
        return np.array([], dtype=int)
    selected = [start_idx]
    dist = np.linalg.norm(positions - positions[start_idx], axis=1)
    while True:
        i = int(np.argmax(dist))
        if dist[i] < min_dist:
            break
        selected.append(i)
        new_d = np.linalg.norm(positions - positions[i], axis=1)
        dist = np.minimum(dist, new_d)
    return np.array(selected)


def pick_min_dist(area_m2):
    if area_m2 > 0.03:
        return 0.05
    elif area_m2 > 0.005:
        return 0.025
    else:
        return 0.0   # keep all


def main():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    regions = json.load(open(os.path.join(DATA_DIR, "regions.json"), encoding="utf-8"))

    mask = P["accessibility"] & P["tool_pose_valid"]
    print(f"Visitable points: {int(mask.sum())}")

    all_pts_W = P["points"]              # 673 points indexed by original id
    all_tool_pos = P["tool_positions"]
    all_tool_rot = P["tool_rotations"]
    region_ids = P["region_ids"]
    boundary = P["boundary_flags"]

    selected_idx = set()
    stats = []

    for rid_str, r in regions.items():
        rid = int(rid_str)
        pt_ids = r["point_indices"]
        # keep only visitable
        pt_ids = [i for i in pt_ids if mask[i]]
        if not pt_ids:
            continue
        area = r["area_m2"]
        min_d = pick_min_dist(area)

        pts_in_region = all_pts_W[pt_ids]
        if min_d == 0.0:
            sel_local = np.arange(len(pts_in_region))
        else:
            sel_local = fps_select(pts_in_region, min_d)
        picked = [pt_ids[i] for i in sel_local]
        selected_idx.update(picked)
        stats.append((rid, len(pt_ids), len(picked), area, min_d))

    # add boundary points via FPS (prevents gaps without flooding)
    added_boundary = 0
    boundary_candidates = np.where(mask & boundary)[0]
    missing = [int(i) for i in boundary_candidates if int(i) not in selected_idx]
    if missing:
        miss_pts = all_pts_W[missing]
        # keep only boundary points at least 2.5cm from an already selected point
        selected_arr = np.array(sorted(selected_idx))
        sel_pts = all_pts_W[selected_arr]
        for i, bp in enumerate(miss_pts):
            d_sel = np.linalg.norm(sel_pts - bp, axis=1).min()
            if d_sel > 0.025:
                selected_idx.add(missing[i])
                added_boundary += 1
                sel_pts = np.vstack([sel_pts, bp[None]])

    selected = sorted(selected_idx)
    print(f"\nTotal selected: {len(selected)}  (original visitable {int(mask.sum())})")
    print(f"Boundary points added back: {added_boundary}")
    print()
    print(f"{'region':>7} {'orig':>5} {'picked':>7} {'area':>7} {'min_d':>7}")
    for rid, orig, picked, area, min_d in sorted(stats, key=lambda x: -x[1]):
        print(f"{rid:>7} {orig:>5} {picked:>7} {area:>7.4f} {min_d:>7.3f}")

    # Build CSV in robot base frame
    T_B_W = np.array(meta["T_B_W"])
    T_L_C = np.array(meta["T_L_C"])
    T_C_L = np.linalg.inv(T_L_C)
    T_L_hand = np.eye(4)
    T_L_hand[:3, :3] = R.from_euler('z', -np.pi / 4).as_matrix()
    T_C_hand = T_C_L @ T_L_hand

    rows = []
    for i, idx in enumerate(selected):
        T_W_C = np.eye(4)
        T_W_C[:3, :3] = all_tool_rot[idx]
        T_W_C[:3, 3] = all_tool_pos[idx]
        T_W_hand = T_W_C @ T_C_hand
        T_B_hand = T_B_W @ T_W_hand
        t = T_B_hand[:3, 3]
        q = R.from_matrix(T_B_hand[:3, :3]).as_quat()
        rows.append((i, *t, *q))

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{OUT_DIR}/{ROBOT}_{ts}_{TASK}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time",
                    "target-POS_X", "target-POS_Y", "target-POS_Z",
                    "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W"])
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {len(rows)} waypoints to {out}")


if __name__ == "__main__":
    main()
