"""Region-wise greedy set cover for car_door waypoint selection.

For each region (88 regions in v2.6):
  1. Compute each waypoint's contact face set (pad disc + 0.5mm indent band).
  2. Greedy set cover: pick waypoints that maximize newly covered face area
     until the region's feasible face area is covered (or we run out).
  3. Order the selected waypoints by (line_id, line_order) for process continuity.

Across regions:
  - Concatenate region orderings in region_id order (simple).
  - Then solve IK sequentially with warm start.

Outputs:
  motions/panda_region_cover_<ts>.csv
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from robot import Robot


DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door")
MOTIONS_DIR = "motions"
ROBOT_NAME = "panda"
TAG = os.environ.get("TAG", "region_cover")

R_TOOL = 0.05
PAD_INDENT = 0.0005
TOOL_OFFSET = 0.11
HAND_YAW = -np.pi / 4


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_T(R_mat, t_vec):
    T = np.eye(4); T[:3, :3] = R_mat; T[:3, 3] = t_vec
    return T


def load_surface_and_mesh():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    return P, M, meta


def compute_tool_C_poses_in_B(P, meta, mask_idx):
    """Return (t_B_C (K,3), R_B_C (K,3,3)) in robot base frame for the K waypoints."""
    T_B_W = np.array(meta["T_B_W"])
    t_W_C = P["tool_positions"][mask_idx]        # (K,3), W frame
    R_W_C = P["tool_rotations"][mask_idx]        # (K,3,3), W frame
    R_BW = T_B_W[:3, :3]; t_BW = T_B_W[:3, 3]
    t_B_C = (R_BW @ t_W_C.T).T + t_BW
    R_B_C = np.einsum("ij,kjl->kil", R_BW, R_W_C)
    return t_B_C, R_B_C


def hand_pose_from_C(t_B_C, R_B_C, meta):
    """Convert tool C-frame pose to panda_hand pose in B frame."""
    T_L_C = np.array(meta["T_L_C"])
    T_C_L = np.linalg.inv(T_L_C)
    T_L_hand = build_T(rz(-np.pi / 4), np.zeros(3))
    T_C_hand = T_C_L @ T_L_hand
    R_C_hand = T_C_hand[:3, :3]
    t_C_hand = T_C_hand[:3, 3]
    t_B_hand = t_B_C + np.einsum("kij,j->ki", R_B_C, t_C_hand)
    R_B_hand = np.einsum("kij,jl->kil", R_B_C, R_C_hand)
    return t_B_hand, R_B_hand


def mesh_centers_areas(M, meta):
    V = M["vertices"]; F = M["faces"]; status = M["face_status"]; freg = M["face_region_ids"]
    v0 = V[F[:, 0]]; v1 = V[F[:, 1]]; v2 = V[F[:, 2]]
    centers_W = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    T_B_W = np.array(meta["T_B_W"])
    R_BW = T_B_W[:3, :3]; t_BW = T_B_W[:3, 3]
    centers_B = (R_BW @ centers_W.T).T + t_BW
    return centers_B, areas, status, freg


def contact_faces_for_waypoint(t_B_C_i, R_B_C_i, centers_B, R=R_TOOL, h=PAD_INDENT):
    """Return boolean mask of faces whose center lies in the pad contact band."""
    delta = centers_B - t_B_C_i                 # (N,3)
    p_C = (R_B_C_i.T @ delta.T).T               # (N,3)
    xy2 = p_C[:, 0] ** 2 + p_C[:, 1] ** 2
    z = p_C[:, 2]
    return (xy2 <= R * R) & (np.abs(z) <= h)


def main():
    P, M, meta = load_surface_and_mesh()

    mask = P["tool_pose_valid"] & P["accessibility"]
    idx = np.where(mask)[0]
    K = len(idx)
    print(f"방문가능 waypoint: {K}")

    region_ids_wp = P["region_ids"][idx]
    line_ids_wp = P["line_ids"][idx]
    line_order_wp = P["line_order"][idx]

    t_B_C, R_B_C = compute_tool_C_poses_in_B(P, meta, idx)

    centers_B, areas, face_status, face_region = mesh_centers_areas(M, meta)
    feasible_face_mask = (face_status == 1)
    print(f"feasible faces: {feasible_face_mask.sum()}, total working area: "
          f"{areas[feasible_face_mask].sum()*1e4:.2f} cm²")

    # per-region grouping
    unique_regions = np.unique(region_ids_wp)
    print(f"regions in use: {len(unique_regions)}")

    selected_wp_global = []  # indices into idx
    region_cov_stats = []

    for rid in unique_regions:
        wp_in = np.where(region_ids_wp == rid)[0]  # local idx into idx[]
        if len(wp_in) == 0:
            continue
        # face set belonging to this region AND feasible
        face_in = np.where((face_region == rid) & feasible_face_mask)[0]
        if len(face_in) == 0:
            # no mesh face assigned to this region; still include waypoints
            selected_wp_global.extend(wp_in.tolist())
            continue

        # compute contact face set for each waypoint, restricted to this region's faces
        patches = []
        for li in wp_in:
            m = contact_faces_for_waypoint(t_B_C[li], R_B_C[li], centers_B[face_in])
            patches.append(np.where(m)[0])  # local idx into face_in

        # greedy set cover (area-weighted)
        face_area_local = areas[face_in]
        covered = np.zeros(len(face_in), dtype=bool)
        total_area = face_area_local.sum()
        selected_local = []

        remaining = set(range(len(wp_in)))
        while remaining:
            best = -1; best_gain = 0.0
            for j in remaining:
                gain = face_area_local[patches[j][~covered[patches[j]]]].sum()
                if gain > best_gain:
                    best_gain = gain; best = j
            if best < 0 or best_gain <= 0:
                break
            selected_local.append(best)
            covered[patches[best]] = True
            remaining.remove(best)

        selected_wp_global.extend(wp_in[selected_local].tolist())
        cov_area = face_area_local[covered].sum()
        region_cov_stats.append((int(rid), len(wp_in), len(selected_local),
                                  cov_area, total_area))

    # order selected waypoints by (line_id, line_order) for process continuity
    sel = np.array(selected_wp_global, dtype=int)
    sort_line = np.where(line_ids_wp[sel] < 0, 10**9, line_ids_wp[sel])
    order = np.lexsort((line_order_wp[sel], sort_line))
    sel_sorted = sel[order]
    print(f"\n선택된 waypoint: {len(sel_sorted)} / {K}  "
          f"(제거: {K - len(sel_sorted)})")

    # IK with warm start (same as line_order runner)
    robot = Robot(ROBOT_NAME)
    tolerances = [0.03, 0.03, 0., 0., 0., 999.]

    # hand pose for each selected waypoint
    t_B_hand, R_B_hand = hand_pose_from_C(t_B_C[sel_sorted], R_B_C[sel_sorted], meta)
    quats = Rot.from_matrix(R_B_hand).as_quat()
    hand_poses = [list(np.concatenate([t_B_hand[i], quats[i]])) for i in range(len(sel_sorted))]

    joints = []; kept_hand = []
    last_ik = []; failed = 0
    for i, pose in enumerate(hand_poses):
        ik = robot.reach_with_relaxed_ik(pose, False, tolerances=tolerances,
                                         start_config=last_ik, max_iter=20)
        if len(ik) == 0:
            for _ in range(3):
                ik = robot.reach_with_relaxed_ik(pose, False, tolerances=tolerances,
                                                 start_config=[], max_iter=20)
                if len(ik) > 0: break
        if len(ik) == 0:
            failed += 1
            continue
        joints.append(ik); kept_hand.append(pose); last_ik = ik
        if (i + 1) % 100 == 0:
            print(f"  IK {i+1}/{len(hand_poses)} (fail={failed})")
    print(f"IK done: ok={len(joints)}, failed={failed}")

    # joint metrics
    ds = 0.0; reconfigs = 0
    for i in range(len(joints) - 1):
        dmax = float(np.max(np.abs(np.array(joints[i]) - np.array(joints[i+1]))))
        posd = float(np.linalg.norm(np.array(kept_hand[i][:3]) - np.array(kept_hand[i+1][:3])))
        if dmax > 1.0 or posd > 0.2:
            reconfigs += 1
        else:
            ds += float(np.linalg.norm(np.array(joints[i]) - np.array(joints[i+1])))
    print(f"joint movement: {ds:.3f}, reconfigs: {reconfigs}")

    # save motion
    os.makedirs(MOTIONS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ROBOT_NAME}_{TAG}_{ts}.csv"
    out = os.path.join(MOTIONS_DIR, fname)
    joint_names = [f"{ROBOT_NAME}-{jn}" for jn in robot.rk._joint_names]
    fieldnames = (['time'] + joint_names +
                  ['target-POS_X','target-POS_Y','target-POS_Z',
                   'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W'])
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for i, (ik, pose) in enumerate(zip(joints, kept_hand)):
            w.writerow(dict(zip(fieldnames, [float(i), *ik, *pose])))
    print(f"Motion saved to: {out}")


if __name__ == "__main__":
    main()
