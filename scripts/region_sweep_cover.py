"""Region-wise greedy set cover with sweep-aware gain.

Same as region_set_cover.py BUT each waypoint's 'covered face set' includes
the interpolated sweep from each already-selected neighbor waypoint in the
same region (if the gap <= SWEEP_INTERP_MAX).

Greedy picks the waypoint that maximizes newly-covered area under the sweep
assumption. This matches the region-sweep evaluation model.
"""
import os, sys, csv, json
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from robot import Robot
from region_set_cover import (DATA_DIR, MOTIONS_DIR, ROBOT_NAME, TAG,
                               R_TOOL, PAD_INDENT, TOOL_OFFSET, HAND_YAW,
                               rz, build_T, load_surface_and_mesh,
                               compute_tool_C_poses_in_B, hand_pose_from_C,
                               mesh_centers_areas, contact_faces_for_waypoint)


INTERP_STEP = 0.002  # 2 mm, same as eval_region_sweep


def quat_slerp(R_a, R_b, u):
    qa = Rot.from_matrix(R_a).as_quat()
    qb = Rot.from_matrix(R_b).as_quat()
    if np.dot(qa, qb) < 0: qb = -qb
    qc = (1 - u) * qa + u * qb
    qc = qc / np.linalg.norm(qc)
    return Rot.from_quat(qc).as_matrix()


def footprint_mask(t_B_C, R_B_C, centers_B):
    delta = centers_B - t_B_C
    p_C = (R_B_C.T @ delta.T).T
    xy2 = p_C[:, 0] ** 2 + p_C[:, 1] ** 2
    z = p_C[:, 2]
    return (xy2 <= R_TOOL * R_TOOL) & (np.abs(z) <= PAD_INDENT)


def sweep_mask_between(ta, Ra, tb, Rb, centers_B):
    """Union footprint along interpolated poses from a to b, inclusive both ends."""
    m = footprint_mask(ta, Ra, centers_B) | footprint_mask(tb, Rb, centers_B)
    d = np.linalg.norm(tb - ta)
    n_sub = max(1, int(np.ceil(d / INTERP_STEP)))
    for s in range(1, n_sub):
        u = s / n_sub
        tm = (1 - u) * ta + u * tb
        Rm = quat_slerp(Ra, Rb, u)
        m |= footprint_mask(tm, Rm, centers_B)
    return m


def main():
    P, M, meta = load_surface_and_mesh()
    mask = P["tool_pose_valid"] & P["accessibility"]
    idx = np.where(mask)[0]
    K = len(idx)
    print(f"방문가능 waypoint: {K}")

    region_ids_wp = P["region_ids"][idx]
    line_ids_wp = P["line_ids"][idx] if "line_ids" in P else np.full(K, -1)
    line_order_wp = P["line_order"][idx] if "line_order" in P else np.arange(K)

    t_B_C, R_B_C = compute_tool_C_poses_in_B(P, meta, idx)

    centers_B, areas, face_status, face_region = mesh_centers_areas(M, meta)
    feasible_face_mask = (face_status == 1)
    print(f"feasible faces: {feasible_face_mask.sum()}, area {areas[feasible_face_mask].sum()*1e4:.2f} cm²")

    unique_regions = np.unique(region_ids_wp)
    print(f"regions in use: {len(unique_regions)}")

    selected_wp_global = []
    stats = []

    for rid in unique_regions:
        wp_in = np.where(region_ids_wp == rid)[0]
        if len(wp_in) == 0: continue
        face_in = np.where((face_region == rid) & feasible_face_mask)[0]
        if len(face_in) == 0:
            selected_wp_global.extend(wp_in.tolist())
            continue

        tc = t_B_C[wp_in]
        Rc = R_B_C[wp_in]
        centers_loc = centers_B[face_in]
        area_loc = areas[face_in]

        # single-point footprint for each candidate
        single = [footprint_mask(tc[i], Rc[i], centers_loc) for i in range(len(wp_in))]

        covered = np.zeros(len(face_in), dtype=bool)
        selected_local = []
        selected_set = set()
        remaining = set(range(len(wp_in)))

        while remaining:
            best = -1; best_gain = 0.0
            for j in remaining:
                # candidate footprint = single[j] UNION sweep-with-each-selected-neighbor
                cand = single[j].copy()
                for k in selected_local:
                    # include sweep between j and k (interpolation)
                    cand |= sweep_mask_between(tc[k], Rc[k], tc[j], Rc[j], centers_loc)
                new_cov = cand & ~covered
                gain = area_loc[new_cov].sum()
                if gain > best_gain:
                    best_gain = gain; best = j; best_mask = cand
            if best < 0 or best_gain <= 0: break
            selected_local.append(best)
            selected_set.add(best)
            covered |= best_mask
            remaining.remove(best)

        selected_wp_global.extend(wp_in[selected_local].tolist())
        stats.append((int(rid), len(wp_in), len(selected_local),
                      area_loc[covered].sum(), area_loc.sum()))

    sel = np.array(selected_wp_global, dtype=int)
    sort_line = np.where(line_ids_wp[sel] < 0, 10**9, line_ids_wp[sel])
    order = np.lexsort((line_order_wp[sel], sort_line))
    sel_sorted = sel[order]
    print(f"\n선택된 waypoint: {len(sel_sorted)} / {K}  (제거 {K - len(sel_sorted)})")

    robot = Robot(ROBOT_NAME)
    tolerances = [0.03, 0.03, 0., 0., 0., 999.]

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
            failed += 1; continue
        joints.append(ik); kept_hand.append(pose); last_ik = ik
        if (i + 1) % 100 == 0:
            print(f"  IK {i+1}/{len(hand_poses)} (fail={failed})")
    print(f"IK done: ok={len(joints)}, failed={failed}")

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
