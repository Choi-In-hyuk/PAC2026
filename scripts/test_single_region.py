"""Single-region test: hrchy_gtsp sweep + stamp supplement.

1) Pick one region
2) Extract its waypoints (feasible ones), write surface CSV
3) Run hrchy_gtsp on just those waypoints
4) Evaluate sweep coverage of the hrchy motion within that region's faces
5) Find uncovered faces, add stamp waypoints from the SAME region's unused waypoints
6) Re-evaluate: sweep + stamp
"""
import os, sys, csv, json, subprocess, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from datetime import datetime

sys.path.insert(0, "scripts")
from eval_contact_area import (R_TOOL, PAD_INDENT, TOOL_OFFSET, HAND_YAW,
                                hand_to_C_transforms)
from region_set_cover import (rz, build_T, hand_pose_from_C)

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
REGION_ID = int(os.environ.get("REGION_ID", "4"))
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "30"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "20"))
INTERP_STEP = 0.002


def footprint_mask(t, R, centers):
    delta = centers - t
    p_C = (R.T @ delta.T).T
    xy2 = p_C[:,0]**2 + p_C[:,1]**2
    z = p_C[:,2]
    return (xy2 <= R_TOOL**2) & (np.abs(z) <= PAD_INDENT)


def slerp(R_a, R_b, u):
    qa = Rot.from_matrix(R_a).as_quat()
    qb = Rot.from_matrix(R_b).as_quat()
    if np.dot(qa, qb) < 0: qb = -qb
    qc = (1-u)*qa + u*qb
    return Rot.from_quat(qc / np.linalg.norm(qc)).as_matrix()


def sweep_visit(t_B_C, R_B_C, centers):
    """Union of footprints along the full sweep (all waypoints as one pass)."""
    N = len(centers)
    mask = np.zeros(N, dtype=bool)
    for k in range(len(t_B_C)):
        mask |= footprint_mask(t_B_C[k], R_B_C[k], centers)
        if k < len(t_B_C) - 1:
            d = np.linalg.norm(t_B_C[k+1] - t_B_C[k])
            n_sub = max(1, int(np.ceil(d / INTERP_STEP)))
            for s in range(1, n_sub):
                u = s / n_sub
                tm = (1-u)*t_B_C[k] + u*t_B_C[k+1]
                Rm = slerp(R_B_C[k], R_B_C[k+1], u)
                mask |= footprint_mask(tm, Rm, centers)
    return mask


def stamp_visit(t_B_C, R_B_C, centers):
    """Just union of individual footprints (no interpolation)."""
    N = len(centers)
    mask = np.zeros(N, dtype=bool)
    for k in range(len(t_B_C)):
        mask |= footprint_mask(t_B_C[k], R_B_C[k], centers)
    return mask


def main():
    print(f"=== Region {REGION_ID} test ===\n")

    # Load source data
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW = T_B_W[:3,:3]; t_BW = T_B_W[:3,3]

    # waypoints in this region
    mask = P["tool_pose_valid"] & P["accessibility"]
    reg = P["region_ids"][mask]
    idx_in_mask = np.where(reg == REGION_ID)[0]
    orig_idx = np.where(mask)[0][idx_in_mask]
    K = len(orig_idx)
    print(f"waypoints in region: {K}")

    t_W_C = P["tool_positions"][orig_idx]
    R_W_C = P["tool_rotations"][orig_idx]
    t_B_C_all = (R_BW @ t_W_C.T).T + t_BW
    R_B_C_all = np.einsum("ij,kjl->kil", R_BW, R_W_C)

    # faces in this region (feasible only)
    F = M["faces"]; V = M["vertices"]
    face_region = M["face_region_ids"]
    face_status = M["face_status"]
    face_in_reg = np.where((face_region == REGION_ID) & (face_status == 1))[0]
    print(f"feasible faces in region: {len(face_in_reg)}")
    v0 = V[F[face_in_reg,0]]; v1 = V[F[face_in_reg,1]]; v2 = V[F[face_in_reg,2]]
    centers_W = (v0+v1+v2)/3.0
    areas = 0.5*np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    centers_B = (R_BW @ centers_W.T).T + t_BW
    total_area = areas.sum()
    print(f"total working area in region: {total_area*1e4:.2f} cm²\n")

    # Step 1: write region surface CSV (hand poses)
    t_B_hand, R_B_hand = hand_pose_from_C(t_B_C_all, R_B_C_all, meta)
    quats = Rot.from_matrix(R_B_hand).as_quat()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    surf_path = f"surfaces/panda_{ts}_test_region{REGION_ID}.csv"
    with open(surf_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time','target-POS_X','target-POS_Y','target-POS_Z',
                    'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W'])
        for i in range(K):
            w.writerow([i, *t_B_hand[i], *quats[i]])
    print(f"surface CSV: {surf_path}\n")

    # Step 2: run hrchy_gtsp on this region
    for name in ("joint_gtsp_high_level.tour", "joint_gtsp_low_level.tour"):
        p = f"GLKH/{name}"
        if os.path.exists(p): os.remove(p)

    env = os.environ.copy()
    env["SURFACE_PATTERN"] = surf_path
    env["TIME_LIMIT"] = str(TIME_LIMIT)
    env["NUM_SAMPLES"] = str(NUM_SAMPLES)

    hard = TIME_LIMIT * 3 + 180
    print(f"running hrchy_gtsp (time_limit={TIME_LIMIT}s, hard timeout={hard}s)...")
    try:
        subprocess.run(["python", "scripts/run_cardoor_hrchy_only.py"],
                       env=env, timeout=hard, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return

    produced = sorted(glob.glob("motions/panda_hrchy_gtsp_*.csv"), key=os.path.getmtime)
    if not produced:
        print("no motion produced"); return
    motion_path = produced[-1]
    print(f"motion: {motion_path}\n")

    # Step 3: Load motion, compute sweep coverage
    with open(motion_path) as f:
        rows = list(csv.DictReader(f))
    motion_hand = np.array([[float(r['target-POS_X']),float(r['target-POS_Y']),float(r['target-POS_Z']),
                              float(r['target-ROT_X']),float(r['target-ROT_Y']),float(r['target-ROT_Z']),float(r['target-ROT_W'])] for r in rows])
    motion_hand_pos = motion_hand[:,:3]
    motion_quats = motion_hand[:,3:7]
    motion_quats /= np.linalg.norm(motion_quats, axis=1, keepdims=True)
    motion_R_hand = Rot.from_quat(motion_quats).as_matrix()
    motion_t_B_C, motion_R_B_C = hand_to_C_transforms(motion_hand_pos, motion_R_hand)
    print(f"motion waypoints: {len(motion_hand)}")

    # Sweep coverage
    swept = sweep_visit(motion_t_B_C, motion_R_B_C, centers_B)
    cov_sweep = areas[swept].sum()
    print(f"\n[sweep only] coverage: {cov_sweep/total_area*100:.2f}% ({cov_sweep*1e4:.2f}/{total_area*1e4:.2f} cm²)")
    uncovered = ~swept
    uncov_area = areas[uncovered].sum()
    print(f"  uncovered: {uncovered.sum()} faces, {uncov_area*1e4:.2f} cm²")

    if uncovered.sum() == 0:
        print("fully covered already, no stamp needed")
        return

    # Step 4: stamp candidates = ALL waypoints in dataset (not just this region),
    # excluding those already used in the sweep motion.
    # Load all waypoints:
    mask_all = P["tool_pose_valid"] & P["accessibility"]
    orig_all = np.where(mask_all)[0]
    t_W_C_all_full = P["tool_positions"][orig_all]
    R_W_C_all_full = P["tool_rotations"][orig_all]
    reg_all = P["region_ids"][orig_all]
    t_B_C_full = (R_BW @ t_W_C_all_full.T).T + t_BW
    R_B_C_full = np.einsum("ij,kjl->kil", R_BW, R_W_C_all_full)

    # waypoints already in the sweep (by nearest match on hand pose)
    t_B_hand_full, R_B_hand_full = hand_pose_from_C(t_B_C_full, R_B_C_full, meta)
    used_full = set()
    for mp in motion_hand_pos:
        d = np.linalg.norm(t_B_hand_full - mp, axis=1)
        used_full.add(int(np.argmin(d)))
    unused_full = [i for i in range(len(orig_all)) if i not in used_full]
    print(f"\nused waypoints: {len(used_full)}, unused in full dataset: {len(unused_full)}")

    # Pre-compute per-waypoint footprint on uncovered faces (restricted to speed up)
    stamp_pick = []
    stamp_pick_region = []
    uncov_mask = uncovered.copy()
    while uncov_mask.any() and unused_full:
        best = -1; best_gain = 0
        for wp in unused_full:
            fp = footprint_mask(t_B_C_full[wp], R_B_C_full[wp], centers_B)
            gain = areas[fp & uncov_mask].sum()
            if gain > best_gain:
                best_gain = gain; best = wp
        if best < 0 or best_gain == 0: break
        stamp_pick.append(best)
        stamp_pick_region.append(int(reg_all[best]))
        fp = footprint_mask(t_B_C_full[best], R_B_C_full[best], centers_B)
        uncov_mask &= ~fp
        unused_full.remove(best)
    print(f"stamps added: {len(stamp_pick)}")
    if stamp_pick_region:
        from collections import Counter
        c = Counter(stamp_pick_region)
        print(f"  stamp 출처 영역: {dict(c)}")

    # Step 5: final coverage = sweep + stamp
    final_mask = swept.copy()
    for wp in stamp_pick:
        fp = footprint_mask(t_B_C_full[wp], R_B_C_full[wp], centers_B)
        final_mask |= fp
    cov_final = areas[final_mask].sum()
    print(f"\n[sweep + stamp] coverage: {cov_final/total_area*100:.2f}% ({cov_final*1e4:.2f}/{total_area*1e4:.2f} cm²)")
    remain_uncov = (~final_mask).sum()
    print(f"  still uncovered: {remain_uncov} faces, {areas[~final_mask].sum()*1e4:.2f} cm²")

    print(f"\n== Summary ==")
    print(f"  Region {REGION_ID}: {K} waypoints, {len(face_in_reg)} faces, {total_area*1e4:.2f} cm²")
    print(f"  Sweep only:        {cov_sweep/total_area*100:5.2f}%  ({len(motion_hand)} wp)")
    print(f"  Sweep + {len(stamp_pick)} stamps: {cov_final/total_area*100:5.2f}%  ({len(motion_hand)+len(stamp_pick)} wp)")


if __name__ == "__main__":
    main()
