"""Full pipeline: per-region sweep (from selected waypoints) + stamp supplement.

For each region (one at a time):
  1. Take waypoints from MOTION_IN (region_cover selected) that belong to this region
  2. Run hrchy_gtsp on just those → sweep motion
  3. Compute sweep coverage of this region's faces
  4. For uncovered faces, greedy-pick stamps from UNUSED waypoints in the FULL
     source dataset (any region, excluding ones already used in sweep/stamp)
  5. Append stamps after the sweep for this region
Concat per-region motions in region-id order.
"""
import os, sys, csv, json, glob, subprocess
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from eval_contact_area import R_TOOL, PAD_INDENT, hand_to_C_transforms
from region_set_cover import rz, build_T, hand_pose_from_C

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
MOTION_IN = os.environ.get("MOTION_IN", "motions/panda_region_cover_dir0_20261008_222631.csv")
TAG = os.environ.get("TAG", "full_dir0")
N_MIN = int(os.environ.get("N_MIN", "15"))
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "20"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "20"))
INTERP_STEP = 0.002


PAD_INDENT_CENTER = 0.0005   # contact_mode==1
PAD_INDENT_EDGE   = 0.0010   # contact_mode==2


def footprint_mask(t, R, centers, h=PAD_INDENT):
    delta = centers - t
    p_C = (R.T @ delta.T).T
    xy2 = p_C[:,0]**2 + p_C[:,1]**2
    z = p_C[:,2]
    return (xy2 <= R_TOOL**2) & (np.abs(z) <= h)


def slerp(R_a, R_b, u):
    qa = Rot.from_matrix(R_a).as_quat()
    qb = Rot.from_matrix(R_b).as_quat()
    if np.dot(qa, qb) < 0: qb = -qb
    qc = (1-u)*qa + u*qb
    return Rot.from_quat(qc / np.linalg.norm(qc)).as_matrix()


def sweep_visit(t_B_C, R_B_C, centers, h_arr):
    """h_arr: (len(t_B_C),) per-waypoint PAD_INDENT. Interpolated segments use max of endpoints."""
    N = len(centers); mask = np.zeros(N, dtype=bool)
    for k in range(len(t_B_C)):
        mask |= footprint_mask(t_B_C[k], R_B_C[k], centers, h_arr[k])
        if k < len(t_B_C) - 1:
            d = np.linalg.norm(t_B_C[k+1] - t_B_C[k])
            n_sub = max(1, int(np.ceil(d / INTERP_STEP)))
            h_seg = max(h_arr[k], h_arr[k+1])
            for s in range(1, n_sub):
                u = s / n_sub
                tm = (1-u)*t_B_C[k] + u*t_B_C[k+1]
                Rm = slerp(R_B_C[k], R_B_C[k+1], u)
                mask |= footprint_mask(tm, Rm, centers, h_seg)
    return mask


def nn_order(poses_xyz):
    n = len(poses_xyz)
    if n <= 1: return list(range(n))
    visited = [0]; rem = set(range(1, n))
    while rem:
        cur = poses_xyz[visited[-1]]
        best = min(rem, key=lambda i: np.linalg.norm(poses_xyz[i] - cur))
        visited.append(best); rem.remove(best)
    return visited


def run_hrchy(region_poses_hand, region_id, time_limit, num_samples):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    surf_path = f"surfaces/panda_{ts}_region{region_id}.csv"
    with open(surf_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time','target-POS_X','target-POS_Y','target-POS_Z',
                    'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W'])
        for i, p in enumerate(region_poses_hand):
            w.writerow([i, *p])
    for name in ("joint_gtsp_high_level.tour", "joint_gtsp_low_level.tour"):
        f = f"GLKH/{name}"
        if os.path.exists(f): os.remove(f)
    env = os.environ.copy()
    env["SURFACE_PATTERN"] = surf_path
    env["TIME_LIMIT"] = str(time_limit)
    env["NUM_SAMPLES"] = str(num_samples)
    hard = time_limit * 3 + 180
    try:
        subprocess.run(["python", "scripts/run_cardoor_hrchy_only.py"],
                       env=env, timeout=hard,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        return None
    produced = sorted(glob.glob("motions/panda_hrchy_gtsp_*.csv"), key=os.path.getmtime)
    return produced[-1] if produced else None


def load_motion_hand(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r['target-POS_X']),float(r['target-POS_Y']),float(r['target-POS_Z']),
                      float(r['target-ROT_X']),float(r['target-ROT_Y']),float(r['target-ROT_Z']),float(r['target-ROT_W'])] for r in rows])


def main():
    print(f"=== Full pipeline ===")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"MOTION_IN = {MOTION_IN}")
    print(f"N_MIN={N_MIN}, TIME_LIMIT={TIME_LIMIT}s, NUM_SAMPLES={NUM_SAMPLES}\n")

    # ---- Load source npz (full 2917) ----
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW = T_B_W[:3,:3]; t_BW = T_B_W[:3,3]

    mask = P["tool_pose_valid"] & P["accessibility"]
    orig_all = np.where(mask)[0]
    reg_all = P["region_ids"][orig_all]
    cm_all = P["contact_modes"][orig_all]
    h_all = np.where(cm_all == 1, PAD_INDENT_CENTER, PAD_INDENT_EDGE)
    t_W_C_all = P["tool_positions"][orig_all]
    R_W_C_all = P["tool_rotations"][orig_all]
    t_B_C_all = (R_BW @ t_W_C_all.T).T + t_BW
    R_B_C_all = np.einsum("ij,kjl->kil", R_BW, R_W_C_all)
    t_B_hand_all, R_B_hand_all = hand_pose_from_C(t_B_C_all, R_B_C_all, meta)
    quat_all = Rot.from_matrix(R_B_hand_all).as_quat()
    hand_pose7_all = np.hstack([t_B_hand_all, quat_all])  # (N_full, 7)
    N_full = len(orig_all)
    print(f"source dataset: {N_full} waypoints, {len(np.unique(reg_all))} regions")

    # ---- Load selected waypoints from MOTION_IN, map each to the nearest full-source index ----
    sel_hand = load_motion_hand(MOTION_IN)
    sel_pos = sel_hand[:, :3]
    sel_to_full = []
    for p in sel_pos:
        d = np.linalg.norm(t_B_hand_all - p, axis=1)
        sel_to_full.append(int(np.argmin(d)))
    sel_to_full = np.array(sel_to_full)
    sel_regions = reg_all[sel_to_full]
    print(f"selected (input): {len(sel_hand)} waypoints across {len(np.unique(sel_regions))} regions\n")

    # ---- Mesh ----
    F = M["faces"]; V = M["vertices"]
    face_region = M["face_region_ids"]
    face_status = M["face_status"]
    feas = (face_status == 1)
    v0 = V[F[:,0]]; v1 = V[F[:,1]]; v2 = V[F[:,2]]
    centers_W = (v0+v1+v2)/3.0
    areas_all_f = 0.5*np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    centers_B_all_f = (R_BW @ centers_W.T).T + t_BW
    total_feasible_area = areas_all_f[feas].sum()
    print(f"total feasible face area: {total_feasible_area*1e4:.2f} cm²\n")

    unique_regions = np.unique(sel_regions)   # process only regions that have selected waypoints
    print(f"regions to process: {len(unique_regions)}\n")

    final_motion = []     # list of (pose7, kind, region)
    used_global = set()   # full-source indices used anywhere
    region_summary = []
    global_face_covered = np.zeros(len(F), dtype=bool)   # mesh 전체 face 커버 상태 (영역 간 공유)

    for rid in unique_regions:
        # ---- 1. Collect selected waypoints in this region ----
        in_sel = np.where(sel_regions == rid)[0]
        if len(in_sel) == 0: continue
        full_abs = sel_to_full[in_sel]              # absolute idx into orig_all
        poses_hand_r = hand_pose7_all[full_abs]
        K = len(poses_hand_r)

        # ---- 2. Region faces ----
        f_in = np.where((face_region == rid) & feas)[0]
        if len(f_in) == 0:
            # no faces to cover, just append sweep in NN order
            order = nn_order(t_B_C_all[full_abs])
            for i in order:
                final_motion.append((hand_pose7_all[int(full_abs[i])], 'sweep', int(rid)))
                used_global.add(int(full_abs[i]))
            region_summary.append((int(rid), K, len(order), 0, 0.0, 0.0, 0.0, 'nofaces'))
            print(f"  region {int(rid):4d}: K={K}, no faces, NN")
            continue
        centers_B_r = centers_B_all_f[f_in]
        areas_r = areas_all_f[f_in]
        total_r = areas_r.sum()

        # ---- 3. Sweep ----
        if K >= N_MIN:
            mp = run_hrchy(poses_hand_r, int(rid), TIME_LIMIT, NUM_SAMPLES)
            if mp is None:
                sweep_order_local = nn_order(t_B_C_all[full_abs])
                how = 'nn_fallback'
            else:
                out_hand = load_motion_hand(mp)
                sweep_order_local = []
                taken = set()
                for p in out_hand:
                    d = np.linalg.norm(poses_hand_r[:, :3] - p[:3], axis=1)
                    for idx in np.argsort(d):
                        if idx not in taken:
                            sweep_order_local.append(int(idx)); taken.add(int(idx)); break
                os.remove(mp)
                how = 'hrchy'
        else:
            sweep_order_local = nn_order(t_B_C_all[full_abs])
            how = 'nn_small'

        sweep_abs = full_abs[sweep_order_local]   # abs idx list in sweep order

        # ---- 4. Measure sweep coverage ----
        t_B_C_sw = t_B_C_all[sweep_abs]
        R_B_C_sw = R_B_C_all[sweep_abs]
        h_sw = h_all[sweep_abs]
        swept_r = sweep_visit(t_B_C_sw, R_B_C_sw, centers_B_r, h_sw)
        swept_r |= global_face_covered[f_in]   # 이전 영역들이 지나가며 덮은 face 반영
        cov_sweep = areas_r[swept_r].sum()

        # ---- 5. Stamp from unused in FULL dataset ----
        stamp_pick = []
        uncov = ~swept_r
        if uncov.any():
            # exclude ones already used globally AND currently in sweep_abs
            unused = [i for i in range(N_full) if i not in used_global and i not in set(sweep_abs.tolist())]
            uncov_mask = uncov.copy()
            while uncov_mask.any() and unused:
                best = -1; best_gain = 0
                for wp in unused:
                    fp = footprint_mask(t_B_C_all[wp], R_B_C_all[wp], centers_B_r, h_all[wp])
                    gain = areas_r[fp & uncov_mask].sum()
                    if gain > best_gain:
                        best_gain = gain; best = wp
                if best < 0 or best_gain == 0: break
                stamp_pick.append(best)
                fp = footprint_mask(t_B_C_all[best], R_B_C_all[best], centers_B_r, h_all[best])
                uncov_mask &= ~fp
                unused.remove(best)
        final_mask = swept_r.copy()
        for wp in stamp_pick:
            final_mask |= footprint_mask(t_B_C_all[wp], R_B_C_all[wp], centers_B_r, h_all[wp])
        cov_final = areas_r[final_mask].sum()

        # ---- 6. Append to global motion ----
        for ai in sweep_abs:
            final_motion.append((hand_pose7_all[int(ai)], 'sweep', int(rid)))
            used_global.add(int(ai))
        for wp in stamp_pick:
            final_motion.append((hand_pose7_all[int(wp)], 'stamp', int(rid)))
            used_global.add(int(wp))

        # 전역 face 커버 마스크 업데이트 (sweep 보간 + stamp 지점)
        sw_mask_all = sweep_visit(t_B_C_all[sweep_abs], R_B_C_all[sweep_abs], centers_B_all_f, h_all[sweep_abs])
        global_face_covered |= (sw_mask_all & feas)
        for wp in stamp_pick:
            fp = footprint_mask(t_B_C_all[int(wp)], R_B_C_all[int(wp)], centers_B_all_f, h_all[int(wp)])
            global_face_covered |= (fp & feas)

        region_summary.append((int(rid), K, len(sweep_abs), len(stamp_pick),
                                total_r*1e4, cov_sweep*1e4, cov_final*1e4, how))
        print(f"  region {int(rid):4d}: K={K:3d} sweep={len(sweep_abs):3d} stamp={len(stamp_pick):2d}  cov {cov_sweep/total_r*100:5.1f}% → {cov_final/total_r*100:5.1f}%  ({how})")

    # ---- Global cleanup stamp: 전체 feasible 미커버 face 를 unused waypoint 로 메꿈 ----
    feas_idx = np.where(feas)[0]
    uncov_global = feas & ~global_face_covered
    if uncov_global.any():
        print(f"\n== Global cleanup stamp ==")
        print(f"  uncovered feasible faces before cleanup: {uncov_global.sum()}  area={areas_all_f[uncov_global].sum()*1e4:.3f} cm²")
        unused = list(range(N_full))   # cleanup 은 used_global 무시 (모션 중복 허용)
        uncov_mask = uncov_global.copy()
        added = 0
        while uncov_mask.any() and unused:
            best = -1; best_gain = 0.0
            for wp in unused:
                fp = footprint_mask(t_B_C_all[wp], R_B_C_all[wp], centers_B_all_f, h_all[wp])
                gain = areas_all_f[fp & uncov_mask].sum()
                if gain > best_gain:
                    best_gain = gain; best = wp
            if best < 0 or best_gain == 0: break
            # append to motion (assign to region = -1 marker)
            final_motion.append((hand_pose7_all[int(best)], 'cleanup', -1))
            used_global.add(int(best))
            fp = footprint_mask(t_B_C_all[best], R_B_C_all[best], centers_B_all_f, h_all[best])
            uncov_mask &= ~fp
            global_face_covered |= (fp & feas)
            unused.remove(best)
            added += 1
        remaining = uncov_mask.sum()
        print(f"  cleanup stamps added: {added}")
        print(f"  uncovered after cleanup: {remaining}  area={areas_all_f[uncov_mask].sum()*1e4:.3f} cm²")

    # Save
    os.makedirs("motions", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"motions/panda_{TAG}_{ts}.csv"
    joint_cols = [f"panda-panda_joint{i}" for i in range(1, 8)]
    fieldnames = ['time'] + joint_cols + ['target-POS_X','target-POS_Y','target-POS_Z',
                                           'target-ROT_X','target-ROT_Y','target-ROT_Z','target-ROT_W']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for i, (p, kind, rid) in enumerate(final_motion):
            row = [float(i)] + [0.0]*7 + list(p)
            w.writerow(dict(zip(fieldnames, row)))
    print(f"\nSaved: {out}  ({len(final_motion)} waypoints)")

    tot_K = sum(s[1] for s in region_summary)
    tot_sw = sum(s[2] for s in region_summary)
    tot_st = sum(s[3] for s in region_summary)
    tot_area = sum(s[4] for s in region_summary)
    cov_sw = sum(s[5] for s in region_summary)
    cov_fin = sum(s[6] for s in region_summary)
    print(f"\n== Totals ==")
    print(f"  input selected: {tot_K}, sweep used: {tot_sw}, stamps added: {tot_st}")
    print(f"  region face area:  {tot_area:.2f} cm²")
    print(f"  sweep-only cov:    {cov_sw/tot_area*100:5.2f}% ({cov_sw:.2f} cm²)")
    print(f"  sweep + stamp cov: {cov_fin/tot_area*100:5.2f}% ({cov_fin:.2f} cm²)")


if __name__ == "__main__":
    main()
