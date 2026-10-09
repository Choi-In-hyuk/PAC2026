"""Per-region coverage breakdown of a motion CSV.

For a given full-pipeline motion, assign each waypoint to the nearest
full-source tool C-frame pose, read its region_id, then measure sweep-style
coverage of each region's feasible faces (interpolated path between consecutive
waypoints of the same region; a region break triggers no interpolation).

Reports regions sorted by coverage gap (worst first), with:
  - K waypoints assigned to the region
  - total region area, covered area, gap area
  - number of uncovered faces (large & small)
  - a hint about likely cause (K too small, large faces, geometry edge, etc.)
"""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from eval_contact_area import R_TOOL, PAD_INDENT, hand_to_C_transforms
from region_set_cover import hand_pose_from_C

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
MOTION = os.environ.get("MOTION", "motions/panda_full_dir0_20261009_011001.csv")
INTERP_STEP = 0.002
GAP_THRESHOLD = 0.05  # m: if consecutive waypoints are farther than this, assume region break (no sweep)


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
    qc = (1 - u) * qa + u * qb
    return Rot.from_quat(qc / np.linalg.norm(qc)).as_matrix()


def load_motion_hand(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                      float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])]
                     for r in rows])


def main():
    print(f"Motion: {MOTION}")
    print(f"Data:   {DATA_DIR}\n")

    # Load source dataset
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"]); R_BW = T_B_W[:3, :3]; t_BW = T_B_W[:3, 3]

    valid = P["tool_pose_valid"] & P["accessibility"]
    orig = np.where(valid)[0]
    reg_all = P["region_ids"][orig]
    t_W_C_all = P["tool_positions"][orig]
    R_W_C_all = P["tool_rotations"][orig]
    t_B_C_all = (R_BW @ t_W_C_all.T).T + t_BW
    R_B_C_all = np.einsum("ij,kjl->kil", R_BW, R_W_C_all)
    t_B_hand_all, _ = hand_pose_from_C(t_B_C_all, R_B_C_all, meta)

    # Mesh
    F = M["faces"]; V = M["vertices"]
    face_region = M["face_region_ids"]
    feas = (M["face_status"] == 1)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    centers_W = (v0 + v1 + v2) / 3.0
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    centers_B = (R_BW @ centers_W.T).T + t_BW

    # Load motion → map each waypoint to nearest source point → derive region
    hand = load_motion_hand(MOTION)
    pos_B = hand[:, :3]
    quat = hand[:, 3:7]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R_B_hand = Rot.from_quat(quat).as_matrix()
    t_B_C_mot, R_B_C_mot = hand_to_C_transforms(pos_B, R_B_hand)

    # nearest source idx by hand position
    nn_src = np.argmin(np.linalg.norm(t_B_hand_all[None, :, :] - pos_B[:, None, :], axis=2), axis=1)
    nn_dist = np.linalg.norm(t_B_hand_all[nn_src] - pos_B, axis=1)
    waypoint_region = reg_all[nn_src]
    print(f"Motion waypoints: {len(pos_B)}, mean NN dist to source: {nn_dist.mean()*1000:.2f} mm, max: {nn_dist.max()*1000:.2f} mm")

    # Per region
    unique_regions = sorted(set(waypoint_region.tolist()))
    print(f"Regions touched by motion: {len(unique_regions)}\n")

    results = []
    for rid in unique_regions:
        wp_idx = np.where(waypoint_region == rid)[0]  # motion indices in this region
        K = len(wp_idx)
        f_in = np.where((face_region == rid) & feas)[0]
        if len(f_in) == 0:
            continue
        reg_centers = centers_B[f_in]
        reg_areas = areas[f_in]
        total = reg_areas.sum()

        covered = np.zeros(len(reg_centers), dtype=bool)
        # process waypoints in motion order; interpolate between consecutive
        # waypoints only if they are close (within GAP_THRESHOLD) — otherwise
        # treat as region-break (lift-off) and no sweep
        for i, mi in enumerate(wp_idx):
            covered |= footprint_mask(t_B_C_mot[mi], R_B_C_mot[mi], reg_centers)
            if i + 1 < len(wp_idx):
                mj = wp_idx[i + 1]
                # only interpolate if adjacent in the motion (mj == mi+1) and close
                if mj == mi + 1:
                    d = np.linalg.norm(t_B_C_mot[mj] - t_B_C_mot[mi])
                    if d <= GAP_THRESHOLD:
                        n_sub = max(1, int(np.ceil(d / INTERP_STEP)))
                        for s in range(1, n_sub):
                            u = s / n_sub
                            tm = (1 - u) * t_B_C_mot[mi] + u * t_B_C_mot[mj]
                            Rm = slerp(R_B_C_mot[mi], R_B_C_mot[mj], u)
                            covered |= footprint_mask(tm, Rm, reg_centers)
        cov_area = reg_areas[covered].sum()
        gap_area = total - cov_area
        cov_pct = cov_area / total * 100 if total > 0 else 100.0
        uncov_face_idx = np.where(~covered)[0]
        n_uncov = len(uncov_face_idx)
        area_uncov_sorted = np.sort(reg_areas[~covered])[::-1] if n_uncov else np.array([])
        largest = area_uncov_sorted[0] * 1e4 if n_uncov else 0.0

        # how many source waypoints exist in this region (upper bound of K)
        n_src_in_reg = int((reg_all == rid).sum())
        results.append(dict(rid=int(rid), K=K, n_src=n_src_in_reg,
                            total_cm2=total * 1e4, cov_cm2=cov_area * 1e4,
                            gap_cm2=gap_area * 1e4, cov_pct=cov_pct,
                            n_faces=len(reg_centers), n_uncov=n_uncov,
                            largest_uncov_cm2=largest))

    results.sort(key=lambda r: (r['cov_pct'], -r['gap_cm2']))

    # Print worst 30
    print(f"{'rid':>5s} {'K':>4s} {'nsrc':>5s} {'nFace':>6s} {'area':>8s} {'cov%':>6s} {'gap_cm2':>8s} {'nUncov':>7s} {'maxUncov':>9s}")
    worst = [r for r in results if r['cov_pct'] < 100]
    for r in worst[:30]:
        print(f"{r['rid']:5d} {r['K']:4d} {r['n_src']:5d} {r['n_faces']:6d} {r['total_cm2']:8.2f} "
              f"{r['cov_pct']:6.2f} {r['gap_cm2']:8.3f} {r['n_uncov']:7d} {r['largest_uncov_cm2']:9.3f}")

    print(f"\nTotal regions touched: {len(results)}, under 100%: {len(worst)}")
    tot_area = sum(r['total_cm2'] for r in results)
    tot_cov = sum(r['cov_cm2'] for r in results)
    tot_gap = sum(r['gap_cm2'] for r in results)
    print(f"Aggregate (regions-touched): area={tot_area:.2f} cm², cov={tot_cov:.2f} cm² ({tot_cov/tot_area*100:.2f}%), gap={tot_gap:.2f} cm²")

    # Also compute coverage against ALL feasible faces in all regions (including untouched)
    all_reg_ids = sorted(set(face_region[feas].tolist()))
    touched_set = {r['rid'] for r in results}
    untouched = [r for r in all_reg_ids if r not in touched_set and r >= 0]
    untouched_area = sum(areas[(face_region == rid) & feas].sum() for rid in untouched)
    print(f"Regions in mesh (feas): {len(all_reg_ids)}, untouched by motion: {len(untouched)}, their area: {untouched_area*1e4:.2f} cm²")


if __name__ == "__main__":
    main()
