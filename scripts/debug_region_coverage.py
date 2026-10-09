"""Debug one failing region: why does region_set_cover claim coverage but
run_full_pipeline sweep says 0%?

For a given region id, show:
  - all source waypoints in that region (n_src)
  - for each, which region faces it contacts (per contact_faces_for_waypoint)
  - which waypoints the region_cover selection picked
  - which of those are present in the final motion CSV
  - re-evaluate the final motion's waypoints for that region with footprint_mask
"""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import (compute_tool_C_poses_in_B, mesh_centers_areas,
                               contact_faces_for_waypoint, hand_pose_from_C,
                               R_TOOL, PAD_INDENT)
from eval_contact_area import hand_to_C_transforms

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
MOTION_IN = os.environ.get("MOTION_IN", "motions/panda_region_cover_dir0_20261008_222631.csv")
MOTION_FULL = os.environ.get("MOTION_FULL", "motions/panda_full_dir0_20261009_011001.csv")
RIDS = [int(x) for x in os.environ.get("RIDS", "49,41,72,66,104").split(",")]


def load_motion_hand(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                      float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])]
                     for r in rows])


def footprint_mask(t, R, centers):
    delta = centers - t
    p_C = (R.T @ delta.T).T
    xy2 = p_C[:, 0] ** 2 + p_C[:, 1] ** 2
    z = p_C[:, 2]
    return (xy2 <= R_TOOL ** 2) & (np.abs(z) <= PAD_INDENT)


def main():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    mask = P["tool_pose_valid"] & P["accessibility"]
    idx = np.where(mask)[0]
    reg = P["region_ids"][idx]
    t_B_C, R_B_C = compute_tool_C_poses_in_B(P, meta, idx)
    t_B_hand, _ = hand_pose_from_C(t_B_C, R_B_C, meta)
    centers_B, areas, face_status, face_region = mesh_centers_areas(M, meta)
    feas = face_status == 1

    # map motion CSVs to source indices (via hand position NN)
    def map_motion(path):
        h = load_motion_hand(path)
        pos = h[:, :3]
        nn = np.argmin(np.linalg.norm(t_B_hand[None, :, :] - pos[:, None, :], axis=2), axis=1)
        return nn

    sel_src = map_motion(MOTION_IN)      # which source idx each region_cover output refers to
    full_src = map_motion(MOTION_FULL)   # same for full pipeline

    sel_set = set(sel_src.tolist())
    full_set = set(full_src.tolist())

    for rid in RIDS:
        print(f"\n{'='*70}\nREGION {rid}")
        wp_in_src = np.where(reg == rid)[0]   # local src indices in this region
        f_in = np.where((face_region == rid) & feas)[0]
        if len(wp_in_src) == 0 or len(f_in) == 0:
            print(f"  no waypoints ({len(wp_in_src)}) or no faces ({len(f_in)}); skip")
            continue
        reg_centers = centers_B[f_in]
        reg_areas = areas[f_in]
        print(f"  n_src={len(wp_in_src)}  n_faces={len(f_in)}  area={reg_areas.sum()*1e4:.3f} cm²")

        # per-waypoint contact against region faces (REGION SET COVER rule)
        contact_patches = []
        for li in wp_in_src:
            m = contact_faces_for_waypoint(t_B_C[li], R_B_C[li], reg_centers)
            contact_patches.append(m)
            z_vals = (R_B_C[li].T @ (reg_centers - t_B_C[li]).T).T[:, 2]
            xy2 = ((R_B_C[li].T @ (reg_centers - t_B_C[li]).T).T[:, 0]**2 +
                   (R_B_C[li].T @ (reg_centers - t_B_C[li]).T).T[:, 1]**2)
            in_disc = (xy2 <= R_TOOL**2).sum()
            in_both = m.sum()
            picked_by_region_cover = li in sel_set
            picked_by_full = li in full_set
            tag = []
            if picked_by_region_cover: tag.append("REGION_COVER_PICK")
            if picked_by_full: tag.append("IN_FULL_MOTION")
            print(f"    src_idx={li:4d}  faces_in_disc={in_disc:3d}  faces_in_band(PAD_INDENT)={in_both:3d}  "
                  f"min|z|={np.min(np.abs(z_vals))*1000:.2f}mm  area_cov={reg_areas[m].sum()*1e4:.3f}cm²  "
                  f"[{','.join(tag)}]")

        # union of all source waypoints' contact (upper bound achievable)
        union = np.zeros(len(f_in), dtype=bool)
        for m in contact_patches: union |= m
        print(f"  UPPER BOUND (union of all {len(wp_in_src)} src waypoints): "
              f"{reg_areas[union].sum()*1e4:.3f} cm² = {reg_areas[union].sum()/reg_areas.sum()*100:.2f}%")

        # how region_set_cover claimed coverage (its own greedy)
        picks_rc = [i for i, li in enumerate(wp_in_src) if li in sel_set]
        rc_cov = np.zeros(len(f_in), dtype=bool)
        for i in picks_rc: rc_cov |= contact_patches[i]
        print(f"  region_cover selected: {len(picks_rc)} waypoints, "
              f"stop-contact cov = {reg_areas[rc_cov].sum()*1e4:.3f} cm² "
              f"= {reg_areas[rc_cov].sum()/reg_areas.sum()*100:.2f}%")

        # what's in final motion
        picks_full = [i for i, li in enumerate(wp_in_src) if li in full_set]
        print(f"  final motion contains: {len(picks_full)} waypoints for this region")


if __name__ == "__main__":
    main()
