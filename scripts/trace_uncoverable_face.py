"""For rid=101's uncovered face, find all 2917 waypoints that cover it
and check: (a) are they in the final motion? (b) if not, why did stamp miss them?
"""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, hand_pose_from_C, R_TOOL
from eval_contact_area import hand_to_C_transforms

DATA_DIR = "meshes/car_door_dir0"
MOTION = "motions/panda_full_dir0_fix_20261009_083611.csv"
RIDS = [101, 92, 77, 106, 100, 124, 56]
PAD_CENTER = 0.0005; PAD_EDGE = 0.0010


def footprint(t, R, centers, h):
    delta = centers - t
    p_C = (R.T @ delta.T).T
    xy2 = p_C[:, 0]**2 + p_C[:, 1]**2
    z = p_C[:, 2]
    return (xy2 <= R_TOOL**2) & (np.abs(z) <= h)


def load_motion(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    arr = np.array([[float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                     float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])] for r in rows])
    pos = arr[:, :3]; quat = arr[:, 3:7]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return pos, Rot.from_quat(quat).as_matrix()


P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

mask = P["tool_pose_valid"] & P["accessibility"]
orig = np.where(mask)[0]
cm_all = P["contact_modes"][orig]
h_all = np.where(cm_all == 1, PAD_CENTER, PAD_EDGE)
reg_all = P["region_ids"][orig]
t_B_C_all, R_B_C_all = compute_tool_C_poses_in_B(P, meta, orig)
t_B_hand_all, _ = hand_pose_from_C(t_B_C_all, R_B_C_all, meta)

centers_B, areas, status, face_region = mesh_centers_areas(M, meta)
feas = status == 1

pos, R_B_hand = load_motion(MOTION)
nn_motion = set(np.argmin(np.linalg.norm(t_B_hand_all[None, :, :] - pos[:, None, :], axis=2), axis=1).tolist())
print(f"Motion covers {len(nn_motion)} unique source waypoints out of {len(orig)}")

for rid in RIDS:
    f_in = np.where((face_region == rid) & feas)[0]
    if len(f_in) == 0:
        continue
    reg_centers = centers_B[f_in]
    reg_areas = areas[f_in]
    N = len(orig)

    # for each waypoint globally, which rid faces it covers
    cov_matrix = np.zeros((N, len(f_in)), dtype=bool)
    for li in range(N):
        cov_matrix[li] = footprint(t_B_C_all[li], R_B_C_all[li], reg_centers, h_all[li])

    # find faces NOT covered by any waypoint in motion
    cov_by_motion = np.zeros(len(f_in), dtype=bool)
    for li in nn_motion:
        cov_by_motion |= cov_matrix[li]

    # faces covered by SOME waypoint globally (2917) but not by motion
    cov_global = cov_matrix.any(axis=0)
    missed = cov_global & ~cov_by_motion   # face s/b coverable by 2917 but motion missed

    print(f"\nrid={rid}  nFace={len(f_in)}  area={reg_areas.sum()*1e4:.3f}cm²")
    print(f"  covered by motion: {cov_by_motion.sum()}/{len(f_in)}  area={reg_areas[cov_by_motion].sum()*1e4:.3f}")
    print(f"  coverable by 2917 (global):  {cov_global.sum()}/{len(f_in)}  area={reg_areas[cov_global].sum()*1e4:.3f}")
    print(f"  MISSED by motion but 2917 could cover: {missed.sum()} faces, area={reg_areas[missed].sum()*1e4:.3f}cm²")
    if missed.any():
        for fi in np.where(missed)[0]:
            helpers = np.where(cov_matrix[:, fi])[0]
            in_motion = [int(li) for li in helpers if li in nn_motion]
            not_motion = [int(li) for li in helpers if li not in nn_motion]
            helpers_regs = sorted({int(reg_all[li]) for li in helpers})
            print(f"    face{fi}  area={reg_areas[fi]*1e4:.3f}cm²  n_helpers={len(helpers)}  "
                  f"in_motion={len(in_motion)}  not_motion={len(not_motion)}  helper_regs={helpers_regs[:6]}")
