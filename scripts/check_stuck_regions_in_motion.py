"""For each stuck region (sweep+stamp < 100% in pipeline log), check whether
the FULL 1236-waypoint motion (including interpolation) covers that region's
feasible faces.

If full-motion cov == pipeline cov -> dataset limit (nothing crosses)
If full-motion cov >  pipeline cov -> reset bug fix is incomplete (coverage from
  other regions' sweeps/stamps was NOT propagated into this region's swept_r)
"""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, hand_pose_from_C, R_TOOL
from eval_contact_area import hand_to_C_transforms

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
MOTION = os.environ.get("MOTION", "motions/panda_full_dir0_fix_20261009_083611.csv")
STUCK = [56, 100, 106, 124, 77, 92, 101]  # from pipeline log
INTERP_STEP = 0.002
PAD_CENTER = 0.0005
PAD_EDGE   = 0.0010


def footprint(t, R, centers, h):
    delta = centers - t
    p_C = (R.T @ delta.T).T
    xy2 = p_C[:, 0]**2 + p_C[:, 1]**2
    z = p_C[:, 2]
    return (xy2 <= R_TOOL**2) & (np.abs(z) <= h)


def slerp(R_a, R_b, u):
    qa = Rot.from_matrix(R_a).as_quat()
    qb = Rot.from_matrix(R_b).as_quat()
    if np.dot(qa, qb) < 0: qb = -qb
    qc = (1 - u)*qa + u*qb
    return Rot.from_quat(qc / np.linalg.norm(qc)).as_matrix()


def load_motion(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    arr = np.array([[float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                     float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])] for r in rows])
    pos = arr[:, :3]; quat = arr[:, 3:7]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
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
t_B_C_mot, R_B_C_mot = hand_to_C_transforms(pos, R_B_hand)
nn = np.argmin(np.linalg.norm(t_B_hand_all[None, :, :] - pos[:, None, :], axis=2), axis=1)
h_mot = h_all[nn]
reg_mot = reg_all[nn]
print(f"Motion {len(pos)} waypoints loaded")

for rid in STUCK:
    f_in = np.where((face_region == rid) & feas)[0]
    if len(f_in) == 0:
        continue
    reg_centers = centers_B[f_in]
    reg_areas = areas[f_in]
    total = reg_areas.sum()

    # (a) 영역 waypoint (motion 안의 reg_mot == rid) 로만 커버
    local_idx = np.where(reg_mot == rid)[0]
    cov_local = np.zeros(len(f_in), dtype=bool)
    for i in local_idx:
        cov_local |= footprint(t_B_C_mot[i], R_B_C_mot[i], reg_centers, h_mot[i])

    # (b) 전체 motion (모든 waypoint + 영역 내 인접 보간) 로 커버
    cov_full = np.zeros(len(f_in), dtype=bool)
    # waypoint 지점 footprint
    for i in range(len(pos)):
        cov_full |= footprint(t_B_C_mot[i], R_B_C_mot[i], reg_centers, h_mot[i])
    # 인접 waypoint 간 보간 (영역 전환 지점이어도 그냥 포함)
    for i in range(len(pos) - 1):
        d = np.linalg.norm(t_B_C_mot[i+1] - t_B_C_mot[i])
        if d > 0.05: continue  # 영역 전환 (lift-off 가정) 건너뜀
        n_sub = max(1, int(np.ceil(d / INTERP_STEP)))
        h_seg = max(h_mot[i], h_mot[i+1])
        for s in range(1, n_sub):
            u = s / n_sub
            tm = (1-u)*t_B_C_mot[i] + u*t_B_C_mot[i+1]
            Rm = slerp(R_B_C_mot[i], R_B_C_mot[i+1], u)
            cov_full |= footprint(tm, Rm, reg_centers, h_seg)

    print(f"\nrid={rid}  nFace={len(f_in)}  area={total*1e4:.3f}cm²")
    print(f"  local only (own-region waypoints in motion): {reg_areas[cov_local].sum()/total*100:5.2f}%")
    print(f"  full motion (all waypoints + intra-region interp): {reg_areas[cov_full].sum()/total*100:5.2f}%")
    gap_full = reg_areas[~cov_full].sum()
    print(f"  uncoverable even by full motion: {gap_full*1e4:.3f} cm² ({(~cov_full).sum()} faces)")
