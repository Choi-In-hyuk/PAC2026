"""Compare v2 (algorithm) vs zigzag vs NN:
  - coverage % (mode-aware)
  - overlap/feas %
  - mean multiplicity
  - joint movement sum (ds)
  - num waypoints
"""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, hand_pose_from_C, R_TOOL
from eval_contact_area import hand_to_C_transforms

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
PAD_CENTER = 0.0005; PAD_EDGE = 0.0010

MOTIONS = [
    ("v2 (algorithm)", "motions/panda_full_dir0_v2_withIK.csv"),
    ("zigzag (all 2917)", "motions/panda_zigzag_dir0_20261009_142728.csv"),
    ("NN (all 2917)",     "motions/panda_nn_dir0_20261009_142759.csv"),
]


def load_motion(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys())
    joint_cols = [c for c in header if c.startswith("panda-")]
    arr = []
    joints = []
    for r in rows:
        arr.append([float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                    float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])])
        joints.append([float(r[c]) for c in joint_cols])
    arr = np.array(arr); joints = np.array(joints)
    return arr, joints


def joint_ds(joints):
    if len(joints) < 2: return 0.0
    diffs = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    return float(diffs.sum())


def main():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    mask = P["tool_pose_valid"] & P["accessibility"]
    orig = np.where(mask)[0]
    cm_all = P["contact_modes"][orig]
    h_all = np.where(cm_all == 1, PAD_CENTER, PAD_EDGE)
    t_B_C_all, R_B_C_all = compute_tool_C_poses_in_B(P, meta, orig)
    t_B_hand_all, _ = hand_pose_from_C(t_B_C_all, R_B_C_all, meta)

    centers_B, areas, status, _ = mesh_centers_areas(M, meta)
    feas = status == 1
    work_centers = centers_B[feas]; work_areas = areas[feas]
    total = work_areas.sum()

    print(f"Feasible area: {total*1e4:.2f} cm²\n")
    print(f"{'motion':<22s} {'T':>5s} {'cov%':>7s} {'ov/feas%':>9s} {'ov/cov%':>8s} {'mean':>6s} {'max':>5s} {'ds(rad)':>9s}")
    print("-" * 90)

    for label, path in MOTIONS:
        if not os.path.exists(path):
            print(f"{label:<22s}  SKIP (not found: {path})")
            continue
        arr, joints = load_motion(path)
        pos = arr[:, :3]; quat = arr[:, 3:7]
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        R_B_hand = Rot.from_quat(quat).as_matrix()
        t_B_C, R_B_C = hand_to_C_transforms(pos, R_B_hand)
        nn = np.argmin(np.linalg.norm(t_B_hand_all[None, :, :] - pos[:, None, :], axis=2), axis=1)
        h_wp = h_all[nn]

        T = len(pos); N = len(work_centers)
        visit = np.zeros(N, dtype=np.int32)
        chunk = 128
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            tt = t_B_C[s:e]; RR = R_B_C[s:e]; hh = h_wp[s:e]
            delta = work_centers[None, :, :] - tt[:, None, :]
            p_C = np.einsum("cji,cnj->cni", RR, delta)
            xy2 = p_C[..., 0]**2 + p_C[..., 1]**2
            z = p_C[..., 2]
            hit = (xy2 <= R_TOOL**2) & (np.abs(z) <= hh[:, None])
            visit += hit.sum(axis=0).astype(np.int32)

        cov = work_areas[visit >= 1].sum()
        ov = work_areas[visit >= 2].sum()
        mean_mult = (visit[visit>=1] * work_areas[visit>=1]).sum() / cov if cov>0 else 0
        ds = joint_ds(joints)
        print(f"{label:<22s} {T:5d} {cov/total*100:7.2f} {ov/total*100:9.2f} {ov/cov*100:8.2f} {mean_mult:6.2f} {int(visit.max()):5d} {ds:9.3f}")


if __name__ == "__main__":
    main()
