"""Mode-aware coverage/overlap evaluation (center=0.5mm, edge=1.0mm per waypoint).

Each motion waypoint is mapped to nearest source index to recover its
contact_mode, then the band thickness is applied per-waypoint.
Reports coverage %, overlap (2+ visits) / covered, mean multiplicity.
"""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, hand_pose_from_C, R_TOOL
from eval_contact_area import hand_to_C_transforms

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
MOTIONS = os.environ.get("MOTIONS",
    "motions/panda_full_dir0_fix_20261009_083611.csv,motions/panda_full_dir0_20261009_011001.csv").split(",")

PAD_CENTER = 0.0005
PAD_EDGE   = 0.0010


def load_motion(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    arr = np.array([[float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                     float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])]
                    for r in rows])
    pos = arr[:, :3]
    quat = arr[:, 3:7]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    R = Rot.from_quat(quat).as_matrix()
    return pos, R


def eval_motion(name, t_B_C_wp, R_B_C_wp, h_wp, work_centers, work_areas):
    T = len(t_B_C_wp); N = len(work_centers)
    visit = np.zeros(N, dtype=np.int32)
    chunk = 128
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tt = t_B_C_wp[s:e]; RR = R_B_C_wp[s:e]; hh = h_wp[s:e]
        delta = work_centers[None, :, :] - tt[:, None, :]
        p_C = np.einsum("cji,cnj->cni", RR, delta)
        xy2 = p_C[..., 0]**2 + p_C[..., 1]**2
        z = p_C[..., 2]
        hit = (xy2 <= R_TOOL**2) & (np.abs(z) <= hh[:, None])
        visit += hit.sum(axis=0)
    total = work_areas.sum()
    cov = work_areas[visit >= 1].sum()
    ov = work_areas[visit >= 2].sum()
    mean_mult = (visit[visit >= 1] * work_areas[visit >= 1]).sum() / cov if cov > 0 else 0
    print(f"\n=== {name}   T={T} ===")
    print(f"  coverage:          {cov/total*100:6.2f}%   ({cov*1e4:.2f} / {total*1e4:.2f} cm²)")
    print(f"  overlap/covered:   {ov/cov*100:6.2f}%   ({ov*1e4:.2f} cm²)")
    print(f"  overlap/feasible:  {ov/total*100:6.2f}%")
    print(f"  mean multiplicity: {mean_mult:.3f}x    max: {int(visit.max())}")


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
    print(f"Feasible faces: {feas.sum()}, area: {work_areas.sum()*1e4:.2f} cm²")

    # 2917 all-stamp baseline
    print(f"\n2917 waypoints: contact_modes center={int((cm_all==1).sum())} edge={int((cm_all==2).sum())}")
    eval_motion("ALL 2917 (stamp each)", t_B_C_all, R_B_C_all, h_all, work_centers, work_areas)

    for p in MOTIONS:
        p = p.strip()
        if not os.path.exists(p):
            print(f"\n[skip] {p}: not found")
            continue
        pos, R_B_hand = load_motion(p)
        t_B_C, R_B_C = hand_to_C_transforms(pos, R_B_hand)
        # map each motion wp to nearest source to recover contact_mode
        nn = np.argmin(np.linalg.norm(t_B_hand_all[None, :, :] - pos[:, None, :], axis=2), axis=1)
        h_wp = h_all[nn]
        eval_motion(os.path.basename(p), t_B_C, R_B_C, h_wp, work_centers, work_areas)


if __name__ == "__main__":
    main()
