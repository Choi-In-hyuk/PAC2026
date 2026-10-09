"""2mm 격자 샘플링 기반 커버리지/중복 평가.

Feasible face 들을 각 삼각형의 넓이에 비례해서 2mm 간격 균일 샘플로 쪼갠 뒤,
waypoint 각각에 대해 샘플이 공구 디스크 ∩ 패드밴드 안에 들어오는지 집계.

coverage = 커버된 샘플 가중치 합 / 전체 샘플 가중치 합
multiplicity = 각 샘플이 몇 번 커버됐나
"""
import os, sys, csv, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, hand_pose_from_C, R_TOOL
from eval_contact_area import hand_to_C_transforms

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
MOTIONS = os.environ.get("MOTIONS",
    "motions/panda_full_dir0_v2_20261009_090106.csv,motions/panda_full_dir0_fix_20261009_083611.csv").split(",")
SAMPLE_SPACING = float(os.environ.get("SAMPLE_SPACING", "0.002"))   # 2 mm
PAD_CENTER = 0.0005
PAD_EDGE   = 0.0010


def sample_triangles(V, F, spacing):
    """Triangle 하나당 (대략) spacing 간격으로 균일 샘플 생성.
    간단한 바리센트릭 균일 샘플: 각 삼각형의 (sqrt(area)/spacing)^2 개 샘플, area/N 가중치."""
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    e1 = v1 - v0; e2 = v2 - v0
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    all_pts = []; all_w = []; all_face = []
    for i in range(len(F)):
        a = areas[i]
        # 샘플 수: 삼각형을 2mm 격자로 덮을 때 들어가는 샘플 개수 근사
        n = max(1, int(np.ceil(a / (spacing * spacing))))
        # 바리센트릭 균일 샘플 (낮은 n 은 중심 1개, 큰 삼각형은 n 개 균일)
        if n == 1:
            u = np.array([1/3.]); v = np.array([1/3.])
        else:
            # k×k 격자, k = ceil(sqrt(n))
            k = int(np.ceil(np.sqrt(n)))
            uu, vv = np.meshgrid(np.linspace(1/(2*k), 1-1/(2*k), k),
                                 np.linspace(1/(2*k), 1-1/(2*k), k))
            uu = uu.ravel(); vv = vv.ravel()
            keep = (uu + vv) <= 1
            u = uu[keep]; v = vv[keep]
            if len(u) == 0:
                u = np.array([1/3.]); v = np.array([1/3.])
        pts = v0[i] + np.outer(u, e1[i]) + np.outer(v, e2[i])
        w = np.full(len(u), a / len(u))
        all_pts.append(pts); all_w.append(w); all_face.append(np.full(len(u), i))
    return (np.concatenate(all_pts, axis=0),
            np.concatenate(all_w),
            np.concatenate(all_face))


def load_motion(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    arr = np.array([[float(r['target-POS_X']), float(r['target-POS_Y']), float(r['target-POS_Z']),
                     float(r['target-ROT_X']), float(r['target-ROT_Y']), float(r['target-ROT_Z']), float(r['target-ROT_W'])] for r in rows])
    pos = arr[:, :3]; quat = arr[:, 3:7]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return pos, Rot.from_quat(quat).as_matrix()


def eval_motion(name, t_B_C, R_B_C, h_wp, pts_B, weights):
    T = len(t_B_C); N = len(pts_B)
    visit = np.zeros(N, dtype=np.int32)
    chunk = 32
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tt = t_B_C[s:e]; RR = R_B_C[s:e]; hh = h_wp[s:e]
        delta = pts_B[None, :, :] - tt[:, None, :]
        p_C = np.einsum("cji,cnj->cni", RR, delta)
        xy2 = p_C[..., 0]**2 + p_C[..., 1]**2
        z = p_C[..., 2]
        hit = (xy2 <= R_TOOL**2) & (np.abs(z) <= hh[:, None])
        visit += hit.sum(axis=0).astype(np.int32)
    total = weights.sum()
    cov = weights[visit >= 1].sum()
    ov = weights[visit >= 2].sum()
    mean_mult = (visit[visit >= 1] * weights[visit >= 1]).sum() / cov if cov > 0 else 0
    print(f"\n=== {name}   T={T}, N_samples={N} ===")
    print(f"  coverage:        {cov/total*100:6.3f}%   ({cov*1e4:.2f} / {total*1e4:.2f} cm²)")
    print(f"  overlap/cov:     {ov/cov*100:6.2f}%")
    print(f"  overlap/feas:    {ov/total*100:6.2f}%")
    print(f"  mean mult:       {mean_mult:.3f}x    max: {int(visit.max())}")
    bins = [(0,0,'0x'),(1,1,'1x'),(2,2,'2x'),(3,3,'3x'),(4,5,'4-5x'),(6,8,'6-8x'),(9,15,'9-15x'),(16,999,'16+x')]
    print(f"  area distribution:")
    for lo, hi, lab in bins:
        m = (visit>=lo) & (visit<=hi)
        a = weights[m].sum()
        print(f"    {lab:>6}: {a*1e4:7.2f} cm² ({a/total*100:5.2f}% of feas, {a/cov*100 if cov>0 else 0:5.2f}% of cov)")


def main():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"]); R_BW = T_B_W[:3,:3]; t_BW = T_B_W[:3,3]

    mask = P["tool_pose_valid"] & P["accessibility"]
    orig = np.where(mask)[0]
    cm_all = P["contact_modes"][orig]
    h_all = np.where(cm_all == 1, PAD_CENTER, PAD_EDGE)
    t_B_C_all, R_B_C_all = compute_tool_C_poses_in_B(P, meta, orig)
    t_B_hand_all, _ = hand_pose_from_C(t_B_C_all, R_B_C_all, meta)

    V = M["vertices"]; F = M["faces"]; status = M["face_status"]
    feas = status == 1
    F_feas = F[feas]
    print(f"Feasible faces: {feas.sum()}")

    # Sample feasible faces in W then convert to B
    pts_W, weights, _ = sample_triangles(V, F_feas, SAMPLE_SPACING)
    pts_B = (R_BW @ pts_W.T).T + t_BW
    print(f"Grid samples @ {SAMPLE_SPACING*1000:.1f} mm: {len(pts_B)}, total weight = {weights.sum()*1e4:.2f} cm²")

    # 2917 baseline
    eval_motion("ALL 2917 (stamp each)", t_B_C_all, R_B_C_all, h_all, pts_B, weights)

    for p in MOTIONS:
        p = p.strip()
        if not os.path.exists(p):
            print(f"[skip] {p}: not found"); continue
        pos, R_B_hand = load_motion(p)
        t_B_C, R_B_C = hand_to_C_transforms(pos, R_B_hand)
        nn = np.argmin(np.linalg.norm(t_B_hand_all[None, :, :] - pos[:, None, :], axis=2), axis=1)
        h_wp = h_all[nn]
        eval_motion(os.path.basename(p), t_B_C, R_B_C, h_wp, pts_B, weights)


if __name__ == "__main__":
    main()
