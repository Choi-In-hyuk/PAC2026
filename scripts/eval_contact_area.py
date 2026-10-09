"""Contact-area coverage/overlap evaluation (Schneyer 2023 style).

For a given motion CSV, compute for each working-face center whether the
tool pad actually touches it. A face is 'touched' by a waypoint iff:
  - its center lies inside the tool-pad disc (radius R_TOOL) in C-frame xy
  - and the pad-to-surface vertical gap along C-frame z is at most PAD_INDENT

The pad plane is at z_C = 0 (contact origin); the pad face normal points along
-z_C (toward the surface). A face center at p_C = (x, y, z) with x^2+y^2 <= R^2
is touched iff -PAD_INDENT <= z <= PAD_INDENT (symmetric band around the pad plane).

Metrics (area-weighted):
  coverage_pct             =  covered area / feasible area
  overlap_frac_of_covered  = (>=2x area) / covered area
  overlap_frac_of_feasible = (>=2x area) / feasible area
  mean_multiplicity        =  area-weighted mean visit count over covered area
"""
import os, sys, csv, json, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot


DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door")
R_TOOL = 0.05       # m, pad disc radius
PAD_INDENT = 0.0005 # m, 0.5 mm pad indentation band (both sides)
TOOL_OFFSET = 0.11  # m, hand-origin -> tool pad center along hand +z
HAND_YAW = -np.pi / 4  # rz(-45 deg)


def load_mesh():
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    return M["vertices"], M["faces"], M["face_status"]


def face_centers_and_areas(V, F):
    v0 = V[F[:, 0]]; v1 = V[F[:, 1]]; v2 = V[F[:, 2]]
    centers = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return centers, areas


def load_motion(path):
    """Return hand positions (T,3), hand rotations (T,3,3) in robot BASE frame."""
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    px, py, pz = (hdr.index(f"target-POS_{c}") for c in "XYZ")
    rx, ry, rz, rw = (hdr.index(f"target-ROT_{c}") for c in "XYZW")
    hand_pos = data[:, [px, py, pz]]
    quat = data[:, [rx, ry, rz, rw]]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    R_B_hand = Rot.from_quat(quat).as_matrix()
    return hand_pos, R_B_hand


def hand_to_C_transforms(hand_pos, R_B_hand):
    """Convert hand pose to tool-C-frame pose in base frame.

    T_B_hand @ T_hand_C
    T_hand_C = inv(T_L_hand) @ T_L_C,  with T_L_hand = Rz(-45deg), T_L_C = diag(1,-1,-1) + [0,0,0.11]
    So T_hand_C in hand frame: R_hand_C = Rz(+45) @ diag(1,-1,-1); t_hand_C = Rz(+45) @ [0,0,0.11].
    Rz(+45) leaves the z-axis unchanged, so t_hand_C = [0, 0, 0.11].
    """
    cz, sz = np.cos(-HAND_YAW), np.sin(-HAND_YAW)
    R_hand_L = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])  # Rz(+45)
    R_L_C = np.diag([1.0, -1.0, -1.0])
    R_hand_C = R_hand_L @ R_L_C
    t_hand_C = np.array([0.0, 0.0, TOOL_OFFSET])

    # tool C-frame pose in B
    t_B_C = hand_pos + np.einsum("tij,j->ti", R_B_hand, t_hand_C)
    R_B_C = R_B_hand @ R_hand_C
    return t_B_C, R_B_C


def coverage_contact(hand_pos, R_B_hand, work_centers_B):
    """Return per-face visit count, counting a visit iff face center is in
    pad disc AND within +-PAD_INDENT along C-frame z of the pad plane.
    """
    t_B_C, R_B_C = hand_to_C_transforms(hand_pos, R_B_hand)
    T = len(hand_pos)
    N = len(work_centers_B)
    visit = np.zeros(N, dtype=np.int32)

    # chunk over waypoints to control memory (chunking faces also fine)
    chunk = 128
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tt = t_B_C[s:e]                         # (c,3)
        RR = R_B_C[s:e]                         # (c,3,3)
        # face centers expressed in each C frame:
        # p_C = R_B_C^T (p_B - t_B_C)
        delta = work_centers_B[None, :, :] - tt[:, None, :]       # (c, N, 3)
        p_C = np.einsum("cji,cnj->cni", RR, delta)                # (c, N, 3)
        xy2 = p_C[..., 0] ** 2 + p_C[..., 1] ** 2
        z = p_C[..., 2]
        hit = (xy2 <= R_TOOL * R_TOOL) & (np.abs(z) <= PAD_INDENT)
        visit += hit.sum(axis=0)
    return visit


def report(motion_path, work_centers_B, work_areas):
    hand_pos, R_B_hand = load_motion(motion_path)
    T = len(hand_pos)
    visit = coverage_contact(hand_pos, R_B_hand, work_centers_B)

    total_area = work_areas.sum()
    cov_mask = visit >= 1
    cov_area = work_areas[cov_mask].sum()
    overlap_area = work_areas[visit >= 2].sum()

    cov_pct = cov_area / total_area * 100
    overlap_pct_cov = (overlap_area / cov_area * 100) if cov_area > 0 else 0.0
    overlap_pct_tot = overlap_area / total_area * 100
    mean_mult = ((visit[cov_mask] * work_areas[cov_mask]).sum() / cov_area) if cov_area > 0 else 0.0
    max_mult = int(visit.max()) if len(visit) else 0

    one = work_areas[visit == 1].sum()
    two = work_areas[visit == 2].sum()
    tri = work_areas[visit >= 3].sum()

    print(f"\n=== {os.path.basename(motion_path)}   T={T} ===")
    print(f"  working area:       {total_area*1e4:7.2f} cm²")
    print(f"  coverage:           {cov_pct:6.2f}%   ({cov_area*1e4:.2f} cm²)")
    print(f"  overlap / covered:  {overlap_pct_cov:6.2f}%   ({overlap_area*1e4:.2f} cm²)")
    print(f"  overlap / feasible: {overlap_pct_tot:6.2f}%")
    print(f"  mean multiplicity:  {mean_mult:.3f} x    max: {max_mult}")
    print(f"  multiplicity breakdown of covered area:")
    if cov_area > 0:
        print(f"     1x:   {one*1e4:7.2f} cm²  ({one/cov_area*100:5.2f}%)")
        print(f"     2x:   {two*1e4:7.2f} cm²  ({two/cov_area*100:5.2f}%)")
        print(f"     >=3x: {tri*1e4:7.2f} cm²  ({tri/cov_area*100:5.2f}%)")
    return dict(name=os.path.basename(motion_path), T=T,
                cov_pct=cov_pct, overlap_pct_cov=overlap_pct_cov,
                overlap_pct_tot=overlap_pct_tot,
                mean_mult=mean_mult, max_mult=max_mult)


def main():
    V, F, status = load_mesh()
    centers_W, areas = face_centers_and_areas(V, F)
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW, t_BW = T_B_W[:3, :3], T_B_W[:3, 3]
    centers_B = (R_BW @ centers_W.T).T + t_BW

    work = (status == 1)
    work_centers_B = centers_B[work]
    work_areas = areas[work]
    print(f"Working faces: {work.sum()}  working area: {work_areas.sum()*1e4:.2f} cm²")
    print(f"Pad disc R = {R_TOOL*100:.0f} cm, indent band = {PAD_INDENT*1e3:.1f} mm (symmetric)")

    paths = sys.argv[1:]
    if not paths:
        line_order = sorted(glob.glob("motions/panda_line_order_*.csv"))
        hrchy = sorted(glob.glob("motions/panda_hrchy_gtsp_*.csv"))
        paths = []
        if line_order: paths.append(line_order[-1])
        if hrchy: paths.append(hrchy[-1])

    rows = [report(p, work_centers_B, work_areas) for p in paths if os.path.exists(p)]
    if len(rows) >= 2:
        print("\n--- side-by-side ---")
        print(f"{'motion':40s} {'T':>5s} {'cov%':>7s} {'ov/cov%':>8s} {'ov/tot%':>8s} {'mean':>6s} {'max':>5s}")
        for r in rows:
            print(f"{r['name']:40s} {r['T']:5d} {r['cov_pct']:7.2f} {r['overlap_pct_cov']:8.2f} "
                  f"{r['overlap_pct_tot']:8.2f} {r['mean_mult']:6.2f} {r['max_mult']:5d}")


if __name__ == "__main__":
    main()
