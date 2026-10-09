"""Region-aware sweep evaluation.

Within the same region, consecutive waypoints are treated as a swept pass:
their tool contact between them is interpolated. Between different regions
the tool is assumed lifted (no interpolation).

Region id for each motion waypoint is recovered by nearest-neighbor matching
to the original surface_points.npz.
"""
import os, sys, csv, json, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
from eval_contact_area import (DATA_DIR, R_TOOL, PAD_INDENT, HAND_YAW, TOOL_OFFSET,
                                load_mesh, face_centers_and_areas, load_motion,
                                hand_to_C_transforms)


INTERP_STEP = 0.002  # 2 mm substeps inside a sweep segment


def load_source_waypoints():
    """Return (tool_pos_B (K,3), region_ids (K,)) for the visitable originals."""
    from eval_contact_area import DATA_DIR as D
    P = np.load(os.path.join(D, "surface_points.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    mask = P["tool_pose_valid"] & P["accessibility"]
    t_W_C = P["tool_positions"][mask]
    pos_B = (T_B_W[:3, :3] @ t_W_C.T).T + T_B_W[:3, 3]
    return pos_B, P["region_ids"][mask]


def match_region(tool_pos_B_motion, src_pos_B, src_region):
    """For each motion waypoint (tool-contact position in B), assign the
    region id of the nearest source waypoint."""
    # pairwise distances
    d = np.linalg.norm(tool_pos_B_motion[:, None, :] - src_pos_B[None, :, :], axis=2)
    nn = np.argmin(d, axis=1)
    return src_region[nn]


def quaternion_slerp(R_a, R_b, u):
    """Rotation.slerp isn't a staticmethod in all scipy versions; roll our own
    via quaternions. Returns a rotation matrix."""
    qa = Rot.from_matrix(R_a).as_quat()
    qb = Rot.from_matrix(R_b).as_quat()
    if np.dot(qa, qb) < 0:
        qb = -qb
    qc = (1 - u) * qa + u * qb
    qc = qc / np.linalg.norm(qc)
    return Rot.from_quat(qc).as_matrix()


def accumulate_visits(t_B_C, R_B_C, regions, work_centers_B):
    """Advance along the motion; within same region, interpolate between
    consecutive poses to produce substeps. Face center is visited once per
    swept segment (not once per substep)."""
    N = len(work_centers_B)
    visit = np.zeros(N, dtype=np.int32)

    def footprint_mask(t, R):
        delta = work_centers_B - t
        p_C = (R.T @ delta.T).T
        xy2 = p_C[:, 0]**2 + p_C[:, 1]**2
        z = p_C[:, 2]
        return (xy2 <= R_TOOL * R_TOOL) & (np.abs(z) <= PAD_INDENT)

    T = len(t_B_C)
    i = 0
    while i < T:
        # start new sweep at i, extend as long as region stays the same
        j = i
        while j + 1 < T and regions[j + 1] == regions[i]:
            j += 1
        # segment indices [i..j]; sweep is union of pose i, i+1, ..., j
        # plus interpolated substeps between consecutive poses in that range
        seg_mask = np.zeros(N, dtype=bool)
        for k in range(i, j + 1):
            seg_mask |= footprint_mask(t_B_C[k], R_B_C[k])
            if k < j:
                d = np.linalg.norm(t_B_C[k + 1] - t_B_C[k])
                n_sub = max(1, int(np.ceil(d / INTERP_STEP)))
                for s in range(1, n_sub):
                    u = s / n_sub
                    t_mid = (1 - u) * t_B_C[k] + u * t_B_C[k + 1]
                    R_mid = quaternion_slerp(R_B_C[k], R_B_C[k + 1], u)
                    seg_mask |= footprint_mask(t_mid, R_mid)
        visit[seg_mask] += 1
        i = j + 1
    return visit


def report(path, work_centers_B, work_areas, src_pos_B, src_region):
    hand_pos, R_B_hand = load_motion(path)
    T = len(hand_pos)
    t_B_C, R_B_C = hand_to_C_transforms(hand_pos, R_B_hand)
    regions = match_region(t_B_C, src_pos_B, src_region)
    n_regions_used = len(np.unique(regions))

    visit = accumulate_visits(t_B_C, R_B_C, regions, work_centers_B)

    total = work_areas.sum()
    cov = work_areas[visit >= 1].sum()
    ov = work_areas[visit >= 2].sum()
    cov_pct = cov / total * 100
    ov_cov = (ov / cov * 100) if cov > 0 else 0.0
    mean_mult = ((visit[visit >= 1] * work_areas[visit >= 1]).sum() / cov) if cov > 0 else 0.0

    one = work_areas[visit == 1].sum()
    two = work_areas[visit == 2].sum()
    tri = work_areas[visit >= 3].sum()

    # how many region segments did we create?
    n_segs = 0
    i = 0
    while i < len(regions):
        j = i
        while j + 1 < len(regions) and regions[j + 1] == regions[i]:
            j += 1
        n_segs += 1
        i = j + 1

    print(f"\n=== {os.path.basename(path)}   T={T}   regions used={n_regions_used}   sweeps={n_segs} ===")
    print(f"  coverage:           {cov_pct:6.2f}%   ({cov*1e4:.2f} / {total*1e4:.2f} cm²)")
    print(f"  overlap / covered:  {ov_cov:6.2f}%   ({ov*1e4:.2f} cm²)")
    print(f"  mean multiplicity:  {mean_mult:.3f} x    max: {int(visit.max())}")
    print(f"  1x:   {one*1e4:7.2f} cm²")
    print(f"  2x:   {two*1e4:7.2f} cm²")
    print(f"  >=3x: {tri*1e4:7.2f} cm²")
    return dict(name=os.path.basename(path), T=T, cov=cov_pct,
                ov_cov=ov_cov, mean=mean_mult, max_v=int(visit.max()),
                sweeps=n_segs)


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
    print(f"Pad disc R={R_TOOL*100:.0f}cm  indent±{PAD_INDENT*1e3:.1f}mm  interp step={INTERP_STEP*1e3:.1f}mm")

    src_pos_B, src_region = load_source_waypoints()
    print(f"source waypoints for region recovery: {len(src_pos_B)}, "
          f"{len(np.unique(src_region))} regions")

    paths = sys.argv[1:]
    if not paths:
        for pat in ("motions/panda_region_cover_*.csv",
                    "motions/panda_line_order_*.csv",
                    "motions/panda_hrchy_gtsp_*.csv"):
            m = sorted(glob.glob(pat))
            if m: paths.append(m[-1])

    rows = [report(p, work_centers_B, work_areas, src_pos_B, src_region)
            for p in paths if os.path.exists(p)]

    if len(rows) >= 2:
        print("\n--- side-by-side ---")
        print(f"{'motion':40s} {'T':>5s} {'sweeps':>7s} {'cov%':>7s} {'ov/cov%':>8s} {'mean':>6s}")
        for r in rows:
            print(f"{r['name']:40s} {r['T']:5d} {r['sweeps']:7d} {r['cov']:7.2f} {r['ov_cov']:8.2f} {r['mean']:6.2f}")


if __name__ == "__main__":
    main()
