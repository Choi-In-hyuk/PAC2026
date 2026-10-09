"""Verify hypothesis: failing regions are contact_mode=2 (edge) and need
PAD_INDENT = 1.0 mm instead of hardcoded 0.5 mm.

For each candidate region:
  - list waypoints with their contact_mode
  - evaluate region coverage under PAD_INDENT 0.5mm vs 1.0mm vs
    per-waypoint (0.5mm for mode1, 1.0mm for mode2)
"""
import os, sys, json
import numpy as np

sys.path.insert(0, "scripts")
from region_set_cover import (compute_tool_C_poses_in_B, mesh_centers_areas,
                               R_TOOL)

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
RIDS = [int(x) for x in os.environ.get(
    "RIDS", "49,41,72,66,104,150,137,151,125,51,75,26,32,101,106,121,105,35,76,95,15,64,45,100,145,74,18,48,37,69").split(",")]


def footprint_mask(t_B_C, R_B_C, centers_B, h):
    delta = centers_B - t_B_C
    p_C = (R_B_C.T @ delta.T).T
    xy2 = p_C[:, 0] ** 2 + p_C[:, 1] ** 2
    z = p_C[:, 2]
    return (xy2 <= R_TOOL ** 2) & (np.abs(z) <= h)


def main():
    P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
    M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

    mask = P["tool_pose_valid"] & P["accessibility"]
    orig = np.where(mask)[0]
    reg = P["region_ids"][orig]
    cm = P["contact_modes"][orig]
    t_B_C, R_B_C = compute_tool_C_poses_in_B(P, meta, orig)

    centers_B, areas, face_status, face_region = mesh_centers_areas(M, meta)
    feas = face_status == 1

    print(f"{'rid':>5s} {'nSrc':>5s} {'edge':>5s} {'cent':>5s} {'nFace':>6s} {'area':>7s}  "
          f"{'cov@0.5':>8s} {'cov@1.0':>8s} {'cov@mix':>8s}")
    print("-" * 80)

    sum_area = 0.0; sum_05 = 0.0; sum_10 = 0.0; sum_mix = 0.0
    for rid in RIDS:
        local = np.where(reg == rid)[0]
        f_in = np.where((face_region == rid) & feas)[0]
        if len(local) == 0 or len(f_in) == 0:
            print(f"{rid:5d}  skip (no wp or no face)")
            continue
        reg_centers = centers_B[f_in]
        reg_areas = areas[f_in]
        total = reg_areas.sum()

        cov05 = np.zeros(len(f_in), dtype=bool)
        cov10 = np.zeros(len(f_in), dtype=bool)
        covmx = np.zeros(len(f_in), dtype=bool)
        n_edge = n_cent = 0
        for li in local:
            cov05 |= footprint_mask(t_B_C[li], R_B_C[li], reg_centers, 0.0005)
            cov10 |= footprint_mask(t_B_C[li], R_B_C[li], reg_centers, 0.0010)
            h_i = 0.0005 if cm[li] == 1 else 0.0010
            covmx |= footprint_mask(t_B_C[li], R_B_C[li], reg_centers, h_i)
            if cm[li] == 1: n_cent += 1
            elif cm[li] == 2: n_edge += 1
        c05 = reg_areas[cov05].sum() / total * 100
        c10 = reg_areas[cov10].sum() / total * 100
        cmx = reg_areas[covmx].sum() / total * 100
        print(f"{rid:5d} {len(local):5d} {n_edge:5d} {n_cent:5d} {len(f_in):6d} {total*1e4:7.3f}  "
              f"{c05:7.2f}% {c10:7.2f}% {cmx:7.2f}%")
        sum_area += total; sum_05 += reg_areas[cov05].sum()
        sum_10 += reg_areas[cov10].sum(); sum_mix += reg_areas[covmx].sum()

    print("-" * 80)
    print(f"TOTAL (these regions): area={sum_area*1e4:.3f} cm²  "
          f"cov@0.5={sum_05/sum_area*100:.2f}%  "
          f"cov@1.0={sum_10/sum_area*100:.2f}%  "
          f"cov@mix={sum_mix/sum_area*100:.2f}%")


if __name__ == "__main__":
    main()
