"""For 'stuck' regions, scan all 2917 waypoints and report which ones
cover each region face at mode-aware PAD_INDENT. Report helper region ids."""
import os, sys, json
import numpy as np

sys.path.insert(0, "scripts")
from region_set_cover import compute_tool_C_poses_in_B, mesh_centers_areas, R_TOOL

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
RIDS = [int(x) for x in os.environ.get("RIDS", "72,150,137,151,125,51").split(",")]


def footprint(t, R, centers, h):
    delta = centers - t
    p_C = (R.T @ delta.T).T
    xy2 = p_C[:, 0]**2 + p_C[:, 1]**2
    z = p_C[:, 2]
    return (xy2 <= R_TOOL**2) & (np.abs(z) <= h)


P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
meta = json.load(open(os.path.join(DATA_DIR, "metadata.json"), encoding="utf-8"))

mask = P["tool_pose_valid"] & P["accessibility"]
orig = np.where(mask)[0]
reg = P["region_ids"][orig]
cm = P["contact_modes"][orig]
t_B_C, R_B_C = compute_tool_C_poses_in_B(P, meta, orig)
centers_B, areas, status, face_region = mesh_centers_areas(M, meta)
feas = status == 1
N = len(orig)

for rid in RIDS:
    f_in = np.where((face_region == rid) & feas)[0]
    reg_centers = centers_B[f_in]
    reg_areas = areas[f_in]
    n_face = len(f_in)
    print(f"\n=== region {rid}  ({n_face} faces, {reg_areas.sum()*1e4:.3f} cm²) ===")

    helpers = [[] for _ in range(n_face)]
    for li in range(N):
        h_i = 0.0005 if cm[li] == 1 else 0.0010
        m = footprint(t_B_C[li], R_B_C[li], reg_centers, h_i)
        for fi in np.where(m)[0]:
            helpers[fi].append((li, int(reg[li])))

    cov = sum(1 for h in helpers if len(h) > 0)
    print(f"  faces coverable globally: {cov}/{n_face}")
    for fi in range(n_face):
        if not helpers[fi]:
            print(f"    face{fi}  area={reg_areas[fi]*1e4:.3f}cm²  NO WAYPOINT COVERS IT")
            continue
        intra = [l for l, r in helpers[fi] if r == rid]
        cross_regs = sorted({r for l, r in helpers[fi] if r != rid})
        print(f"    face{fi}  area={reg_areas[fi]*1e4:.3f}cm²  n_helpers={len(helpers[fi])}  "
              f"intra={len(intra)}  cross_regs={cross_regs}")
