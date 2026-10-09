"""Visualize mesh with face coloring by region_id.

Saves two views (front, top) as PNG. Also an interactive-less orthographic
projection. Uses matplotlib Poly3DCollection.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
OUT_DIR = os.environ.get("OUT_DIR", "visualizations")
TAG = os.path.basename(DATA_DIR.rstrip("/"))

os.makedirs(OUT_DIR, exist_ok=True)

M = np.load(os.path.join(DATA_DIR, "mesh.npz"))
V = M["vertices"]
F = M["faces"]
freg = M["face_region_ids"]
status = M["face_status"]

print(f"vertices: {len(V)}, faces: {len(F)}")
print(f"regions in faces: {len(np.unique(freg))}  (range {freg.min()}..{freg.max()})")
print(f"face_status: feasible={int((status==1).sum())}, total={len(F)}")

# Build Poly3DCollection vertices
tris = V[F]   # (N, 3, 3)

# color per face by region
unique_regs = np.unique(freg)
# map region id to color (continuous colormap for many regions)
import matplotlib.cm as cm
n_reg = len(unique_regs)
rng = np.random.default_rng(42)
reg_colors = rng.random((n_reg, 3))   # distinct random colors per region
reg_colors = 0.3 + 0.7 * reg_colors   # avoid too dark
reg_to_idx = {int(r): i for i, r in enumerate(unique_regs)}

# faces with region == -1 (unassigned) get gray
colors = np.zeros((len(F), 4))
for i, r in enumerate(freg):
    r = int(r)
    if r < 0:
        colors[i] = [0.75, 0.75, 0.75, 1.0]
    else:
        c = reg_colors[reg_to_idx[r]]
        colors[i] = [c[0], c[1], c[2], 1.0]
# dim non-feasible
alpha = np.where(status == 1, 1.0, 0.25)
colors[:, 3] = alpha

mins = V.min(axis=0); maxs = V.max(axis=0)
ctr = (mins + maxs) / 2.0
span = (maxs - mins).max()


def render(elev, azim, out_path, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, facecolors=colors, edgecolors=(0, 0, 0, 0.05), linewidths=0.1)
    ax.add_collection3d(coll)
    ax.set_xlim(ctr[0]-span/2, ctr[0]+span/2)
    ax.set_ylim(ctr[1]-span/2, ctr[1]+span/2)
    ax.set_zlim(ctr[2]-span/2, ctr[2]+span/2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{TAG}  —  {title}  —  {n_reg} regions")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


render(elev=20, azim=-60, out_path=os.path.join(OUT_DIR, f"regions_{TAG}_persp.png"),
       title="perspective")
render(elev=90, azim=-90, out_path=os.path.join(OUT_DIR, f"regions_{TAG}_top.png"),
       title="top")
render(elev=0, azim=-90, out_path=os.path.join(OUT_DIR, f"regions_{TAG}_front.png"),
       title="front")
render(elev=0, azim=0, out_path=os.path.join(OUT_DIR, f"regions_{TAG}_side.png"),
       title="side")
