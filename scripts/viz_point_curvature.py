"""Visualize surface points colored by curvature.

Uses surface_points.npz:
  - points (M, 3): surface positions in W frame
  - curvatures (M,): max(|k1|, |k2|), 1/m
  - curvature_valid (M,): bool
  - region_ids (M,): work region id (-1 = outside)

Saves 4 views (persp, top, front, side) and a histogram of curvature.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa

DATA_DIR = os.environ.get("DATA_DIR", "meshes/car_door_dir0")
OUT_DIR = os.environ.get("OUT_DIR", "visualizations")
TAG = os.path.basename(DATA_DIR.rstrip("/"))
CMAP = os.environ.get("CMAP", "viridis")
# percentile clip for color scale (robust to outliers)
PCT_LO = float(os.environ.get("PCT_LO", "2"))
PCT_HI = float(os.environ.get("PCT_HI", "98"))

os.makedirs(OUT_DIR, exist_ok=True)

P = np.load(os.path.join(DATA_DIR, "surface_points.npz"))
pts = P["points"]            # (M, 3), W frame
cur = P["curvatures"]        # (M,)
valid = P["curvature_valid"] # (M,)
region = P["region_ids"]     # (M,)

print(f"total points: {len(pts)}")
print(f"curvature valid: {int(valid.sum())}")
print(f"region != -1: {int((region != -1).sum())}")

# use only valid curvatures for color scale
mask = valid & np.isfinite(cur)
vals = cur[mask]
lo, hi = np.percentile(vals, [PCT_LO, PCT_HI])
print(f"curvature range (p{PCT_LO:.0f}-p{PCT_HI:.0f}): [{lo:.3f}, {hi:.3f}] 1/m")
print(f"  min/max all valid: [{vals.min():.3f}, {vals.max():.3f}] 1/m")

# color per point
c = np.clip(cur, lo, hi)
# invalid points: gray
c_valid = np.where(mask, c, lo)

mins = pts.min(axis=0); maxs = pts.max(axis=0)
ctr = (mins + maxs) / 2.0
span = (maxs - mins).max()


def render(elev, azim, out_path, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # draw invalid first (gray)
    inv = ~mask
    if inv.any():
        ax.scatter(pts[inv, 0], pts[inv, 1], pts[inv, 2], s=2,
                   c="lightgray", alpha=0.3)
    sc = ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                    c=c_valid[mask], cmap=CMAP, s=3, vmin=lo, vmax=hi)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("curvature  max(|k1|,|k2|)  [1/m]")
    ax.set_xlim(ctr[0]-span/2, ctr[0]+span/2)
    ax.set_ylim(ctr[1]-span/2, ctr[1]+span/2)
    ax.set_zlim(ctr[2]-span/2, ctr[2]+span/2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{TAG}  —  {title}  —  curvature")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


render(20, -60, os.path.join(OUT_DIR, f"curvature_{TAG}_persp.png"), "perspective")
render(90, -90, os.path.join(OUT_DIR, f"curvature_{TAG}_top.png"), "top")
render(0, -90, os.path.join(OUT_DIR, f"curvature_{TAG}_front.png"), "front")
render(0, 0,   os.path.join(OUT_DIR, f"curvature_{TAG}_side.png"), "side")

# histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(vals, bins=80, color="steelblue", edgecolor="black", linewidth=0.3)
ax.axvline(lo, color="red", linestyle="--", label=f"p{PCT_LO:.0f} = {lo:.2f}")
ax.axvline(hi, color="red", linestyle="--", label=f"p{PCT_HI:.0f} = {hi:.2f}")
ax.set_xlabel("curvature  max(|k1|,|k2|)  [1/m]")
ax.set_ylabel("count")
ax.set_title(f"{TAG} — curvature histogram  (N={len(vals)})")
ax.legend()
plt.tight_layout()
hp = os.path.join(OUT_DIR, f"curvature_{TAG}_hist.png")
plt.savefig(hp, dpi=120)
plt.close(fig)
print(f"saved {hp}")
