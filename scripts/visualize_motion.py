"""Visualize an arm_coverage motion CSV against its surface target CSV.

Produces:
 - a static 3D plot (surface points + EE trajectory + stick-figure at a few frames)
 - an animated GIF of the robot stick-figure sweeping the surface

Usage:
    python scripts/visualize_motion.py <motion_csv> <surface_csv> <urdf> [base_link] [tip_link]

Headless-safe (Agg backend, pillow writer).
"""
import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

import PyKDL
from urdf_parser_py.urdf import URDF
from robot_kdl.kdl_parser import kdl_tree_from_urdf_model


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    arr = np.array([[float(x) for x in r] for r in data])
    return header, arr


def joint_positions_along_chain(chain, q):
    """Return (N+1, 3) positions of each segment's tip in base frame."""
    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    n_seg = chain.getNrOfSegments()
    n_j = chain.getNrOfJoints()
    jarr = PyKDL.JntArray(n_j)
    for i in range(n_j):
        jarr[i] = q[i]
    pts = [(0.0, 0.0, 0.0)]
    for s in range(1, n_seg + 1):
        fr = PyKDL.Frame()
        fk.JntToCart(jarr, fr, s)
        pts.append((fr.p[0], fr.p[1], fr.p[2]))
    return np.array(pts)


def main():
    motion_csv = sys.argv[1]
    surface_csv = sys.argv[2]
    urdf_path = sys.argv[3]
    base_link = sys.argv[4] if len(sys.argv) > 4 else None
    tip_link = sys.argv[5] if len(sys.argv) > 5 else None

    with open(urdf_path, 'rb') as f:
        urdf_xml = f.read()
    robot = URDF.from_xml_string(urdf_xml)
    tree = kdl_tree_from_urdf_model(robot)

    # auto-detect base/tip if not given
    if base_link is None:
        base_link = robot.get_root()
    if tip_link is None:
        # pick a leaf: a link that is not a parent of any joint
        parents = {j.parent for j in robot.joints}
        leaves = [l.name for l in robot.links if l.name not in parents]
        tip_link = leaves[-1]
    print(f"base={base_link} tip={tip_link}")

    chain = tree.getChain(base_link, tip_link)
    n_j = chain.getNrOfJoints()

    # load data
    m_hdr, m = load_csv(motion_csv)
    s_hdr, s = load_csv(surface_csv)
    # motion columns: time + n_j joint angles + 7 target pose
    q_cols = list(range(1, 1 + n_j))
    ee_cols_motion = [m_hdr.index("target-POS_X"), m_hdr.index("target-POS_Y"), m_hdr.index("target-POS_Z")]
    surf_cols = [s_hdr.index("target-POS_X"), s_hdr.index("target-POS_Y"), s_hdr.index("target-POS_Z")]

    qs = m[:, q_cols]
    ee_traj = m[:, ee_cols_motion]
    surf_pts = s[:, surf_cols]

    # FK to get stick figure positions over time
    all_pts = np.array([joint_positions_along_chain(chain, q) for q in qs])
    # all_pts: (T, n_seg+1, 3)

    # workspace limits
    allp = np.vstack([all_pts.reshape(-1, 3), surf_pts, ee_traj])
    mn, mx = allp.min(0), allp.max(0)
    ctr = (mn + mx) / 2
    rng = max((mx - mn).max() * 0.6, 0.5)

    out_prefix = os.path.splitext(os.path.basename(motion_csv))[0]
    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

    # 1) static plot: surface + ee trajectory + a few stick-figure snapshots
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2], s=12, c="tab:orange", label="surface targets", alpha=0.9)
    ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], "-", c="tab:blue", lw=1.2, label="EE trajectory")
    T = len(qs)
    snapshot_idx = np.linspace(0, T - 1, 5).astype(int)
    for ti in snapshot_idx:
        p = all_pts[ti]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", c="gray", lw=1.4, ms=3, alpha=0.6)
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(out_prefix)
    ax.legend(loc="upper left", fontsize=8)
    static_path = os.path.join(out_dir, out_prefix + ".png")
    fig.savefig(static_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", static_path)

    # 2) animation
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2], s=10, c="tab:orange", alpha=0.6)
    traj_line, = ax.plot([], [], [], "-", c="tab:blue", lw=1.2)
    robot_line, = ax.plot([], [], [], "-o", c="black", lw=2.2, ms=4)
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_title(out_prefix)

    def update(i):
        p = all_pts[i]
        robot_line.set_data(p[:, 0], p[:, 1])
        robot_line.set_3d_properties(p[:, 2])
        traj_line.set_data(ee_traj[: i + 1, 0], ee_traj[: i + 1, 1])
        traj_line.set_3d_properties(ee_traj[: i + 1, 2])
        return robot_line, traj_line

    frames = min(T, 150)
    step = max(T // frames, 1)
    anim = FuncAnimation(fig, update, frames=range(0, T, step), interval=60, blit=False)
    gif_path = os.path.join(out_dir, out_prefix + ".gif")
    anim.save(gif_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print("saved", gif_path)


if __name__ == "__main__":
    main()
