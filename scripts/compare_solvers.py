"""Side-by-side comparison of the three solvers on each task.

Produces a single PNG with 3 rows (tasks) x 3 cols (solvers),
plus a bar chart of joint_movement vs num_reconfigs from the performance CSV.
"""
import sys, os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import PyKDL
from urdf_parser_py.urdf import URDF
from robot_kdl.kdl_parser import kdl_tree_from_urdf_model


SOLVERS = ["tsp_iklink", "hrchy_gtsp", "gtsp"]
SOLVER_LABEL = {"tsp_iklink": "Cartesian-TSP", "hrchy_gtsp": "Hrchy-GTSP (ours)", "gtsp": "Joint-GTSP"}
TASK_CFG = {
    "panda":    ("configs/urdfs/panda.urdf",    "world",     "panda_hand", "surfaces/panda_20240825_154124_floor.csv"),
    "ur5":      ("configs/urdfs/ur5.urdf",      "base_link", "tool_tip",   "surfaces/ur5_20240819_164816_wok.csv"),
    "ur5sander":("configs/urdfs/ur5sander.urdf","base_link", "tool_tip",   "surfaces/ur5sander_20240902_151946_wokreal.csv"),
}


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], np.array([[float(x) for x in r] for r in rows[1:]])


def build_chain(urdf_path, base_link, tip_link):
    with open(urdf_path) as f:
        urdf_xml = f.read()
    robot = URDF.from_xml_string(urdf_xml)
    tree = kdl_tree_from_urdf_model(robot)
    return tree.getChain(base_link, tip_link)


def joint_positions(chain, q):
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


def latest_motion(robot, solver):
    files = sorted(glob.glob(f"motions/{robot}_{solver}_*.csv"))
    return files[-1] if files else None


def plot_snapshot(ax, chain, qs, ee_traj, surf_pts, title):
    allp = np.vstack([ee_traj, surf_pts])
    mn, mx = allp.min(0), allp.max(0)
    ctr = (mn + mx) / 2
    rng = max((mx - mn).max() * 0.65, 0.4)
    ax.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2], s=4, c="tab:orange", alpha=0.5)
    ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], "-", c="tab:blue", lw=0.9, alpha=0.9)
    for ti in np.linspace(0, len(qs) - 1, 4).astype(int):
        p = joint_positions(chain, qs[ti])
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", c="gray", lw=1.1, ms=2, alpha=0.6)
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    tasks = list(TASK_CFG.keys())
    fig = plt.figure(figsize=(14, 12))

    for ti, task in enumerate(tasks):
        urdf, base, tip, surf_csv = TASK_CFG[task]
        chain = build_chain(urdf, base, tip)
        s_hdr, s_data = load_csv(surf_csv)
        sx = s_hdr.index("target-POS_X"); sy = s_hdr.index("target-POS_Y"); sz = s_hdr.index("target-POS_Z")
        surf_pts = s_data[:, [sx, sy, sz]]

        for si, solver in enumerate(SOLVERS):
            ax = fig.add_subplot(3, 3, ti * 3 + si + 1, projection="3d")
            mcsv = latest_motion(task, solver)
            if not mcsv:
                ax.set_title(f"{task} / {SOLVER_LABEL[solver]}\n(no motion)")
                continue
            m_hdr, m_data = load_csv(mcsv)
            n_j = chain.getNrOfJoints()
            qs = m_data[:, 1:1 + n_j]
            mx_col = m_hdr.index("target-POS_X"); my_col = m_hdr.index("target-POS_Y"); mz_col = m_hdr.index("target-POS_Z")
            ee_traj = m_data[:, [mx_col, my_col, mz_col]]
            plot_snapshot(ax, chain, qs, ee_traj, surf_pts,
                          f"{task} / {SOLVER_LABEL[solver]}")

    plt.suptitle("arm_coverage: three solvers on three surfaces", fontsize=13)
    plt.tight_layout()
    out = "visualizations/comparison.png"
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    # bar chart from latest performance CSV
    perf_files = sorted(glob.glob("performance/performance_*.csv"))
    perf = perf_files[-1]
    with open(perf) as f:
        rows = list(csv.DictReader(f))
    # take latest row per (task, solver)
    latest = {}
    for r in rows:
        latest[(r["task"], r["solver"])] = r

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    task_order = ["floor", "wok", "wokreal"]
    task_display = {"floor": "panda / floor", "wok": "ur5 / wok", "wokreal": "ur5sander / wokreal"}
    solver_order = ["cartesian_tsp", "hrchy_gtsp", "joint_gtsp"]
    solver_display = {"cartesian_tsp": "Cart-TSP", "hrchy_gtsp": "Hrchy-GTSP (ours)", "joint_gtsp": "Joint-GTSP"}
    colors = {"cartesian_tsp": "tab:gray", "hrchy_gtsp": "tab:green", "joint_gtsp": "tab:red"}

    xs = np.arange(len(task_order))
    width = 0.26
    for i, solv in enumerate(solver_order):
        jm = [float(latest.get((t, solv), {"joint_movement": 0})["joint_movement"]) for t in task_order]
        rc = [int(latest.get((t, solv), {"num_reconfigs": 0})["num_reconfigs"]) for t in task_order]
        axes[0].bar(xs + (i - 1) * width, jm, width, label=solver_display[solv], color=colors[solv])
        axes[1].bar(xs + (i - 1) * width, rc, width, label=solver_display[solv], color=colors[solv])

    for ax, ylabel, title in [(axes[0], "joint movement (rad)", "Joint movement (lower is better)"),
                              (axes[1], "num reconfigurations", "Reconfigurations (lower is better)")]:
        ax.set_xticks(xs)
        ax.set_xticklabels([task_display[t] for t in task_order])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    out2 = "visualizations/performance_bars.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out2)


if __name__ == "__main__":
    main()
