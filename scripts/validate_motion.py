"""Validate an arm_coverage motion CSV.

Checks per motion file:
  1. Joint limits (from URDF)
  2. Target pose error (FK vs. target columns)
  3. Reconfiguration jumps (large joint step between consecutive waypoints)

Usage:
  python scripts/validate_motion.py <motion_csv> <urdf> [base_link] [tip_link]
  python scripts/validate_motion.py --all   # scan motions/ against configs/urdfs/

Reports a one-line summary and detailed violations.
"""
import sys, os, csv, glob, re
import numpy as np
import PyKDL
from urdf_parser_py.urdf import URDF
from robot_kdl.kdl_parser import kdl_tree_from_urdf_model


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], np.array([[float(x) for x in r] for r in rows[1:]])


def build_chain(urdf_path, base_link=None, tip_link=None):
    with open(urdf_path) as f:
        urdf_xml = f.read()
    robot = URDF.from_xml_string(urdf_xml)
    tree = kdl_tree_from_urdf_model(robot)
    if base_link is None:
        base_link = robot.get_root()
    if tip_link is None:
        parents = {j.parent for j in robot.joints}
        leaves = [l.name for l in robot.links if l.name not in parents]
        tip_link = leaves[-1]
    chain = tree.getChain(base_link, tip_link)

    joint_names = []
    limits = []  # (lower, upper, vel)
    for jn in robot.get_chain(base_link, tip_link, links=False, fixed=False):
        for j in robot.joints:
            if j.name == jn and j.type != 'fixed':
                if j.type == 'continuous':
                    limits.append((-4 * np.pi, 4 * np.pi, j.limit.velocity if j.limit else 999))
                else:
                    limits.append((j.limit.lower, j.limit.upper, j.limit.velocity))
                joint_names.append(j.name)
                break
    return chain, joint_names, np.array(limits), robot


def fk_pose(chain, q):
    """Returns (pos (3,), quat (x,y,z,w) (4,))."""
    jarr = PyKDL.JntArray(len(q))
    for i, v in enumerate(q):
        jarr[i] = v
    fr = PyKDL.Frame()
    PyKDL.ChainFkSolverPos_recursive(chain).JntToCart(jarr, fr)
    pos = np.array([fr.p[0], fr.p[1], fr.p[2]])
    x, y, z, w = fr.M.GetQuaternion()
    return pos, np.array([x, y, z, w])


def quat_angle_err(q1, q2):
    """Smallest angle between two quaternions (xyzw)."""
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return 2.0 * np.arccos(d)


def validate(motion_csv, urdf_path, tolerances=None, reconfig_thresh=0.5, base_link=None, tip_link=None):
    chain, joint_names, limits, robot = build_chain(urdf_path, base_link, tip_link)
    hdr, data = load_csv(motion_csv)

    n_j = len(joint_names)
    q_cols = list(range(1, 1 + n_j))
    px, py, pz = hdr.index("target-POS_X"), hdr.index("target-POS_Y"), hdr.index("target-POS_Z")
    rx, ry, rz, rw = (hdr.index(f"target-ROT_{c}") for c in "XYZW")

    qs = data[:, q_cols]
    tgt_pos = data[:, [px, py, pz]]
    tgt_rot = data[:, [rx, ry, rz, rw]]

    T = len(qs)

    # 1) joint limits
    lower, upper = limits[:, 0], limits[:, 1]
    below = qs < (lower - 1e-6)
    above = qs > (upper + 1e-6)
    violations = np.where(below | above)
    joint_usage = (qs.max(0) - qs.min(0)) / (upper - lower + 1e-9)

    # 2) FK target pose error
    pos_errs, rot_errs = [], []
    for t in range(T):
        p_fk, q_fk = fk_pose(chain, qs[t])
        pos_errs.append(np.linalg.norm(p_fk - tgt_pos[t]))
        q_tgt = tgt_rot[t] / (np.linalg.norm(tgt_rot[t]) + 1e-12)
        rot_errs.append(quat_angle_err(q_fk, q_tgt))
    pos_errs = np.array(pos_errs); rot_errs = np.array(rot_errs)

    # 3) reconfig jumps: max |Δθ| per joint between consecutive waypoints
    dq = np.abs(np.diff(qs, axis=0))
    max_jump_per_step = dq.max(axis=1)
    reconfig_idx = np.where(max_jump_per_step > reconfig_thresh)[0]

    # tolerance interpretation: tolerances = [dx, dy, dz, drx, dry, drz]
    # The surface files sometimes give rotation-free axes. We use loose defaults for pass/fail.
    pos_tol_default = 0.01       # 1 cm
    rot_tol_default = np.deg2rad(5)  # 5 deg
    pos_tol = tolerances[0] if tolerances else pos_tol_default
    rot_tol = np.deg2rad(5) if (tolerances is None or tolerances[-1] >= 999) else rot_tol_default

    pos_ok = pos_errs.max() <= max(pos_tol, pos_tol_default) + 1e-6
    rot_ok = rot_errs.max() <= rot_tol_default * 2 + 1e-6 if (tolerances and tolerances[-1] >= 999) else rot_errs.max() <= rot_tol_default
    limits_ok = not (below.any() or above.any())

    name = os.path.basename(motion_csv)
    print(f"\n=== {name} ===")
    print(f"  Robot: {robot.name}  Chain: {len(joint_names)} joints  Waypoints: {T}")
    print(f"  Joint limits:   {'OK' if limits_ok else 'VIOLATED'}  "
          f"(usage %: {', '.join(f'{int(u*100)}' for u in joint_usage)})")
    if not limits_ok:
        for t, j in zip(*violations):
            side = "below" if below[t, j] else "above"
            print(f"      t={t}  joint[{j}] {joint_names[j]}={qs[t,j]:.3f} {side} [{lower[j]:.3f},{upper[j]:.3f}]")
    print(f"  Target position error:  max={pos_errs.max()*1000:.2f} mm   mean={pos_errs.mean()*1000:.2f} mm  "
          f"[{'OK' if pos_ok else 'HIGH'}]")
    print(f"  Target rotation error:  max={np.rad2deg(rot_errs.max()):.2f} deg  "
          f"mean={np.rad2deg(rot_errs.mean()):.2f} deg")
    print(f"  Large jumps (|dq|>{reconfig_thresh} rad): {len(reconfig_idx)} segments "
          f"{'(reconfig points)' if len(reconfig_idx) else ''}")

    return {
        "name": name, "T": T,
        "limits_ok": limits_ok,
        "pos_err_max_mm": pos_errs.max() * 1000,
        "pos_err_mean_mm": pos_errs.mean() * 1000,
        "rot_err_max_deg": float(np.rad2deg(rot_errs.max())),
        "rot_err_mean_deg": float(np.rad2deg(rot_errs.mean())),
        "n_large_jumps": int(len(reconfig_idx)),
    }


ROBOT_URDF = {
    "ur5": ("configs/urdfs/ur5.urdf", "base_link", "tool_tip"),
    "ur5sander": ("configs/urdfs/ur5sander.urdf", "base_link", "tool_tip"),
    "panda": ("configs/urdfs/panda.urdf", "world", "panda_hand"),
}
TOL = {
    "wokreal": [0., 0., 0., 0., 0., 999.],
    "wok":     [0.05, 0.05, 0., 0., 0., 999.],
    "floor":   [0., 0., 0., 0., 0., 0.],
}


def guess_urdf_tol(csv_name):
    # name like "ur5sander_hrchy_gtsp_...csv" or "ur5_gtsp_...csv" or "panda_tsp_iklink_...csv"
    stem = csv_name.split("_")[0]
    urdf_info = ROBOT_URDF.get(stem)
    urdf = urdf_info  # tuple (urdf_path, base, tip)
    # match task by surface naming scheme — fall back to looking at existing surfaces
    if stem == "ur5sander":
        tol = TOL["wokreal"]
    elif stem == "ur5":
        tol = TOL["wok"]
    elif stem == "panda":
        tol = TOL["floor"]
    else:
        tol = None
    return urdf, tol


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        rows = []
        for f in sorted(glob.glob("motions/*.csv")):
            urdf_info, tol = guess_urdf_tol(os.path.basename(f))
            if urdf_info is None:
                print(f"skip {f} (unknown robot)")
                continue
            urdf_path, base, tip = urdf_info
            rows.append(validate(f, urdf_path, tol, base_link=base, tip_link=tip))
        print("\n\n=== Summary ===")
        print(f"{'file':<55} {'T':>4} {'lim':>4} {'pos_max_mm':>11} {'rot_max_deg':>12} {'jumps':>6}")
        for r in rows:
            print(f"{r['name']:<55} {r['T']:>4} {'OK' if r['limits_ok'] else 'BAD':>4} "
                  f"{r['pos_err_max_mm']:>11.2f} {r['rot_err_max_deg']:>12.2f} {r['n_large_jumps']:>6}")
    else:
        motion_csv = sys.argv[1]
        urdf = sys.argv[2]
        validate(motion_csv, urdf)


if __name__ == "__main__":
    main()
