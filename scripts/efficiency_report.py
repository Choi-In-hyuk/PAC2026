"""Efficiency comparison of Hrchy-GTSP vs baselines.

Produces visualizations/efficiency_report.png with 4 panels:
  1. Absolute metrics (joint movement + reconfigs)
  2. Improvement % of Hrchy-GTSP over each baseline
  3. Paper Table I (reference comparison)
  4. Quality vs. graph size scaling (log scale, from paper + our data)
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_latest_perf():
    files = sorted(glob.glob("performance/performance_*.csv"))
    with open(files[-1]) as f:
        rows = list(csv.DictReader(f))
    latest = {}
    for r in rows:
        latest[(r["task"], r["solver"])] = r
    return latest


# Paper Table I averages (10 trials)
PAPER = {
    ("floor", "cartesian_tsp"): {"joint_movement": 166.23, "num_reconfigs": 2.0,  "time_s": 262.18},
    ("floor", "hrchy_gtsp"):    {"joint_movement": 123.37, "num_reconfigs": 1.3,  "time_s": 207.84},
    ("floor", "joint_gtsp"):    {"joint_movement": 161.29, "num_reconfigs": 21.6, "time_s": 1106.65},
    ("wok", "cartesian_tsp"):   {"joint_movement": 57.78,  "num_reconfigs": 0.0,  "time_s": 99.82},
    ("wok", "hrchy_gtsp"):      {"joint_movement": 41.71,  "num_reconfigs": 0.0,  "time_s": 74.56},
    ("wok", "joint_gtsp"):      {"joint_movement": 58.27,  "num_reconfigs": 14.7, "time_s": 566.26},
}

TASK_ORDER = ["floor", "wok", "wokreal"]
TASK_LABEL = {"floor": "panda / floor", "wok": "ur5 / wok", "wokreal": "ur5sander / wokreal"}
SOLV_ORDER = ["cartesian_tsp", "hrchy_gtsp", "joint_gtsp"]
SOLV_LABEL = {"cartesian_tsp": "Cart-TSP", "hrchy_gtsp": "Hrchy-GTSP (ours)", "joint_gtsp": "Joint-GTSP"}
COLOR = {"cartesian_tsp": "#888888", "hrchy_gtsp": "#2ca02c", "joint_gtsp": "#d62728"}


def _get(d, task, solver, key):
    v = d.get((task, solver), {}).get(key, 0)
    try: return float(v)
    except: return 0.0


def panel_absolute(ax1, ax2, latest):
    x = np.arange(len(TASK_ORDER))
    w = 0.26
    for i, s in enumerate(SOLV_ORDER):
        jm = [_get(latest, t, s, "joint_movement") for t in TASK_ORDER]
        rc = [_get(latest, t, s, "num_reconfigs") for t in TASK_ORDER]
        ax1.bar(x + (i - 1) * w, jm, w, label=SOLV_LABEL[s], color=COLOR[s])
        ax2.bar(x + (i - 1) * w, rc, w, label=SOLV_LABEL[s], color=COLOR[s])
    for ax, ttl, yl in [(ax1, "Joint movement (lower is better)", "radians"),
                        (ax2, "Reconfigurations (lower is better)", "count")]:
        ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], fontsize=9)
        ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)


def panel_improvement(ax, latest):
    """Hrchy-GTSP의 개선율(%) over each baseline."""
    labels, imp_jm_cart, imp_jm_joint, imp_rc_joint = [], [], [], []
    for t in TASK_ORDER:
        jm_h = _get(latest, t, "hrchy_gtsp", "joint_movement")
        jm_c = _get(latest, t, "cartesian_tsp", "joint_movement")
        jm_j = _get(latest, t, "joint_gtsp", "joint_movement")
        rc_h = _get(latest, t, "hrchy_gtsp", "num_reconfigs")
        rc_j = _get(latest, t, "joint_gtsp", "num_reconfigs")
        labels.append(TASK_LABEL[t])
        imp_jm_cart.append((jm_c - jm_h) / jm_c * 100 if jm_c > 0 else 0)
        imp_jm_joint.append((jm_j - jm_h) / jm_j * 100 if jm_j > 0 else 0)
        imp_rc_joint.append((rc_j - rc_h) / rc_j * 100 if rc_j > 0 else 0)
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, imp_jm_cart, w, label="joint mov vs Cart-TSP", color="#888888")
    ax.bar(x,      imp_jm_joint, w, label="joint mov vs Joint-GTSP", color="#d62728")
    ax.bar(x + w,  imp_rc_joint, w, label="reconfig vs Joint-GTSP", color="#8b0000")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("improvement (%)"); ax.set_title("Improvement of Hrchy-GTSP (higher is better)")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    # 숫자 표시
    for i, (a, b, c) in enumerate(zip(imp_jm_cart, imp_jm_joint, imp_rc_joint)):
        for xx, yy in [(i - w, a), (i, b), (i + w, c)]:
            if abs(yy) > 1:
                ax.text(xx, yy + (1 if yy >= 0 else -3), f"{yy:.0f}%",
                        ha="center", fontsize=8)


def panel_paper_compare(ax, latest):
    """논문 평균 대비 본 재현 결과 (joint_movement)."""
    labels = []
    paper_vals, ours_vals = [], []
    for t in ["floor", "wok"]:
        for s in SOLV_ORDER:
            labels.append(f"{t}\n{SOLV_LABEL[s]}")
            paper_vals.append(PAPER[(t, s)]["joint_movement"])
            ours_vals.append(_get(latest, t, s, "joint_movement"))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, paper_vals, w, label="paper (10-trial avg)", color="#1f77b4")
    ax.bar(x + w/2, ours_vals,  w, label="our single run",       color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("joint movement (rad)")
    ax.set_title("Paper vs. Reproduced (joint movement)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)


def panel_time(ax, latest):
    """계산 시간 비교 (논문 Table I만, 재현은 time 안 저장함)."""
    # Paper time only (we don't store wall time in our perf CSV)
    x = np.arange(2)
    w = 0.26
    for i, s in enumerate(SOLV_ORDER):
        vals = [PAPER[("floor", s)]["time_s"], PAPER[("wok", s)]["time_s"]]
        ax.bar(x + (i - 1) * w, vals, w, label=SOLV_LABEL[s], color=COLOR[s])
    ax.set_xticks(x); ax.set_xticklabels(["panda / floor", "ur5 / wok"], fontsize=9)
    ax.set_ylabel("computation time (s)")
    ax.set_title("Computation time (paper Table I)\nJoint-GTSP is 5-7x slower")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both"); ax.legend(fontsize=8)
    # 숫자 표시
    for i, s in enumerate(SOLV_ORDER):
        vals = [PAPER[("floor", s)]["time_s"], PAPER[("wok", s)]["time_s"]]
        for xi, v in enumerate(vals):
            ax.text(xi + (i - 1) * w, v * 1.1, f"{v:.0f}", ha="center", fontsize=7)


def main():
    latest = load_latest_perf()
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    panel_absolute(ax1, ax2, latest)
    panel_improvement(ax3, latest)
    panel_time(ax4, latest)

    fig.suptitle("arm_coverage: H-Joint-GTSP vs baselines — efficiency report",
                 fontsize=14, y=0.995)
    out = "visualizations/efficiency_report.png"
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    # 2nd figure: paper reproduction scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    panel_paper_compare(ax, latest)
    out2 = "visualizations/paper_reproduction.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out2)

    # Text summary
    print("\n=== Improvement Summary (our runs) ===")
    for t in TASK_ORDER:
        jm_h = _get(latest, t, "hrchy_gtsp", "joint_movement")
        jm_c = _get(latest, t, "cartesian_tsp", "joint_movement")
        jm_j = _get(latest, t, "joint_gtsp", "joint_movement")
        rc_h = int(_get(latest, t, "hrchy_gtsp", "num_reconfigs"))
        rc_j = int(_get(latest, t, "joint_gtsp", "num_reconfigs"))
        print(f"\n[{TASK_LABEL[t]}]")
        print(f"  joint_mov:  Hrchy {jm_h:.2f} vs Cart {jm_c:.2f} "
              f"({(jm_c-jm_h)/jm_c*100:+.1f}%) vs Joint {jm_j:.2f} "
              f"({(jm_j-jm_h)/jm_j*100:+.1f}%)")
        if rc_j > 0:
            print(f"  reconfigs:  Hrchy {rc_h} vs Joint {rc_j} "
                  f"({(rc_j-rc_h)/rc_j*100:+.0f}%)")


if __name__ == "__main__":
    main()
