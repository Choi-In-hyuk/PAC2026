"""Run only the hrchy_gtsp solver on the latest car_door surface CSV."""
import sys, glob, csv, os
from datetime import datetime
sys.path.insert(0, "scripts")

from hrchy_gtsp import HrchyGTSP


def main():
    pattern = os.environ.get("SURFACE_PATTERN", "./surfaces/panda_*cardoor.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("no cardoor surface CSV found")
        return
    file_name = files[-1]
    print("file:", file_name)

    tolerances = [0.03, 0.03, 0., 0., 0., 999.]
    task = "cardoor"
    time_limit = int(os.environ.get("TIME_LIMIT", 120))
    num_samples = int(os.environ.get("NUM_SAMPLES", 30))
    print(f"time_limit={time_limit}s, num_samples={num_samples}")

    os.makedirs("performance", exist_ok=True)
    perf_name = f"performance/performance_cardoor_hrchy_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with open(perf_name, "w") as f:
        fields = ['task', 'solver', 'timestamp', 'joint_movement', 'num_reconfigs', 'num_points_on_surface']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        print("=====> Hrchy-Joint-GTSP")
        s = HrchyGTSP(file_name, tolerances)
        s.solve(time_limit, num_samples)
        m = s.motions
        i = m.best_idx
        w.writerow({
            'task': task, 'solver': "hrchy_gtsp",
            'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
            'joint_movement': m.ds[i],
            'num_reconfigs': m.num_reconfigs[i],
            'num_points_on_surface': len(m.motions[i]),
        })

    print("done, perf saved to", perf_name)


if __name__ == "__main__":
    main()
