"""Run Cart-TSP + Hrchy-GTSP on a given surface CSV.

Usage:
  python scripts/run_one_task.py <surface_csv>
"""
import sys, csv
from datetime import datetime
sys.path.insert(0, "scripts")

from cartesian_tsp import CartesianTSP
from hrchy_gtsp import HrchyGTSP


def write_perf(writer, task, solver, motions):
    i = motions.best_idx
    writer.writerow({
        'task': task, 'solver': solver,
        'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
        'joint_movement': motions.ds[i],
        'num_reconfigs': motions.num_reconfigs[i],
        'num_points_on_surface': len(motions.motions[i]),
    })


def main():
    surface_csv = sys.argv[1]
    task = surface_csv.split('/')[-1].split('.')[0].split('_')[-1]
    tolerances = [0.03, 0.03, 0., 0., 0., 999.]
    time_limit = 300
    num_samples = 100
    print(f"file: {surface_csv}  task: {task}")

    perf_name = f"performance/performance_{task}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with open(perf_name, "w") as f:
        fields = ['task', 'solver', 'timestamp', 'joint_movement', 'num_reconfigs', 'num_points_on_surface']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        print("=====> Cartesian-TSP")
        s1 = CartesianTSP(surface_csv, tolerances)
        s1.solve(time_limit, num_samples)
        write_perf(w, task, "cartesian_tsp", s1.motions)
        f.flush()

        print("=====> Hrchy-Joint-GTSP")
        s2 = HrchyGTSP(surface_csv, tolerances)
        s2.solve(time_limit, num_samples)
        write_perf(w, task, "hrchy_gtsp", s2.motions)
        f.flush()

    print("done, perf saved to", perf_name)


if __name__ == "__main__":
    main()
