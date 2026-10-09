"""Run the three solvers only on the car_door surface CSV."""
import sys, glob, csv
from datetime import datetime
sys.path.insert(0, "scripts")

from joint_gtsp import JointGTSP
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
    files = sorted(glob.glob("./surfaces/panda_*cardoor.csv"))
    if not files:
        print("no cardoor surface CSV found")
        return
    file_name = files[-1]
    print("file:", file_name)

    tolerances = [0.03, 0.03, 0., 0., 0., 999.]
    task = "cardoor"
    time_limit = 300   # car_door is harder than built-in benchmarks; give more time
    num_samples = 100

    perf_name = f"performance/performance_cardoor_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with open(perf_name, "w") as f:
        fields = ['task', 'solver', 'timestamp', 'joint_movement', 'num_reconfigs', 'num_points_on_surface']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        import os as _os
        if _os.environ.get("SKIP_CART"):
            print("=====> Skip Cartesian-TSP (SKIP_CART set)")
        else:
            print("=====> Cartesian-TSP")
            s1 = CartesianTSP(file_name, tolerances)
            s1.solve(time_limit, num_samples)
            write_perf(w, task, "cartesian_tsp", s1.motions)
            f.flush()

        print("=====> Hrchy-Joint-GTSP")
        s2 = HrchyGTSP(file_name, tolerances)
        s2.solve(time_limit, num_samples)
        write_perf(w, task, "hrchy_gtsp", s2.motions)
        f.flush()

        print("=====> Joint-GTSP")
        s3 = JointGTSP(file_name, tolerances)
        s3.solve(time_limit, num_samples)
        write_perf(w, task, "joint_gtsp", s3.motions)
        f.flush()

    print("done, perf saved to", perf_name)


if __name__ == "__main__":
    main()
