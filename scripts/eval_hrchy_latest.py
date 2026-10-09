"""Area-based coverage + overlap for the latest hrchy_gtsp motion on car_door."""
import os, glob, json
import numpy as np

import sys
sys.path.insert(0, "scripts")
from evaluate_coverage import load_motion_targets, R_TOOL
from evaluate_mesh_coverage import load_mesh, face_centers_and_areas, coverage_by_area


def main():
    files = sorted(glob.glob("motions/panda_hrchy_gtsp_*.csv"))
    if not files:
        print("no hrchy_gtsp motion found")
        return
    motion_path = files[-1]
    print(f"motion: {motion_path}")

    V_W, F, status, region = load_mesh()
    centers_W, areas, _ = face_centers_and_areas(V_W, F)

    meta = json.load(open("meshes/car_door/metadata.json", encoding="utf-8"))
    T_B_W = np.array(meta["T_B_W"])
    R_BW = T_B_W[:3, :3]; t_BW = T_B_W[:3, 3]
    centers_B = (R_BW @ centers_W.T).T + t_BW

    work_mask = (status == 1)
    work_centers = centers_B[work_mask]
    work_areas = areas[work_mask]
    total_work_area = work_areas.sum()
    print(f"Working faces: {work_mask.sum()}  working area: {total_work_area*1e4:.2f} cm²  R_tool: {R_TOOL*100:.0f} cm")

    tool = load_motion_targets(motion_path)
    print(f"motion waypoints T = {len(tool)}")

    visit = coverage_by_area(tool, work_centers, work_areas, R_TOOL)

    covered_mask = visit >= 1
    covered_area = work_areas[covered_mask].sum()
    cov_ratio = covered_area / total_work_area

    # overlap definitions
    overlap_mask = visit >= 2
    overlap_area = work_areas[overlap_mask].sum()
    overlap_frac_of_covered = overlap_area / covered_area if covered_area > 0 else 0
    overlap_frac_of_feasible = overlap_area / total_work_area

    # area-weighted mean visit count (on covered faces)
    mean_multiplicity_covered = (visit[covered_mask] * work_areas[covered_mask]).sum() / covered_area if covered_area > 0 else 0

    # multiplicity bucket area fractions (within covered area)
    one_area = work_areas[visit == 1].sum()
    two_area = work_areas[visit == 2].sum()
    three_plus_area = work_areas[visit >= 3].sum()

    print()
    print(f"Area coverage       (covered / feasible):     {cov_ratio*100:6.2f}%   ({covered_area*1e4:.2f} / {total_work_area*1e4:.2f} cm²)")
    print(f"Overlap ratio       (>=2x / covered):          {overlap_frac_of_covered*100:6.2f}%   ({overlap_area*1e4:.2f} cm²)")
    print(f"Overlap ratio       (>=2x / feasible):         {overlap_frac_of_feasible*100:6.2f}%")
    print(f"Mean multiplicity   (area-weighted, covered):  {mean_multiplicity_covered:.3f} x")
    print(f"Max visit count on any single face:            {int(visit.max())}")
    print()
    print("Multiplicity breakdown within COVERED area:")
    if covered_area > 0:
        print(f"  1x:   {one_area*1e4:7.2f} cm²  ({one_area/covered_area*100:5.2f}%)")
        print(f"  2x:   {two_area*1e4:7.2f} cm²  ({two_area/covered_area*100:5.2f}%)")
        print(f"  >=3x: {three_plus_area*1e4:7.2f} cm²  ({three_plus_area/covered_area*100:5.2f}%)")


if __name__ == "__main__":
    main()
