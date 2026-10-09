"""Extract waypoint hand poses from a motion CSV back to surface CSV format."""
import os, sys, csv
from datetime import datetime


def main():
    motion_path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "selected"

    with open(motion_path) as f:
        rows = list(csv.DictReader(f))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"surfaces/panda_{ts}_{tag}.csv"
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["time", "target-POS_X", "target-POS_Y", "target-POS_Z",
                    "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W"])
        for i, r in enumerate(rows):
            w.writerow([i,
                        r["target-POS_X"], r["target-POS_Y"], r["target-POS_Z"],
                        r["target-ROT_X"], r["target-ROT_Y"], r["target-ROT_Z"], r["target-ROT_W"]])
    print(f"saved {out}  ({len(rows)} waypoints)")


if __name__ == "__main__":
    main()
