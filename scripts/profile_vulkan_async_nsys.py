#!/usr/bin/env python3

import sqlite3
import statistics
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: profile_vulkan_async_nsys.py <nsys sqlite>")
        return 1

    db_path = sys.argv[1]
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    rows = cur.execute(
        """
        SELECT start, end
        FROM VULKAN_WORKLOAD
        WHERE end > start
        ORDER BY start
        """
    ).fetchall()
    if not rows:
        print("No VULKAN_WORKLOAD rows found")
        return 2

    first_start = rows[0][0]
    last_end = rows[-1][1]
    total_ns = last_end - first_start

    busy_ns = 0
    gaps = []
    cur_start, cur_end = rows[0]
    for start, end in rows[1:]:
        if start > cur_end:
            busy_ns += cur_end - cur_start
            gaps.append(start - cur_end)
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    busy_ns += cur_end - cur_start
    idle_ns = total_ns - busy_ns

    print(f"workloads={len(rows)}")
    print(f"total_ns={total_ns}")
    print(f"busy_ns={busy_ns}")
    print(f"idle_ns={idle_ns}")
    print(f"busy_pct={100.0 * busy_ns / total_ns:.6f}")
    print(f"idle_pct={100.0 * idle_ns / total_ns:.6f}")
    print(f"gaps_count={len(gaps)}")
    if gaps:
        print(f"median_gap_ns={int(statistics.median(gaps))}")
        print(f"max_gap_ns={max(gaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
