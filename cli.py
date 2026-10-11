#!/usr/bin/env python3
"""
Runnable demo for FlowLedger.

Run sequence (prints real, captured output at every step):
  1. First run over the 8-row source CSV (2 rows deliberately invalid).
  2. Second run with the SAME input, immediately after -- must load zero
     new rows and the warehouse table must have no duplicates.
  3. A new row is appended to the source CSV; a third run must pick up
     only that one new row.
  4. Print the dead-letter queue contents and the final warehouse state.
"""
import random
import shutil
import sqlite3
from pathlib import Path

from gen4 import Memory
from pipeline import init_warehouse, run_pipeline

HERE = Path(__file__).parent
WORK = HERE / "data" / "_run"


def reset_run_dir():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    shutil.copy(HERE / "data" / "loan_applications_source.csv", WORK / "source.csv")


def row_count(conn):
    return conn.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0]


def main():
    reset_run_dir()
    memory = Memory(WORK / "pipeline_state.db")
    warehouse = sqlite3.connect(WORK / "warehouse.db")
    init_warehouse(warehouse)
    dlq_path = WORK / "dead_letter.csv"
    rng = random.Random(7)  # deterministic failure injection for a reproducible demo

    print("=" * 70)
    print("RUN 1 -- first pass over the 8-row source CSV")
    print("=" * 70)
    r1 = run_pipeline(WORK / "source.csv", memory, warehouse, dlq_path, fail_rate=0.5, rng=rng,
                      sleep=lambda s: None)
    print(f"  new rows seen: {r1['new_rows_seen']}  loaded: {r1['loaded_this_run']}  "
          f"rejected (DLQ): {r1['rejected_this_run']}")
    print(f"  running totals: {r1['totals']}")
    print(f"  warehouse row count: {row_count(warehouse)}")

    print()
    print("=" * 70)
    print("RUN 2 -- same input, run again immediately (idempotency check)")
    print("=" * 70)
    r2 = run_pipeline(WORK / "source.csv", memory, warehouse, dlq_path, fail_rate=0.5, rng=rng,
                      sleep=lambda s: None)
    print(f"  new rows seen: {r2['new_rows_seen']}  loaded: {r2['loaded_this_run']}  "
          f"rejected (DLQ): {r2['rejected_this_run']}")
    print(f"  running totals: {r2['totals']}")
    print(f"  warehouse row count: {row_count(warehouse)}  (must be unchanged from Run 1)")
    assert r2["new_rows_seen"] == 0, "watermark should have excluded all Run 1 rows"
    assert row_count(warehouse) == 6, "6 valid rows from Run 1 should still be exactly 6, no duplicates"

    print()
    print("=" * 70)
    print("RUN 3 -- one new row appended to the source, run again")
    print("=" * 70)
    with (WORK / "source.csv").open("a") as f:
        f.write("LA-1009,2026-10-01T09:40:00+00:00,Tanvi Joshi,205000,disbursed\n")
    r3 = run_pipeline(WORK / "source.csv", memory, warehouse, dlq_path, fail_rate=0.5, rng=rng,
                      sleep=lambda s: None)
    print(f"  new rows seen: {r3['new_rows_seen']}  loaded: {r3['loaded_this_run']}  "
          f"rejected (DLQ): {r3['rejected_this_run']}")
    print(f"  running totals: {r3['totals']}")
    print(f"  warehouse row count: {row_count(warehouse)}  (must be 7 -- 6 + the 1 new row)")
    assert r3["new_rows_seen"] == 1 and r3["loaded_this_run"] == 1
    assert row_count(warehouse) == 7

    print()
    print("=" * 70)
    print("Dead-letter queue contents (rows that failed validation)")
    print("=" * 70)
    print((dlq_path).read_text())

    print("=" * 70)
    print("Final warehouse contents")
    print("=" * 70)
    for row in warehouse.execute("SELECT * FROM loan_applications ORDER BY id"):
        print(" ", row)

    print()
    print("All assertions passed: no duplicates across repeated runs, "
          "incremental pickup of only new rows, invalid rows isolated to the DLQ.")


if __name__ == "__main__":
    main()
