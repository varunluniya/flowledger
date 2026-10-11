"""
FlowLedger -- an incremental loading pipeline with retry logic.

Built on the shared kernel, not from scratch: the watermark-persistence,
grace-period, and idempotent-dedup pattern here is the same `gen4.Memory`
(SQLite-backed `get_param`/`set_param`) and `gen4.feedback.accumulate()`
primitives used across the rest of this portfolio to keep feedback-loop
aggregation incremental rather than re-scanning everything on every call.
`csv_new_rows_since()` below is the direct generic-CSV counterpart of
`gen4/feedback.py`'s `new_rows_since()` -- same watermark-minus-grace-
period filter, same persisted processed-id dedup set, just reading rows
from a CSV source instead of `Memory.decisions()`.

Two things this pipeline needed that the shared kernel didn't already
have: retry logic with exponential backoff + jitter (the destination
"warehouse" write is simulated as a flaky operation), and a dead-letter
queue for rows that fail validation. Those are new code, not reused from
the kernel.
"""

from __future__ import annotations

import csv
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from gen4 import Memory
from gen4 import feedback as fb

HERE = Path(__file__).parent
PIPELINE_KEY = "loan_applications_load"
GRACE_DAYS = 0.25  # 6 hours -- this source's updates can land slightly
                    # out of order within the same ingestion window


class TransientError(Exception):
    """Raised by the simulated flaky write -- the only exception type the
    retry decorator treats as retryable, per the guide's own Section 8.5
    pattern (any other exception type is re-raised immediately)."""


# -- retry decorator: exponential backoff + jitter ---------------------------

def retry_with_backoff(max_retries: int = 5, base_delay: float = 0.05,
                       max_delay: float = 2.0, sleep=time.sleep, rand=random.random):
    """wait_time = random(0, min(base_delay * 2**attempt, max_delay)) --
    the guide's own golden formula (Section 7 / 8.5): full jitter, not a
    fixed offset added to a deterministic backoff curve. Only TransientError
    triggers a retry; anything else is re-raised immediately. After
    max_retries attempts, the last TransientError is re-raised."""
    def decorator(func):
        def wrapped(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except TransientError:
                    if attempt >= max_retries:
                        raise
                    wait = rand() * min(base_delay * (2 ** attempt), max_delay)
                    print(f"    [retry] attempt {attempt + 1} failed transiently -- "
                          f"backing off {wait:.3f}s")
                    sleep(wait)
                    attempt += 1
        return wrapped
    return decorator


# -- the simulated flaky destination write ------------------------------------

def _flaky_warehouse_write(conn: sqlite3.Connection, row: dict, fail_rate: float, rng: random.Random):
    """Simulates a destination write that fails transiently some percentage
    of the time (a flaky network call to a warehouse API, in a real
    pipeline). Raises TransientError on simulated failure; a permanent
    failure (e.g. a constraint violation) would raise something else and
    must NOT be retried -- the guide's own distinction between transient
    and permanent failures (Section 7.1)."""
    if rng.random() < fail_rate:
        raise TransientError(f"simulated transient failure writing {row['id']}")
    conn.execute(
        "INSERT INTO loan_applications (id, updated_at, applicant_name, amount, status) "
        "VALUES (:id, :updated_at, :applicant_name, :amount, :status) "
        "ON CONFLICT(id) DO UPDATE SET updated_at=:updated_at, applicant_name=:applicant_name, "
        "amount=:amount, status=:status",
        row,
    )


# -- validation / dead-letter queue -------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None


def validate(row: dict) -> ValidationResult:
    """One required validation rule: applicant_name and amount must both be non-null. A real
    pipeline would have more rules; this is enough to demonstrate the DLQ
    mechanism end to end."""
    if not row.get("applicant_name"):
        return ValidationResult(False, "applicant_name is required but missing")
    if not row.get("amount"):
        return ValidationResult(False, "amount is required but missing")
    return ValidationResult(True)


def write_dead_letter(dlq_path: Path, row: dict, reason: str) -> None:
    exists = dlq_path.exists()
    with dlq_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "updated_at", "applicant_name", "amount",
                                                "status", "failure_reason"])
        if not exists:
            writer.writeheader()
        writer.writerow({**row, "failure_reason": reason})


# -- the generic-CSV counterpart of gen4/feedback.py's new_rows_since() ------

def csv_new_rows_since(csv_path: Path, memory: Memory, pipeline_key: str,
                       grace_days: float = 0.0) -> list[dict]:
    """`gen4/feedback.py`'s `new_rows_since()` pulls decisions since a
    persisted watermark (minus a grace period) and filters out already-
    processed ids, via `Memory.get_param`/`set_param`. This is the same
    pattern applied to a generic CSV source instead of `Memory.decisions()`:
    nothing here is a new persistence mechanism, it's the identical
    watermark+grace-period+processed-id-dedup design, reading rows from a
    file instead of a SQL join."""
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    wm = memory.get_param(f"watermark:{pipeline_key}")
    since = None
    if wm:
        since = datetime.fromisoformat(wm) - timedelta(days=grace_days)

    processed = set(memory.get_param(f"processed:{pipeline_key}", []))
    candidates = []
    for row in rows:
        ts = datetime.fromisoformat(row["updated_at"])
        if since is not None and ts <= since:
            continue
        if row["id"] in processed:
            continue
        candidates.append(row)
    return candidates


def mark_processed(memory: Memory, pipeline_key: str, rows: list[dict]) -> None:
    """Identical in shape to `gen4/feedback.py`'s `mark_processed()`:
    extend the processed-id set (so a grace-period re-pull of the same
    rows next run is a no-op) and advance the watermark to the latest
    `updated_at` actually seen -- only after the caller's writes already
    succeeded, so a mid-run failure never loses or duplicates anything on
    the next attempt."""
    if not rows:
        return
    processed = set(memory.get_param(f"processed:{pipeline_key}", []))
    processed.update(r["id"] for r in rows)
    memory.set_param(f"processed:{pipeline_key}", sorted(processed),
                     reason="incremental pipeline: mark rows processed")
    latest = max(r["updated_at"] for r in rows)
    memory.set_param(f"watermark:{pipeline_key}", latest,
                     reason="incremental pipeline: advance watermark")


# -- the pipeline itself -------------------------------------------------------

def run_pipeline(csv_path: Path, memory: Memory, warehouse_conn: sqlite3.Connection,
                 dlq_path: Path, fail_rate: float = 0.4, rng: random.Random | None = None,
                 sleep=time.sleep) -> dict:
    rng = rng or random.Random()
    new = csv_new_rows_since(csv_path, memory, PIPELINE_KEY, grace_days=GRACE_DAYS)

    loaded, rejected = [], []
    write = retry_with_backoff(sleep=sleep)(_flaky_warehouse_write)
    for row in new:
        result = validate(row)
        if not result.valid:
            write_dead_letter(dlq_path, row, result.reason)
            rejected.append(row)
            continue
        write(warehouse_conn, row, fail_rate, rng)
        loaded.append(row)
    warehouse_conn.commit()

    totals = fb.accumulate(memory, PIPELINE_KEY, loaded=len(loaded), rejected=len(rejected))
    mark_processed(memory, PIPELINE_KEY, new)
    return {"new_rows_seen": len(new), "loaded_this_run": len(loaded),
           "rejected_this_run": len(rejected), "totals": totals}


def init_warehouse(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loan_applications (
            id TEXT PRIMARY KEY, updated_at TEXT, applicant_name TEXT, amount TEXT, status TEXT
        )
    """)
    conn.commit()
