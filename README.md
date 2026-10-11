# FlowLedger

A working incremental loading pipeline for a CSV source (loan
applications — id, updated_at, applicant_name, amount, status) into a
SQLite "warehouse" table: timestamp watermarking with a grace period,
idempotent upsert loading, retry logic with exponential backoff and
jitter, and a dead-letter queue for records that fail validation.

## Built on the shared kernel, not from scratch

This repo's own `gen4/` kernel is the same incrementally-correct one used
across the rest of this portfolio (a `(system, ts DESC)` index,
append-only fact history, and the `new_rows_since`/`mark_processed`/
`accumulate`/`incremental_calibration` primitives). `pipeline.py`'s
`csv_new_rows_since()` is the direct generic-CSV counterpart of
`gen4/feedback.py`'s `new_rows_since()` — the identical watermark +
grace-period + persisted-processed-id-dedup pattern, reading rows from a
CSV instead of `Memory.decisions()`. `mark_processed()` and the running
totals use the real `Memory.get_param`/`set_param` and
`feedback.accumulate()` calls, unmodified.

Two things this pipeline needed that the shared kernel didn't already
have: retry logic with exponential backoff + jitter, and a dead-letter
queue. Those are new, built here.

## What's implemented

- **Timestamp watermarking with a configurable grace period**, persisted
  between runs via `Memory`'s SQLite-backed params (`watermark:<key>`).
  `GRACE_DAYS = 0.25` (6 hours) in `pipeline.py`.
- **Idempotent loading via upsert**, keyed on `id`
  (`INSERT ... ON CONFLICT(id) DO UPDATE`). Combined with a persisted
  processed-id set, so a grace-period re-pull of already-loaded rows
  never double-processes or duplicates them.
- **Retry with exponential backoff + jitter** (`retry_with_backoff`):
  `wait = random(0, min(base_delay * 2**attempt, max_delay))`, full
  jitter rather than a fixed offset. Only `TransientError` is retried;
  any other exception is re-raised immediately, un-retried. Wraps a
  simulated flaky warehouse write (`_flaky_warehouse_write`, fails
  transiently a configurable percentage of the time).
- **Dead-letter queue**: one validation rule (`applicant_name` and
  `amount` must both be non-null) — rows that fail it are written to
  `dead_letter.csv` with the specific reason, never reach the warehouse,
  and never block the rest of the batch.

## Run it

```
python3 cli.py
```

Runs the pipeline three times against the same 8-row source CSV (2 rows
deliberately invalid): once cold, once again immediately with no new
data (proving no duplicates), then once more after appending one new
row (proving only that new row is picked up). Prints the DLQ contents
and final warehouse state, with assertions on row counts baked in.

```
python3 -m pytest -q
```

9 tests: validation, full-load-then-idempotent-rerun, incremental pickup
of only new rows, DLQ isolation, a late-arriving-row-within-grace-period
case (a row whose timestamp is earlier than the current watermark but
still inside the grace window), and three retry-decorator tests (retries
only transient errors, eventually raises after max_retries, succeeds once
the flaky operation stops failing).

## Honest disclosure

The flaky warehouse write and the two invalid source rows are
deliberately injected to exercise the retry and DLQ paths on every
demo run — there's no real external warehouse API or network call here,
consistent with the rest of this portfolio's disclosed stand-ins for
pieces that would be live infrastructure in production.
