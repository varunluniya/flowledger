import random
import sqlite3
from pathlib import Path

import pytest

from gen4 import Memory
from pipeline import (TransientError, init_warehouse, mark_processed,
                      retry_with_backoff, run_pipeline, validate)


@pytest.fixture
def warehouse():
    conn = sqlite3.connect(":memory:")
    init_warehouse(conn)
    return conn


@pytest.fixture
def memory():
    return Memory()


def _write_csv(path: Path, rows: list[str]):
    path.write_text("id,updated_at,applicant_name,amount,status\n" + "\n".join(rows) + "\n")


def test_validation_rejects_missing_required_fields():
    assert validate({"applicant_name": "A", "amount": "100"}).valid
    assert not validate({"applicant_name": "", "amount": "100"}).valid
    assert not validate({"applicant_name": "A", "amount": ""}).valid


def test_full_load_then_idempotent_rerun_no_duplicates(tmp_path, memory, warehouse):
    csv_path = tmp_path / "source.csv"
    dlq_path = tmp_path / "dlq.csv"
    _write_csv(csv_path, [
        "a1,2026-01-01T00:00:00+00:00,Alice,1000,ok",
        "a2,2026-01-01T00:01:00+00:00,Bob,2000,ok",
    ])

    first = run_pipeline(csv_path, memory, warehouse, dlq_path, fail_rate=0.0, sleep=lambda s: None)
    assert first["loaded_this_run"] == 2
    assert warehouse.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0] == 2

    second = run_pipeline(csv_path, memory, warehouse, dlq_path, fail_rate=0.0, sleep=lambda s: None)
    assert second["new_rows_seen"] == 0
    assert second["loaded_this_run"] == 0
    assert warehouse.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0] == 2


def test_incremental_pickup_of_only_new_rows(tmp_path, memory, warehouse):
    csv_path = tmp_path / "source.csv"
    dlq_path = tmp_path / "dlq.csv"
    _write_csv(csv_path, ["a1,2026-01-01T00:00:00+00:00,Alice,1000,ok"])
    run_pipeline(csv_path, memory, warehouse, dlq_path, fail_rate=0.0, sleep=lambda s: None)

    with csv_path.open("a") as f:
        f.write("a2,2026-01-01T00:05:00+00:00,Bob,2000,ok\n")
    second = run_pipeline(csv_path, memory, warehouse, dlq_path, fail_rate=0.0, sleep=lambda s: None)

    assert second["new_rows_seen"] == 1
    assert second["loaded_this_run"] == 1
    assert warehouse.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0] == 2


def test_invalid_rows_go_to_dead_letter_queue_not_the_warehouse(tmp_path, memory, warehouse):
    csv_path = tmp_path / "source.csv"
    dlq_path = tmp_path / "dlq.csv"
    _write_csv(csv_path, [
        "a1,2026-01-01T00:00:00+00:00,Alice,1000,ok",
        "a2,2026-01-01T00:01:00+00:00,,2000,ok",  # missing applicant_name
    ])

    result = run_pipeline(csv_path, memory, warehouse, dlq_path, fail_rate=0.0, sleep=lambda s: None)

    assert result["loaded_this_run"] == 1
    assert result["rejected_this_run"] == 1
    assert warehouse.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0] == 1
    dlq_contents = dlq_path.read_text()
    assert "a2" in dlq_contents and "applicant_name is required" in dlq_contents


def test_grace_period_catches_a_late_arriving_row_after_watermark_advanced(tmp_path, memory, warehouse):
    """A classic late-arriving-data failure mode: a row whose
    updated_at is earlier than the current watermark must still be picked
    up, as long as it's within the grace period."""
    csv_path = tmp_path / "source.csv"
    dlq_path = tmp_path / "dlq.csv"
    _write_csv(csv_path, ["a1,2026-01-01T12:00:00+00:00,Alice,1000,ok"])
    run_pipeline(csv_path, memory, warehouse, dlq_path, fail_rate=0.0, sleep=lambda s: None)

    # a2 arrives with a timestamp BEFORE a1's (simulating a late write),
    # but still within the 0.25-day (6h) grace period behind the watermark.
    with csv_path.open("a") as f:
        f.write("a2,2026-01-01T07:00:00+00:00,Bob,2000,ok\n")
    second = run_pipeline(csv_path, memory, warehouse, dlq_path, fail_rate=0.0, sleep=lambda s: None)

    assert second["new_rows_seen"] == 1
    assert warehouse.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0] == 2


def test_retry_decorator_retries_only_transient_errors_and_eventually_raises():
    calls = {"n": 0}
    sleeps = []

    @retry_with_backoff(max_retries=2, base_delay=1.0, max_delay=4.0,
                        sleep=lambda s: sleeps.append(s), rand=lambda: 1.0)
    def always_fails_transiently():
        calls["n"] += 1
        raise TransientError("nope")

    with pytest.raises(TransientError):
        always_fails_transiently()
    assert calls["n"] == 3  # initial attempt + 2 retries
    assert sleeps == [1.0, 2.0]  # base_delay*2**0, base_delay*2**1, with rand()=1.0 (no jitter reduction)


def test_retry_decorator_does_not_retry_non_transient_errors():
    calls = {"n": 0}

    @retry_with_backoff(max_retries=5, sleep=lambda s: None)
    def raises_value_error():
        calls["n"] += 1
        raise ValueError("permanent failure, not retryable")

    with pytest.raises(ValueError):
        raises_value_error()
    assert calls["n"] == 1  # no retries attempted


def test_retry_decorator_succeeds_after_transient_failures():
    attempts = {"n": 0}

    @retry_with_backoff(max_retries=5, sleep=lambda s: None)
    def fails_twice_then_succeeds():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientError("still flaky")
        return "ok"

    assert fails_twice_then_succeeds() == "ok"
    assert attempts["n"] == 3


def test_mark_processed_is_a_noop_on_empty_rows(memory):
    before_wm = memory.get_param("watermark:x")
    mark_processed(memory, "x", [])
    assert memory.get_param("watermark:x") == before_wm
