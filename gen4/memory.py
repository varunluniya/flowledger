"""
Memory layer — "What has the system earned from what happened before?"

One SQLite file holds four things:
    decisions   every decision the system made, with its input features
    outcomes    what actually happened afterwards (joined by decision id)
    params      learned parameters (thresholds, weights) with change history
    facts       a small knowledge graph of (subject, predicate, object) triples
                written by the feedback loop

SQLite is deliberate: zero-ops, file-based, fine for a single service
instance, and trivially swapped for Postgres by re-implementing this class.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY, system TEXT, subject TEXT, ts TEXT,
    features TEXT, output TEXT
);
CREATE INDEX IF NOT EXISTS ix_decisions_subject ON decisions(subject);
-- decisions(system=..., limit=...) orders by d.ts DESC with a
-- LIMIT, which looks bounded but without this index SQLite still does a
-- full table SCAN + a temp B-tree sort to produce that order (verified via
-- EXPLAIN QUERY PLAN) -- cost grows with total decision count regardless
-- of how small the LIMIT or result window is. This index lets it SEARCH
-- by system and walk straight in ts-descending order instead.
CREATE INDEX IF NOT EXISTS ix_decisions_system_ts ON decisions(system, ts DESC);
CREATE TABLE IF NOT EXISTS outcomes (
    decision_id TEXT PRIMARY KEY, ts TEXT, outcome TEXT
);
CREATE TABLE IF NOT EXISTS params (
    name TEXT PRIMARY KEY, value TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS param_history (
    name TEXT, value TEXT, reason TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    subject TEXT, predicate TEXT, object TEXT, weight REAL, source TEXT, ts TEXT,
    PRIMARY KEY (subject, predicate)
);
CREATE TABLE IF NOT EXISTS fact_history (
    subject TEXT, predicate TEXT, object TEXT, weight REAL, source TEXT, ts TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Memory:
    def __init__(self, path: str | os.PathLike = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._tx() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # -- decisions & outcomes -------------------------------------------------
    def record_decision(self, system: str, subject: str, features: dict, output: dict,
                        decision_id: str | None = None, ts: str | None = None) -> str:
        did = decision_id or uuid.uuid4().hex[:12]
        with self._tx() as c:
            c.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)",
                      (did, system, subject, ts or _now(), json.dumps(features, default=str),
                       json.dumps(output, default=str)))
        return did

    def record_outcome(self, decision_id: str, outcome: dict, ts: str | None = None) -> bool:
        with self._tx() as c:
            c.execute("SELECT 1 FROM decisions WHERE id=?", (decision_id,))
            if not c.fetchone():
                return False
            c.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?)",
                      (decision_id, ts or _now(), json.dumps(outcome, default=str)))
        return True

    def backdate(self, decision_id: str, ts: str, outcome_ts: str | None = None) -> None:
        """Move a decision (and optionally its outcome) into the past -- used by
        seed scripts so synthetic history has realistic timestamps."""
        with self._tx() as c:
            c.execute("UPDATE decisions SET ts=? WHERE id=?", (ts, decision_id))
            if outcome_ts:
                c.execute("UPDATE outcomes SET ts=? WHERE decision_id=?", (outcome_ts, decision_id))

    def get_decision(self, decision_id: str) -> dict | None:
        rows = self._select("WHERE d.id=?", (decision_id,))
        return rows[0] if rows else None

    def history(self, subject: str, limit: int = 50) -> list[dict]:
        return self._select("WHERE d.subject=? ORDER BY d.ts DESC LIMIT ?", (subject, limit))

    def decisions(self, system: str | None = None, with_outcome: bool | None = None,
                  limit: int = 10_000, feature: tuple[str, Any] | None = None,
                  outcome_since: str | None = None) -> list[dict]:
        """`feature=("dealer_id", "D-13")` filters on a top-level feature value in SQL.

        `outcome_since` is the incremental-loading hook: pass a timestamp to
        get back only rows whose OUTCOME arrived after it (not the decision
        itself -- outcomes routinely land long after their decision, so this
        is the column that actually needs watermarking). Implies
        `with_outcome=True` unless explicitly overridden. See
        `gen4/feedback.py`'s `new_rows_since()` for the idempotent,
        grace-period-aware wrapper that should normally be used instead of
        calling this directly with a raw cutoff."""
        where, args = [], []
        if system:
            where.append("d.system=?")
            args.append(system)
        if feature:
            where.append("json_extract(d.features, ?) = ?")
            args += [f"$.{feature[0]}", feature[1]]
        if outcome_since is not None:
            where.append("o.ts > ?")
            args.append(outcome_since)
            if with_outcome is None:
                with_outcome = True
        if with_outcome is True:
            where.append("o.decision_id IS NOT NULL")
        elif with_outcome is False:
            where.append("o.decision_id IS NULL")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        return self._select(f"{clause} ORDER BY d.ts DESC LIMIT ?", (*args, limit))

    def _select(self, clause: str, args: Iterable[Any]) -> list[dict]:
        with self._tx() as c:
            c.execute("SELECT d.*, o.outcome AS outcome, o.ts AS outcome_ts FROM decisions d "
                      "LEFT JOIN outcomes o ON o.decision_id = d.id " + clause, tuple(args))
            rows = c.fetchall()
        return [{
            "id": r["id"], "system": r["system"], "subject": r["subject"], "ts": r["ts"],
            "features": json.loads(r["features"]), "output": json.loads(r["output"]),
            "outcome": json.loads(r["outcome"]) if r["outcome"] else None,
            "outcome_ts": r["outcome_ts"],
        } for r in rows]

    def similar(self, features: dict, keys: list[str], system: str | None = None,
                k: int = 5, require_outcome: bool = True, scales: dict | None = None) -> list[dict]:
        """k nearest past decisions on the given numeric feature keys
        (scaled Euclidean distance). Used to answer "what happened to cases
        like this one?" -- the memory prior."""
        scales = scales or {}
        pool = self.decisions(system=system, with_outcome=True if require_outcome else None)
        out = []
        for d in pool:
            f = d["features"]
            try:
                dist = math.sqrt(sum(((float(f[key]) - float(features[key])) / scales.get(key, 1.0)) ** 2
                                     for key in keys))
            except (KeyError, TypeError, ValueError):
                continue
            out.append({**d, "distance": round(dist, 4)})
        out.sort(key=lambda d: d["distance"])
        return out[:k]

    # -- learned parameters ------------------------------------------------------
    def get_param(self, name: str, default: Any = None) -> Any:
        with self._tx() as c:
            c.execute("SELECT value FROM params WHERE name=?", (name,))
            row = c.fetchone()
        return json.loads(row["value"]) if row else default

    def set_param(self, name: str, value: Any, reason: str = "", ts: str | None = None) -> None:
        v = json.dumps(value)
        with self._tx() as c:
            c.execute("INSERT OR REPLACE INTO params VALUES (?,?,?)", (name, v, ts or _now()))
            c.execute("INSERT INTO param_history VALUES (?,?,?,?)", (name, v, reason, ts or _now()))

    def param_history(self, name: str | None = None) -> list[dict]:
        with self._tx() as c:
            if name:
                c.execute("SELECT * FROM param_history WHERE name=? ORDER BY ts", (name,))
            else:
                c.execute("SELECT * FROM param_history ORDER BY ts")
            rows = c.fetchall()
        return [{"name": r["name"], "value": json.loads(r["value"]), "reason": r["reason"],
                 "ts": r["ts"]} for r in rows]

    # -- knowledge graph ---------------------------------------------------------
    def add_fact(self, subject: str, predicate: str, obj: Any, weight: float = 1.0,
                 source: str = "feedback") -> None:
        """Upserts the CURRENT value of (subject, predicate) in `facts` --
        but also appends it to `fact_history`, so a value that gets
        recomputed and overwritten repeatedly (a calibration result, an
        override-rate snapshot) still has a recoverable history instead of
        only ever showing the latest write. Previously `facts` alone lost
        that history on every overwrite -- the SCD Type 2 pattern `params`/
        `param_history` already used, just missing here."""
        ts = _now()
        with self._tx() as c:
            c.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?)",
                      (subject, predicate, json.dumps(obj, default=str), weight, source, ts))
            c.execute("INSERT INTO fact_history VALUES (?,?,?,?,?,?)",
                      (subject, predicate, json.dumps(obj, default=str), weight, source, ts))

    def facts(self, subject: str | None = None, predicate: str | None = None) -> list[dict]:
        where, args = [], []
        if subject:
            where.append("subject=?")
            args.append(subject)
        if predicate:
            where.append("predicate=?")
            args.append(predicate)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._tx() as c:
            c.execute(f"SELECT * FROM facts {clause} ORDER BY subject, predicate", tuple(args))
            rows = c.fetchall()
        return [{"subject": r["subject"], "predicate": r["predicate"],
                 "object": json.loads(r["object"]), "weight": r["weight"],
                 "source": r["source"], "ts": r["ts"]} for r in rows]

    def fact(self, subject: str, predicate: str, default: Any = None) -> Any:
        f = self.facts(subject, predicate)
        return f[0]["object"] if f else default

    def fact_history(self, subject: str | None = None, predicate: str | None = None) -> list[dict]:
        """Every value (subject, predicate) has ever held, oldest first --
        the recovery path for a fact that `add_fact` has since overwritten."""
        where, args = [], []
        if subject:
            where.append("subject=?")
            args.append(subject)
        if predicate:
            where.append("predicate=?")
            args.append(predicate)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._tx() as c:
            c.execute(f"SELECT * FROM fact_history {clause} ORDER BY ts", tuple(args))
            rows = c.fetchall()
        return [{"subject": r["subject"], "predicate": r["predicate"],
                 "object": json.loads(r["object"]), "weight": r["weight"],
                 "source": r["source"], "ts": r["ts"]} for r in rows]

    def stats(self) -> dict:
        with self._tx() as c:
            counts = {}
            for t in ("decisions", "outcomes", "params", "facts", "fact_history"):
                c.execute(f"SELECT COUNT(*) AS n FROM {t}")
                counts[t] = c.fetchone()["n"]
        return counts
