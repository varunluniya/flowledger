"""
Helpers for seed scripts: deterministic synthetic history for a service's
memory database. Everything a seed writes is marked synthetic in the
knowledge graph (subject "dataset", predicate "seed"), and /health reports it.
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 9, 24, 9, 0, tzinfo=timezone.utc)


def iso(days_ago: float, now: datetime = NOW) -> str:
    return (now - timedelta(days=days_ago)).isoformat()


def args(default_db: str, description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--db", default=os.environ.get("GEN4_DB", default_db))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--if-empty", action="store_true", help="skip if the database already has decisions")
    p.add_argument("--reset", action="store_true", help="delete the database first")
    a = p.parse_args()
    if os.environ.get("GEN4_SEED", "1") == "0":
        print("GEN4_SEED=0 -- seeding disabled")
        raise SystemExit(0)
    if a.reset and os.path.exists(a.db):
        os.remove(a.db)
    return a, random.Random(a.seed)


def already_seeded(memory) -> bool:
    return memory.stats()["decisions"] > 0


def mark(memory, name: str, seed: int, counts: dict) -> None:
    memory.add_fact("dataset", "seed", {
        "synthetic": True, "generator": "seed.py", "service": name, "seed": seed,
        "generated_at": NOW.isoformat(), "counts": counts,
        "note": "Synthetic demo data. No real customers, patients or tickets."}, source="seed")
