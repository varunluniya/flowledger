"""
Feedback layer — "How does the system get smarter from its own output?"

Small, auditable primitives. The rule every service follows:
  * Learning moves parameters in small, bounded steps (never a jump).
  * Every change is written to Memory.param_history with its reason.
  * Anything with regulatory or safety weight is *proposed*, not applied,
    until a human approves it (see each service's /feedback/proposals).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


def new_rows_since(memory, system: str, watermark_key: str, grace_days: float = 0.0,
                   limit: int = 10_000, feature: tuple | None = None) -> list[dict]:
    """The incremental-loading fix for `learn()`-style functions: instead
    of re-pulling and re-aggregating every decision that has ever had an
    outcome, pull only the outcomes that arrived after the last watermark,
    with a `grace_days` buffer to catch outcomes that were recorded with a
    timestamp earlier than the current watermark (a classic late-arriving-
    data gap -- these services genuinely have that shape, since an
    outcome's timestamp is backdated to when the underlying event
    happened, which can be well before the watermark that was already
    advanced past it).

    The grace-period re-pull necessarily re-returns some already-processed
    rows -- that overlap is handled by `mark_processed()` below via a
    persisted set of decision ids, not by the timestamp filter alone,
    which is what keeps repeated calls idempotent: safe to re-run without
    double-counting."""
    wm = memory.get_param(f"watermark:{watermark_key}")
    since = None
    if wm:
        since = (datetime.fromisoformat(wm) - timedelta(days=grace_days)).isoformat()
    candidates = memory.decisions(system=system, with_outcome=True, outcome_since=since,
                                  limit=limit, feature=feature)
    processed = set(memory.get_param(f"processed:{watermark_key}", []))
    return [r for r in candidates if r["id"] not in processed]


def mark_processed(memory, watermark_key: str, rows: list[dict]) -> None:
    """Commit a batch `new_rows_since()` returned: extend the processed-id
    set (so a grace-period re-pull of the same rows next time is a no-op)
    and advance the watermark to the latest outcome timestamp actually seen
    -- never further, and only after the caller's merge into its running
    aggregate has already succeeded, so a failure midway never loses or
    duplicates anything on the next attempt."""
    if not rows:
        return
    processed = set(memory.get_param(f"processed:{watermark_key}", []))
    processed.update(r["id"] for r in rows)
    memory.set_param(f"processed:{watermark_key}", sorted(processed),
                     reason="incremental learn(): mark rows processed")
    latest = max(r["outcome_ts"] for r in rows if r.get("outcome_ts"))
    memory.set_param(f"watermark:{watermark_key}", latest,
                     reason="incremental learn(): advance watermark")


def accumulate(memory, key: str, **deltas: float) -> dict:
    """Merge numeric deltas (counts, sums) into a persisted running total
    under `accum:{key}`, and return the updated totals. The point: a
    `learn()`-style function should only ever need to know the delta its
    NEW batch contributes -- never re-derive the running total by re-reading
    every historical row, which is the full-load pattern this whole fix
    replaces."""
    state = memory.get_param(f"accum:{key}", {})
    for k, v in deltas.items():
        state[k] = state.get(k, 0) + v
    memory.set_param(f"accum:{key}", state, reason="incremental learn() update")
    return state


def incremental_calibration(memory, key: str, new_pairs: list[tuple[float, int]], bins: int = 5) -> dict:
    """Same contract as `calibration()` below, but the result is built from
    a persisted per-bin running state (n, sum of predicted, sum of observed)
    updated with only the NEW pairs -- so repeated calls never need to
    re-read or re-bin a single old decision+outcome row."""
    state = memory.get_param(f"calstate:{key}", {
        "bins": [{"n": 0, "sum_p": 0.0, "sum_y": 0.0} for _ in range(bins)],
        "n": 0, "sum_sq_err": 0.0,
    })
    for p, y in new_pairs:
        i = min(bins - 1, max(0, int(p * bins)))
        b = state["bins"][i]
        b["n"] += 1
        b["sum_p"] += p
        b["sum_y"] += y
        state["n"] += 1
        state["sum_sq_err"] += (p - y) ** 2
    memory.set_param(f"calstate:{key}", state, reason="incremental calibration update")
    if state["n"] == 0:
        return {"n": 0, "brier": None, "ece": None, "bins": []}
    out, ece = [], 0.0
    for i, b in enumerate(state["bins"]):
        if b["n"] == 0:
            continue
        pred, obs = b["sum_p"] / b["n"], b["sum_y"] / b["n"]
        ece += abs(pred - obs) * b["n"] / state["n"]
        lo, hi = i / bins, (i + 1) / bins
        out.append({"range": f"{lo:.1f}-{hi:.1f}", "n": b["n"],
                    "predicted": round(pred, 3), "observed": round(obs, 3)})
    brier = state["sum_sq_err"] / state["n"]
    return {"n": state["n"], "brier": round(brier, 4), "ece": round(ece, 4), "bins": out}


def bounded_step(current: float, target: float, rate: float, lo: float, hi: float,
                 max_step: float | None = None) -> float:
    """Move `current` toward `target` by `rate` (0..1), clamp to [lo, hi],
    and never move more than `max_step` in one update."""
    delta = (target - current) * rate
    if max_step is not None:
        delta = max(-max_step, min(max_step, delta))
    return max(lo, min(hi, current + delta))


def calibration(pairs: list[tuple[float, int]], bins: int = 5) -> dict:
    """pairs = (predicted probability of the positive class, actual 0/1).
    Returns per-bin predicted vs observed rates, the Brier score, and the
    expected calibration error (ECE)."""
    if not pairs:
        return {"n": 0, "brier": None, "ece": None, "bins": []}
    out, ece = [], 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (i == bins - 1 and p == 1.0)]
        if not sel:
            continue
        pred = sum(p for p, _ in sel) / len(sel)
        obs = sum(y for _, y in sel) / len(sel)
        ece += abs(pred - obs) * len(sel) / len(pairs)
        out.append({"range": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                    "predicted": round(pred, 3), "observed": round(obs, 3)})
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    return {"n": len(pairs), "brier": round(brier, 4), "ece": round(ece, 4), "bins": out}


def rate(numer: int, denom: int) -> float | None:
    return round(numer / denom, 4) if denom else None


def _wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def wilson_upper(successes: int, n: int, z: float = 1.96) -> float:
    """Plausible worst case of a rate -- use to decide something is *safe*."""
    return _wilson(successes, n, z)[1]


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    """Plausible best case of a rate -- use to decide something is *broken*
    (a circuit breaker should trip on evidence, not on small-sample noise)."""
    return _wilson(successes, n, z)[0]


def propose(memory, param: str, current, proposed, reason: str, evidence: dict | None = None) -> dict:
    """Store a human-approval-required parameter change. Returns the proposal."""
    p = {"param": param, "current": current, "proposed": proposed, "reason": reason,
         "evidence": evidence or {}, "status": "pending"}
    memory.add_fact(f"proposal:{param}", "pending", p, source="feedback")
    return p


def pending_proposals(memory) -> list[dict]:
    return [f["object"] for f in memory.facts(predicate="pending")
            if f["subject"].startswith("proposal:") and f["object"].get("status") == "pending"]


def approve(memory, param: str, approver: str = "human") -> dict | None:
    p = memory.fact(f"proposal:{param}", "pending")
    if not p or p.get("status") != "pending":
        return None
    memory.set_param(param, p["proposed"], reason=f"approved by {approver}: {p['reason']}")
    p["status"] = "approved"
    memory.add_fact(f"proposal:{param}", "pending", p, source=approver)
    return p


@dataclass
class Guardrail:
    """Circuit breaker on an observed bad-outcome rate.

    Trips when the Wilson *lower* bound of the bad rate exceeds `limit`
    (i.e. we are ~95% confident the true rate is above the limit) and at
    least `min_n` outcomes have been observed. The upper bound is reported
    too, as "could still be above the limit" -- a warning, not a trip."""
    name: str
    limit: float
    min_n: int = 20

    def check(self, bad: int, n: int) -> dict:
        lo, hi = wilson_lower(bad, n), wilson_upper(bad, n)
        tripped = n >= self.min_n and lo > self.limit
        warning = n >= self.min_n and not tripped and hi > self.limit
        status = ("insufficient_data" if n < self.min_n else
                  "TRIPPED" if tripped else "watch" if warning else "ok")
        return {"guardrail": self.name, "n": n, "bad": bad, "observed_rate": rate(bad, n),
                "lower_bound": round(lo, 4), "upper_bound": round(hi, 4), "limit": self.limit,
                "tripped": tripped, "status": status}
