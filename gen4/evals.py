"""
Eval layer — Part 2 of the framework made executable.

An eval case is (id, input, expected, segment). `run_eval` calls the system
`runs` times per case and reports accuracy and consistency separately,
per segment, with every failure listed. `gate` turns a report into a
pass/fail used by CI: a pull request that makes the system worse fails.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class EvalCase:
    id: str
    input: Any
    expected: Any
    segment: str = "all"
    note: str = ""


@dataclass
class EvalReport:
    accuracy: float
    consistency: float
    n_cases: int
    runs: int
    by_segment: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"accuracy": self.accuracy, "consistency": self.consistency,
                "n_cases": self.n_cases, "runs": self.runs,
                "by_segment": self.by_segment, "failures": self.failures}


def run_eval(cases: list[EvalCase], system: Callable[[Any], Any], runs: int = 3,
             match: Callable[[Any, Any], bool] | None = None) -> EvalReport:
    match = match or (lambda got, exp: got == exp)
    seg_hits, seg_total, seg_cons = defaultdict(int), defaultdict(int), defaultdict(list)
    hits = total = 0
    cons_all, failures = [], []
    for case in cases:
        outs = []
        for r in range(runs):
            try:
                got = system(case.input)
            except Exception as e:  # a crash is a failure, not an abort
                got = f"<error: {e}>"
            outs.append(got)
            ok = match(got, case.expected)
            hits += ok
            total += 1
            seg_hits[case.segment] += ok
            seg_total[case.segment] += 1
            if not ok:
                failures.append({"id": case.id, "segment": case.segment, "run": r,
                                 "expected": case.expected, "got": got, "note": case.note})
        top = Counter(json.dumps(o, sort_keys=True, default=str) for o in outs).most_common(1)[0][1]
        cons = top / runs
        cons_all.append(cons)
        seg_cons[case.segment].append(cons)
    by_seg = {s: {"accuracy": round(seg_hits[s] / seg_total[s], 4),
                  "consistency": round(sum(seg_cons[s]) / len(seg_cons[s]), 4),
                  "n_cases": len(seg_cons[s])} for s in seg_total}
    return EvalReport(accuracy=round(hits / total, 4) if total else 0.0,
                      consistency=round(sum(cons_all) / len(cons_all), 4) if cons_all else 0.0,
                      n_cases=len(cases), runs=runs, by_segment=by_seg, failures=failures)


def gate(report: EvalReport, min_accuracy: float, min_consistency: float = 1.0,
         segment_floors: dict | None = None) -> tuple[bool, list[str]]:
    reasons = []
    if report.accuracy < min_accuracy:
        reasons.append(f"accuracy {report.accuracy:.1%} < floor {min_accuracy:.1%}")
    if report.consistency < min_consistency:
        reasons.append(f"consistency {report.consistency:.1%} < floor {min_consistency:.1%}")
    for seg, floor in (segment_floors or {}).items():
        got = report.by_segment.get(seg, {}).get("accuracy")
        if got is not None and got < floor:
            reasons.append(f"segment '{seg}' accuracy {got:.1%} < floor {floor:.1%}")
    return (not reasons), reasons


def print_and_exit(name: str, report: EvalReport, passed: bool, reasons: list[str],
                   out_path: str | None = "eval_report.json") -> None:
    print(f"== {name} eval ==")
    print(f"cases={report.n_cases} runs/case={report.runs} "
          f"accuracy={report.accuracy:.1%} consistency={report.consistency:.1%}")
    for seg, s in sorted(report.by_segment.items()):
        print(f"  {seg:<28} acc={s['accuracy']:.1%} cons={s['consistency']:.1%} n={s['n_cases']}")
    for f in report.failures[:10]:
        print(f"  FAIL {f['id']} run{f['run']}: expected={f['expected']!r} got={f['got']!r}")
    print("GATE:", "PASS" if passed else "FAIL -- " + "; ".join(reasons))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({**report.as_dict(), "gate": {"passed": passed, "reasons": reasons}},
                      fh, indent=2, default=str)
    sys.exit(0 if passed else 1)
