"""
Trace — the per-request record of what each Gen-4 layer contributed.

Every service returns a trace with its decision, so any output can be
explained after the fact: which knowledge was cited, which live signals
were read, which past cases influenced it, which model (live or offline)
wrote the language, and which learned parameters were in force.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    retrieval: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def cite(self, passages) -> "Trace":
        self.retrieval = [{"source": p.source, "heading": p.heading, "score": round(p.score, 3)}
                          for p in passages]
        return self

    def note(self, msg: str) -> "Trace":
        self.notes.append(msg)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"retrieval": self.retrieval, "context": self.context, "memory": self.memory,
                "model": self.model, "params": self.params, "notes": self.notes}
