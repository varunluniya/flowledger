"""
Context layer — "What signals matter at this exact moment?"

Retrieval is what's always true; context is what's true *right now* for
this request: the live fields of the case, the current state of the queue
or portfolio, the time of day, the segment's recent behavior. A Context is
a named bag of signals, each with a short note on why it matters, so the
trace shows not just what the system saw but why it looked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Signal:
    name: str
    value: Any
    why: str = ""


@dataclass
class Context:
    signals: dict[str, Signal] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, name: str, value: Any, why: str = "") -> "Context":
        self.signals[name] = Signal(name, value, why)
        return self

    def get(self, name: str, default: Any = None) -> Any:
        s = self.signals.get(name)
        return s.value if s else default

    def as_dict(self) -> dict:
        return {k: {"value": s.value, "why": s.why} for k, s in self.signals.items()}

    def render(self) -> str:
        """Prompt-assembly view: one line per signal."""
        return "\n".join(f"- {s.name}: {s.value}" + (f"  ({s.why})" if s.why else "")
                         for s in self.signals.values())


def assemble_prompt(task: str, passages, context: Context, memory_notes: list[str],
                    output_contract: str) -> str:
    """Prompt Assembly stage: the only place the four layers meet the model.
    Keeping it in one function makes EQ(POST) fixes cheap -- a prompt change
    is an edit here, not a refactor."""
    knowledge = "\n\n".join(f"[{p.cite()}]\n{p.text}" for p in passages) or "(none retrieved)"
    memory = "\n".join(f"- {m}" for m in memory_notes) or "(no relevant history)"
    return (
        f"TASK\n{task}\n\n"
        f"KNOWLEDGE (cite by [source § heading])\n{knowledge}\n\n"
        f"CONTEXT (live signals)\n{context.render() or '(none)'}\n\n"
        f"MEMORY (what past cases taught us)\n{memory}\n\n"
        f"OUTPUT CONTRACT\n{output_contract}\n"
    )
