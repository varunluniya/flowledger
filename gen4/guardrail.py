"""
Guardrail layer -- post-model output validation.

Retrieval and context cut the odds of a wrong answer, but nothing about them
stops a fluent, well-cited-sounding response from still contradicting or
misapplying the very policy passages it was grounded in. This is exactly the
layer the Air Canada chatbot was missing: it retrieved
nothing and had no check before answering, so a confident, wrong answer
reached the customer and the tribunal held the airline liable for it.

`guardrail_check` runs *after* the model answers and *before* the answer
reaches anyone, using the same retrieved passages as the yardstick -- the
standard retrieve -> respond -> guardrail-check pattern:

    is_safe, reason = guardrail_check(response, passages, llm, offline=...)
    return response if is_safe else fallback_message(reason)

`offline` is a repo-specific, rule-based stand-in for the live model call
(same honest-disclosure pattern as every other LLM call in this codebase --
see gen4/llm.py). It gets the response text and the retrieved passages and
returns (is_safe, reason); it is what actually runs in this environment
since no external LLM API key is configured, and it is also what runs if a
live call ever fails or returns something unusable.
"""

from __future__ import annotations

from typing import Callable


def guardrail_check(response: str, passages, llm,
                     offline: Callable[[str, list], tuple[bool, str]]) -> tuple[bool, str]:
    """Returns (is_safe, reason). is_safe=True means the response does not
    contradict or misapply the cited policy passages; otherwise `reason`
    names the specific violation."""
    knowledge = "\n\n".join(f"[{p.cite()}]\n{p.text}" for p in passages) or "(none retrieved)"

    def _offline() -> str:
        ok, reason = offline(response, passages)
        return "SAFE" if ok else f"VIOLATION: {reason}"

    prompt = (
        f"Policy:\n{knowledge}\n\n"
        f"Response:\n{response}\n\n"
        f"Does this response contradict or misapply the policy above?\n"
        f"Answer: SAFE or VIOLATION: [reason]"
    )
    result = llm.complete(prompt, offline=_offline, max_tokens=80).strip()
    if result.startswith("SAFE"):
        return True, ""
    return False, result.replace("VIOLATION:", "").strip()
