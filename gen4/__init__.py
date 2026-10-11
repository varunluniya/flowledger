"""
gen4 — a small, dependency-free kernel for the Generative-4 architecture.

    User Query -> Retrieval -> Context -> Memory -> Prompt Assembly
               -> Model -> Response -> Feedback -> Knowledge Graph

Each layer answers one question:
    Retrieval  What must the system know to be true?      (retrieval.KnowledgeBase)
    Context    What signals matter at this exact moment?  (context.Context)
    Memory     What has the system earned from the past?  (memory.Memory)
    Feedback   How does it get smarter from its output?   (feedback.*, memory facts)

The kernel is vendored (copied) into each service so every repo stays
self-contained. Nothing here needs a network connection or an API key:
llm.LLMClient uses a live model when a key is configured and a deterministic
offline fallback otherwise, and says which one it used in every trace.
"""

from .retrieval import KnowledgeBase, Passage
from .context import Context
from .memory import Memory
from .llm import LLMClient
from .trace import Trace
from .guardrail import guardrail_check
from . import feedback, evals

__all__ = [
    "KnowledgeBase", "Passage", "Context", "Memory", "LLMClient", "Trace",
    "feedback", "evals", "guardrail_check",
]
__version__ = "1.0.0"
