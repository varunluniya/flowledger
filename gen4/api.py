"""
Shared FastAPI routes every Gen-4 service exposes (imported only by app.py,
so the rest of the kernel stays dependency-free).

    GET  /health                       liveness + which model provider is active
    GET  /knowledge/search?q=...       Retrieval layer, directly
    GET  /memory/stats                 Memory layer counts
    GET  /memory/facts                 Knowledge graph written by the feedback loop
    GET  /params                       learned parameters + change history
    GET  /proposals                    parameter changes awaiting human approval
    POST /proposals/{param}/approve    a human approves one
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import __version__
from . import feedback as fb


def common_router(service, name: str) -> APIRouter:
    r = APIRouter()

    @r.get("/health", tags=["ops"])
    def health():
        return {"status": "ok", "service": name, "gen4": __version__,
                "llm_provider": service.llm.provider, "knowledge_sections": len(service.kb),
                "memory": service.memory.stats(),
                "dataset": service.memory.fact("dataset", "seed")}

    @r.get("/knowledge/search", tags=["gen4"])
    def knowledge_search(q: str, k: int = 3):
        return [p.as_dict() for p in service.kb.search(q, k=k)]

    @r.get("/memory/stats", tags=["gen4"])
    def memory_stats():
        return service.memory.stats()

    @r.get("/memory/facts", tags=["gen4"])
    def memory_facts(subject: str | None = None):
        return service.memory.facts(subject=subject)

    @r.get("/params", tags=["feedback"])
    def params():
        return {"history": service.memory.param_history()}

    @r.get("/proposals", tags=["feedback"])
    def proposals():
        return fb.pending_proposals(service.memory)

    @r.post("/proposals/{param}/approve", tags=["feedback"])
    def approve(param: str, approver: str = "human"):
        p = fb.approve(service.memory, param, approver)
        if not p:
            raise HTTPException(404, f"no pending proposal for {param}")
        return p

    return r
