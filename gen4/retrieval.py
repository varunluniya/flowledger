"""
Retrieval layer — "What must the system know to be true?"

Loads a folder of Markdown knowledge files (policies, rubrics, protocols),
splits them into sections at every heading, and ranks sections with BM25.
Pure Python, no vector database: for policy-sized corpora (tens to a few
thousand sections) lexical BM25 is fast, explainable, and good enough.
Swap `search()` for an embedding index later without touching callers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:[.$%][0-9]+)?")
STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be been it this that with as at by "
    "from if then than so not no do does did has have had can will would should may might "
    "i you we they he she them our your their its what which who whom how when where why".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Passage:
    source: str      # file name
    heading: str     # nearest heading
    text: str
    score: float = 0.0

    def cite(self) -> str:
        return f"{self.source} § {self.heading}"

    def as_dict(self) -> dict:
        return {"source": self.source, "heading": self.heading,
                "text": self.text, "score": round(self.score, 3)}


class KnowledgeBase:
    def __init__(self, passages: list[Passage] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.passages: list[Passage] = []
        self._docs: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        for p in passages or []:
            self.add(p)

    # -- loading -------------------------------------------------------------
    @classmethod
    def from_dir(cls, path: str | Path) -> "KnowledgeBase":
        kb = cls()
        root = Path(path)
        if root.exists():
            for f in sorted(root.glob("**/*.md")):
                for heading, body in _split_sections(f.read_text(encoding="utf-8")):
                    if body.strip():
                        kb.add(Passage(f.name, heading, body.strip()))
        return kb

    def add(self, passage: Passage) -> None:
        tokens = tokenize(passage.heading + " " + passage.text)
        tf = Counter(tokens)
        self.passages.append(passage)
        self._docs.append(tf)
        self._df.update(tf.keys())
        n = len(self._docs)
        self._avgdl = ((self._avgdl * (n - 1)) + sum(tf.values())) / n

    # -- search --------------------------------------------------------------
    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[Passage]:
        q = tokenize(query)
        if not q or not self._docs:
            return []
        n = len(self._docs)
        scored = []
        for p, tf in zip(self.passages, self._docs):
            dl = sum(tf.values()) or 1
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if s > min_score:
                scored.append(Passage(p.source, p.heading, p.text, s))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.passages)


def _split_sections(markdown: str):
    heading, buf = "Overview", []
    for line in markdown.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if buf:
                yield heading, "\n".join(buf)
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if buf:
        yield heading, "\n".join(buf)
