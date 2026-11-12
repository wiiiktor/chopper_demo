"""Haystack 2.x / 3.x: ``ChopperDocumentCompressor`` component, placed after a retriever or ranker."""
from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict

from .core import DEFAULT_REPO, Chopper


@component
class ChopperDocumentCompressor:
    """Replaces the documents by one Document per chop (the exact quote). ``meta`` = the source document's meta +
    ``chop_start``/``chop_end`` (character offsets in the source ``content``), ``source_index``, ``source_id`` and
    ``chop_score``; ``score`` = the source document's score. Outputs ``documents`` and ``decision``
    ("ANSWER" or "SEARCH": whether the documents hold the answer)."""

    def __init__(self, repo: str = DEFAULT_REPO, variant: str = "base", device: str = "auto",
                 threshold: float | None = None) -> None:
        self.repo, self.variant, self.device, self.threshold = repo, variant, device, threshold
        self._chopper: Chopper | None = None

    def warm_up(self) -> None:
        if self._chopper is None:
            self._chopper = Chopper.from_pretrained(self.repo, variant=self.variant, device=self.device,
                                                    threshold=self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, repo=self.repo, variant=self.variant, device=self.device,
                               threshold=self.threshold)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChopperDocumentCompressor":
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], decision=str)
    def run(self, documents: list[Document], query: str) -> dict[str, Any]:
        self.warm_up()
        if not documents:
            return {"documents": [], "decision": "SEARCH"}
        res = self._chopper.run(query, [d.content or "" for d in documents])
        out = [Document(content=c.text, score=documents[c.passage_index].score,
                        meta={**documents[c.passage_index].meta, "chop_start": c.start, "chop_end": c.end,
                              "source_index": c.passage_index, "source_id": documents[c.passage_index].id,
                              "chop_score": c.score})
               for c in res.chops]
        return {"documents": out, "decision": res.decision}
