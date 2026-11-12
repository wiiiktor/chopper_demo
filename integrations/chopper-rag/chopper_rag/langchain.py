"""LangChain: ``ChopperCompressor`` (a ``BaseDocumentCompressor``), e.g. inside a ``ContextualCompressionRetriever``."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict

from .core import DEFAULT_REPO, Chopper


class ChopperCompressor(BaseDocumentCompressor):
    """Returns one Document per chop: ``page_content`` = the exact quote, ``metadata`` = the source document's
    metadata + ``chop_start``/``chop_end`` (character offsets in the source ``page_content``), ``source_index``
    (position of the source in the input list) and ``chop_score`` (mean p(KEEP)).
    The ANSWER / SEARCH decision of the last call is in ``last_decision``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chopper: Any = None
    last_decision: str | None = None

    @classmethod
    def from_pretrained(cls, repo: str = DEFAULT_REPO, variant: str = "base", device: str = "auto",
                        threshold: float | None = None) -> "ChopperCompressor":
        return cls(chopper=Chopper.from_pretrained(repo, variant=variant, device=device, threshold=threshold))

    def model_post_init(self, __context: Any) -> None:
        if self.chopper is None:
            self.chopper = Chopper.from_pretrained()

    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Callbacks | None = None) -> Sequence[Document]:
        documents = list(documents)
        if not documents:
            return []
        res = self.chopper.run(query, [d.page_content for d in documents])
        self.last_decision = res.decision
        return [Document(page_content=c.text,
                         metadata={**documents[c.passage_index].metadata, "chop_start": c.start, "chop_end": c.end,
                                   "source_index": c.passage_index, "chop_score": c.score})
                for c in res.chops]
