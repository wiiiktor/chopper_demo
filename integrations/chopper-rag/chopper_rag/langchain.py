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
    (position of the source in the input list), ``chop_score`` (mean p(KEEP)) and ``source_relevance``.
    ``mode="rerank"``: returns the input documents instead, most relevant first (``top_n`` of them), with
    ``metadata["relevance_score"]``: the chopper as a drop-in reranker, from the same single pass.
    The ANSWER / SEARCH decision of the last call is in ``last_decision``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chopper: Any = None
    mode: str = "chop"                    # "chop" (exact quotes) or "rerank" (whole documents, most relevant first)
    top_n: int | None = None              # rerank mode: how many documents to return (None = all)
    last_decision: str | None = None

    @classmethod
    def from_pretrained(cls, repo: str = DEFAULT_REPO, variant: str = "base", device: str = "auto",
                        threshold: float | None = None, mode: str = "chop", top_n: int | None = None) -> "ChopperCompressor":
        return cls(chopper=Chopper.from_pretrained(repo, variant=variant, device=device, threshold=threshold),
                   mode=mode, top_n=top_n)

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
        if self.mode == "rerank":
            return [Document(page_content=documents[i].page_content,
                             metadata={**documents[i].metadata, "relevance_score": r, "source_index": i})
                    for i, r in res.ranking(self.top_n)]
        return [Document(page_content=c.text,
                         metadata={**documents[c.passage_index].metadata, "chop_start": c.start, "chop_end": c.end,
                                   "source_index": c.passage_index, "chop_score": c.score,
                                   "source_relevance": res.passage_scores[c.passage_index]})
                for c in res.chops]
