"""LlamaIndex: ``ChopperPostprocessor`` (a ``BaseNodePostprocessor``), e.g. ``node_postprocessors=[...]`` of a
query engine."""
from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeRelationship, NodeWithScore, QueryBundle, TextNode

from .core import DEFAULT_REPO, Chopper


class ChopperPostprocessor(BaseNodePostprocessor):
    """Replaces the retrieved nodes by one TextNode per chop (the exact quote). Metadata = the source node's
    metadata + ``chop_start``/``chop_end`` (character offsets in the source node's text), ``source_index`` and
    ``chop_score``; the SOURCE relationship points to the source node; the score is the source node's score.
    The ANSWER / SEARCH decision of the last call is in ``last_decision``."""

    repo: str = Field(default=DEFAULT_REPO, description="HF repo id or local directory of the chopper weights")
    variant: str = Field(default="base", description="'base' (0.31B, recommended) or 'small' (0.14B)")
    device: str = Field(default="auto")
    threshold: Optional[float] = Field(default=None, description="p(KEEP) cut-off; default from the model config")
    last_decision: Optional[str] = None
    _chopper: Any = PrivateAttr(default=None)

    def __init__(self, chopper: Optional[Chopper] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._chopper = chopper or Chopper.from_pretrained(self.repo, variant=self.variant, device=self.device,
                                                           threshold=self.threshold)

    @classmethod
    def class_name(cls) -> str:
        return "ChopperPostprocessor"

    def _postprocess_nodes(self, nodes: List[NodeWithScore],
                           query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("ChopperPostprocessor needs the query (query_bundle or query_str)")
        if not nodes:
            return []
        texts = [n.node.get_content(metadata_mode=MetadataMode.NONE) for n in nodes]
        res = self._chopper.run(query_bundle.query_str, texts)
        self.last_decision = res.decision
        out = []
        for c in res.chops:
            src = nodes[c.passage_index]
            node = TextNode(
                text=c.text,
                metadata={**src.node.metadata, "chop_start": c.start, "chop_end": c.end,
                          "source_index": c.passage_index, "chop_score": c.score},
                excluded_embed_metadata_keys=list(src.node.excluded_embed_metadata_keys),
                excluded_llm_metadata_keys=list(src.node.excluded_llm_metadata_keys),
                relationships={NodeRelationship.SOURCE: src.node.as_related_node_info()},
            )
            out.append(NodeWithScore(node=node, score=src.score))
        return out
