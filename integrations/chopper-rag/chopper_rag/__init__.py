"""chopper-rag: the chopper context compressor (exact quotes cut out of retrieved snippets) for RAG pipelines.

Framework adapters live in submodules so that importing this package needs none of the frameworks:
    chopper_rag.langchain.ChopperCompressor             (langchain-core)
    chopper_rag.llama_index.ChopperPostprocessor        (llama-index-core)
    chopper_rag.haystack.ChopperDocumentCompressor      (haystack-ai)
"""
from .core import Chop, ChopResult, Chopper

__all__ = ["Chop", "ChopResult", "Chopper"]
__version__ = "0.1.0"
