"""Tests against the real mmbert-small weights (downloaded from the Hub on first run).
Run on CPU:  CUDA_VISIBLE_DEVICES="" pytest tests/   (CHOPPER_VARIANT=base to test the base model)"""
import copy
import os

import numpy as np
import pytest
import torch

from chopper_rag import Chop, Chopper
from chopper_rag.core import kept_text, shape

QUESTION = "When was the Eiffel Tower completed?"
KEY = "It was designed by the engineering company of Gustave Eiffel and completed in 1889."
PASSAGES = [
    "Bananas are rich in potassium and are grown in tropical regions around the world.",
    "  The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. " + KEY
    + " It is 330 metres tall.\n",
    "Mount Everest is the highest mountain above sea level, located in the Mahalangur Himal of the Himalayas.",
    "The Great Wall of China is a series of fortifications built across the historical northern borders of China.",
]


@pytest.fixture(scope="session")
def chopper():
    return Chopper.from_pretrained(variant=os.environ.get("CHOPPER_VARIANT", "small"), device="cpu")


def _exact(chops, passages):
    for c in chops:
        assert isinstance(c, Chop)
        assert 0 <= c.start < c.end <= len(passages[c.passage_index])
        assert passages[c.passage_index][c.start:c.end] == c.text
        assert c.text == c.text.strip() and len(c.text) > 2


def test_relevant_sentence_kept_and_exact(chopper):
    r = chopper.run(QUESTION, PASSAGES)
    _exact(r.chops, PASSAGES)
    assert any("1889" in c.text and c.passage_index == 1 for c in r.chops)
    assert not any(c.passage_index == 0 for c in r.chops)        # bananas are irrelevant
    assert r.decision in ("ANSWER", "SEARCH") and 0.0 <= r.p_answer <= 1.0
    assert r.texts == [c.text for c in r.chops] and r.n_passes == 1


def test_matches_reference_operator(chopper):
    """identical text to the reference chop() in release/operator-for-llm-with-mmbert.py"""
    tk, enc, HD = chopper.tokenizer, chopper.encoder, chopper.heads
    head = f"Question: {QUESTION}"
    s = head + "".join(f" {tk.sep_token} " + p.strip() for p in PASSAGES)
    e = tk(s, return_offsets_mapping=True, truncation=True, max_length=8192, return_tensors="pt")
    off = e.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        h = enc(**e).last_hidden_state[0].float()
    p = torch.sigmoid(h @ HD["keep.weight"].T + HD["keep.bias"])[:, 0].numpy()
    body = np.array([a >= len(head) and b > a for a, b in off])
    q = shape((p > chopper.threshold) & body) & body
    ref = []
    for a, b in kept_text(s, off, q):
        ref += [t.strip() for t in s[a:b].split(tk.sep_token) if len(t.strip()) > 2]
    assert chopper.chop(QUESTION, PASSAGES) and [c.text for c in chopper.chop(QUESTION, PASSAGES)] == ref


def test_input_longer_than_budget_is_split(chopper):
    small = copy.copy(chopper)
    small.max_length = 128                                          # forces several passes and a windowed passage
    long_p = " ".join(["Filler sentence about nothing in particular number %d." % i for i in range(40)]) + " " + KEY
    passages = PASSAGES + [long_p]
    r = small.run(QUESTION, passages)
    assert r.n_passes > 1
    _exact(r.chops, passages)
    assert any("1889" in c.text for c in r.chops)


def test_empty_inputs(chopper):
    assert chopper.chop(QUESTION, []) == []
    assert chopper.chop(QUESTION, ["", "   "]) == []


def test_langchain(chopper):
    from langchain_core.documents import BaseDocumentCompressor, Document
    from chopper_rag.langchain import ChopperCompressor
    comp = ChopperCompressor(chopper=chopper)
    assert isinstance(comp, BaseDocumentCompressor)
    docs = [Document(page_content=p, metadata={"id": i, "src": "toy"}) for i, p in enumerate(PASSAGES)]
    out = comp.compress_documents(docs, QUESTION)
    assert out and all(isinstance(d, Document) for d in out)
    for d in out:
        m = d.metadata
        assert m["src"] == "toy" and m["id"] == m["source_index"]
        assert PASSAGES[m["source_index"]][m["chop_start"]:m["chop_end"]] == d.page_content
    assert any("1889" in d.page_content for d in out)
    assert comp.last_decision in ("ANSWER", "SEARCH")
    assert comp.compress_documents([], QUESTION) == []


def test_llama_index(chopper):
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeRelationship, NodeWithScore, TextNode
    from chopper_rag.llama_index import ChopperPostprocessor
    pp = ChopperPostprocessor(chopper=chopper)
    assert isinstance(pp, BaseNodePostprocessor)
    nodes = [NodeWithScore(node=TextNode(text=p, metadata={"src": "toy"}), score=1.0 - i / 10)
             for i, p in enumerate(PASSAGES)]
    out = pp.postprocess_nodes(nodes, query_str=QUESTION)
    assert out and all(isinstance(n, NodeWithScore) and isinstance(n.node, TextNode) for n in out)
    for n in out:
        m = n.node.metadata
        assert m["src"] == "toy"
        assert PASSAGES[m["source_index"]][m["chop_start"]:m["chop_end"]] == n.node.text
        assert n.node.relationships[NodeRelationship.SOURCE].node_id == nodes[m["source_index"]].node.node_id
        assert n.score == nodes[m["source_index"]].score
    assert any("1889" in n.node.text for n in out)
    with pytest.raises(ValueError):
        pp.postprocess_nodes(nodes)


def test_haystack(chopper):
    from haystack import Document, Pipeline
    from chopper_rag.haystack import ChopperDocumentCompressor
    comp = ChopperDocumentCompressor(variant="small", device="cpu")
    comp._chopper = chopper                                         # reuse the loaded model
    docs = [Document(content=p, meta={"src": "toy"}, score=0.5) for p in PASSAGES]
    res = comp.run(documents=docs, query=QUESTION)
    assert set(res) == {"documents", "decision"} and res["decision"] in ("ANSWER", "SEARCH")
    out = res["documents"]
    assert out and all(isinstance(d, Document) for d in out)
    for d in out:
        m = d.meta
        assert m["src"] == "toy" and m["source_id"] == docs[m["source_index"]].id
        assert PASSAGES[m["source_index"]][m["chop_start"]:m["chop_end"]] == d.content
    assert any("1889" in d.content for d in out)
    # serialisation round trip + use inside a Pipeline
    d = comp.to_dict()
    assert ChopperDocumentCompressor.from_dict(d).variant == "small"
    pipe = Pipeline()
    pipe.add_component("chopper", comp)
    r = pipe.run({"chopper": {"documents": docs, "query": QUESTION}})
    assert any("1889" in x.content for x in r["chopper"]["documents"])
