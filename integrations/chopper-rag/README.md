# chopper-rag

The **chopper** as a drop-in context compressor for LangChain, LlamaIndex and Haystack.

The chopper is a fine-tuned mmBERT encoder. It reads the question and the snippets your retriever returned, and cuts out
**chops**: exact quotes cut out of retrieved snippets, never rewritten. A chop is typically 3-5% of the retrieved
text. Every chop records where it came from (`passage[start:end] == text`), and the chops work with any answering model.
It also gives one **ANSWER / SEARCH** decision: whether the snippets seem to contain the answer, or you should search again.

- Model: [huggingface.co/wiiiktor/chopper_mmbert](https://huggingface.co/wiiiktor/chopper_mmbert): `base` (0.31B, recommended) and `small` (0.14B)
- Code and demo: [github.com/wiiiktor/chopper_demo](https://github.com/wiiiktor/chopper_demo)

## Install

```bash
pip install chopper-rag                  # core only (torch, transformers)
pip install "chopper-rag[langchain]"     # or [llama-index], [haystack], [all]
```

## Core

```python
from chopper_rag import Chopper

chopper = Chopper.from_pretrained(variant="base")          # device="auto", threshold from the model config
result = chopper.run("When was the Eiffel Tower completed?", passages)
for c in result.chops:                                      # Chop(passage_index, start, end, text, score)
    assert passages[c.passage_index][c.start:c.end] == c.text
print(result.decision)                                      # "ANSWER" or "SEARCH"
chops = chopper.chop(question, passages)                    # just the list of Chop
```

## LangChain

```python
from langchain_classic.retrievers import ContextualCompressionRetriever  # langchain<1.0: langchain.retrievers
from chopper_rag.langchain import ChopperCompressor

compressor = ChopperCompressor.from_pretrained(variant="base")
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_kwargs={"k": 16}))
docs = retriever.invoke("When was the Eiffel Tower completed?")
for d in docs:   # one Document per chop
    print(d.page_content, d.metadata["source_index"], d.metadata["chop_start"], d.metadata["chop_end"])
# or directly: compressor.compress_documents(documents, query)
```

## LlamaIndex

```python
from chopper_rag.llama_index import ChopperPostprocessor

chopper = ChopperPostprocessor(variant="base")
query_engine = index.as_query_engine(similarity_top_k=16, node_postprocessors=[chopper])
response = query_engine.query("When was the Eiffel Tower completed?")
for n in response.source_nodes:   # one TextNode per chop, SOURCE relationship -> the retrieved node
    print(n.node.text, n.node.metadata["chop_start"], n.node.metadata["chop_end"])
# or directly: chopper.postprocess_nodes(nodes, query_str="...")
```

## Haystack (2.x and later)

```python
from haystack import Pipeline
from chopper_rag.haystack import ChopperDocumentCompressor

pipe = Pipeline()
pipe.add_component("retriever", retriever)                          # any retriever, top_k=16
pipe.add_component("chopper", ChopperDocumentCompressor(variant="base"))
pipe.connect("retriever.documents", "chopper.documents")
out = pipe.run({"retriever": {"query": q}, "chopper": {"query": q}})
print([d.content for d in out["chopper"]["documents"]], out["chopper"]["decision"])
```

## Output metadata (all three frameworks)

One output document / node per chop, in passage order. Metadata = the source's metadata plus:

| key | meaning |
|---|---|
| `chop_start`, `chop_end` | character offsets of the chop in the source text |
| `source_index` | position of the source in the input list |
| `chop_score` | mean p(KEEP) over the chop's tokens |
| `source_id` | (Haystack only) id of the source Document |

The score of each output is the score of its source. The last ANSWER / SEARCH decision is in `last_decision`
(LangChain, LlamaIndex) or in the `decision` output (Haystack). A source with no chops produces no output; if nothing
is kept, the result is empty (usually together with `SEARCH`).

## Details

- Input: `Question: <q> [SEP] <passage 1> [SEP] ... [SEP] <passage N>`, up to 8,192 tokens. Decoding: keep tokens
  with p(KEEP) > 0.5, merge runs at most 5 tokens apart, drop runs shorter than 4 tokens (from `chopper_config.json`;
  `threshold=` overrides the cut-off: lower keeps more).
- Inputs longer than 8,192 tokens are processed in several passes: passages are packed in order into groups that fit
  next to the question, and a single passage longer than the budget is cut into token windows. Offsets always refer
  to the original passage. The model was trained on ~20 passages of ~1,000 characters, which fit one pass.
- Leading/trailing whitespace of a passage is ignored; offsets still refer to the unstripped text.
- Device: `"auto"` = CUDA if available (bfloat16), else CPU (float32). Pass `repo=` a local directory holding
  `mmbert-base/` / `mmbert-small/` to run offline.

## Tests

```bash
pip install -e ".[all,test]"
CUDA_VISIBLE_DEVICES="" pytest tests/          # uses mmbert-small; CHOPPER_VARIANT=base for the base model
```

License: MIT.
