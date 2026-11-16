# chopper-rag

<!-- mcp-name: io.github.wiiiktor/chopper -->

The **chopper** as a drop-in context compressor **and reranker** for LangChain, LlamaIndex and Haystack, and as an MCP
server for Claude.

The chopper is a fine-tuned mmBERT encoder. It reads the question and the snippets your retriever returned, and cuts out
**chops**: exact quotes cut out of retrieved snippets, never rewritten. A chop is typically 3-5% of the retrieved
text. Every chop records where it came from (`passage[start:end] == text`), and the chops work with any answering model.
It also gives one **ANSWER / SEARCH** decision: whether the snippets seem to contain the answer, or you should search again.

**It is also a reranker.** The same single pass scores every passage (mean p(KEEP) of its 16 most likely tokens). On a
held-out set in 8 languages (400 queries, 16 passages each) it ranks almost as well as dedicated rerankers, at the speed
of bge-reranker-v2-m3, with the chops and the decision included:

| | nDCG@10 | Hit@1 | GPU time for 16 passages |
|---|---|---|---|
| **chopper (mmBERT-base, one pass)** | **0.973** | **0.972** | **0.20 s**, chops included |
| bge-reranker-v2-m3 | 0.984 | 0.985 | 0.20 s |
| Qwen3-Reranker-0.6B | 0.977 | 0.972 | 0.77 s |
| Qwen3-Reranker-4B | 0.986 | 0.990 | 3.3 s |

So in most pipelines you can drop the reranker: retrieve, then rank and chop in one call. Details and caveats:
[DETAILS.md](https://github.com/wiiiktor/chopper_demo/blob/main/docs/DETAILS.md#the-chopper-as-a-reranker).

- Model: [huggingface.co/wiiiktor/chopper_mmbert](https://huggingface.co/wiiiktor/chopper_mmbert): `base` (0.31B, recommended) and `small` (0.14B); for Polish documents `base-pl` and `small-pl`, and `plmodernbert-base` (0.15B, on OPI PIB's pl-ModernBERT-base, Apache 2.0) for Polish-only text: better Polish answers (two-hop 56% vs 43% with `base-pl`), slightly weaker Polish ranking, weak outside Polish
- Code and demo: [github.com/wiiiktor/chopper_demo](https://github.com/wiiiktor/chopper_demo)

## Install

```bash
pip install chopper-rag                  # core only (torch, transformers)
pip install "chopper-rag[langchain]"     # or [llama-index], [haystack], [mcp], [all]
```

## Core

```python
from chopper_rag import Chopper

chopper = Chopper.from_pretrained(variant="base")          # device="auto", threshold from the model config
result = chopper.run("When was the Eiffel Tower completed?", passages)
for c in result.chops:                                      # Chop(passage_index, start, end, text, score)
    assert passages[c.passage_index][c.start:c.end] == c.text
print(result.decision)                                      # "ANSWER" or "SEARCH"
print(result.ranking(top_k=5))                              # [(passage_index, relevance), ...] most relevant first
chops = chopper.chop(question, passages)                    # just the list of Chop
ranked = chopper.rerank(question, passages, top_k=5)        # the chopper as a reranker
```

In every framework below, `mode="rerank"` turns the component into a reranker: it returns the input documents / nodes,
most relevant first (`top_n`, or `top_k` in Haystack), with the chopper's relevance as their score
(`metadata["relevance_score"]` in LangChain). The default `mode="chop"` returns the chops.

## LangChain

Listed in the [LangChain integrations](https://docs.langchain.com/oss/python/integrations/providers/all_providers) as ChopperCompressor.

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

## MCP server (Claude Desktop, Claude Code, any MCP client)

`pip install chopper-rag` installs `chopper-mcp`, a stdio MCP server with three tools (no install needed with uv:
`uvx chopper-rag` runs the same server). It is listed in the official MCP Registry as `io.github.wiiiktor/chopper`.

| tool | what it does |
|---|---|
| `chop_passages(question, passages)` | cut the exact quotes that answer `question` out of a list of texts (search results, tool outputs) |
| `chop_files(question, paths)` | read local text files, split them into ~1,000-character passages, cut out the quotes; each quote has its file path and character offset |
| `rank_passages(question, passages, top_k)` | the chopper as a reranker: the top_k passages, most relevant first, with their relevance |

The chop tools return `decision` (ANSWER / SEARCH), the `chops`, a ready-to-read `context` string and the passage
`ranking`, so the model reads a few quotes instead of whole documents and needs no separate reranker. Claude Code:

```bash
claude mcp add chopper -- chopper-mcp        # or, without installing:  claude mcp add chopper -- uvx chopper-rag
```

Claude Desktop (`claude_desktop_config.json`):

```json
{"mcpServers": {"chopper": {"command": "chopper-mcp", "env": {"CHOPPER_VARIANT": "base", "CHOPPER_DEVICE": "auto"}}}}
```

Settings: `CHOPPER_VARIANT` (base, small, base-pl, small-pl, plmodernbert-base), `CHOPPER_DEVICE` (auto, cpu, cuda), `CHOPPER_SNAP` (word, sentence, none), `CHOPPER_REPO`,
`CHOPPER_ALLOWED_DIRS` (directories `chop_files` may read, default: your home directory). A GPU is recommended; on a CPU
it works but is slow (about 10 s for 16 passages of 1,000 characters with `small`, 20 s with `base`). The model is
downloaded on the first call.

## Whole sentences by default (`snap`)

The model decides per token, so a raw chop can start or end inside a word or a clause. Since 0.1.7 every chop is
extended to the **whole sentences** it touches, so chops read as complete sentences (97% end at a sentence end, vs 36%
raw) at the cost of a little more text (4.1% of the input kept instead of 3.4% on our held-out set). `snap="word"` only
avoids cutting inside a word; `snap="none"` gives the raw token runs. **All F1 / accuracy numbers we publish were
measured on raw chops (`snap="none"`)**; on that held-out set word snapping leaves F1 unchanged (0.458) and sentence
snapping gives 0.433, because whole sentences also include words outside the gold answer span.

```python
ch = Chopper.from_pretrained(snap="word")          # or "none"; or later: ch.snap = "none"
```

In LangChain / LlamaIndex / Haystack pass such a `Chopper` as `chopper=`; in the MCP server set `CHOPPER_SNAP`.

## Output metadata (all three frameworks)

One output document / node per chop, in passage order. Metadata = the source's metadata plus:

| key | meaning |
|---|---|
| `chop_start`, `chop_end` | character offsets of the chop in the source text |
| `source_index` | position of the source in the input list |
| `chop_score` | mean p(KEEP) over the chop's tokens |
| `source_relevance` | (LangChain) the chopper's relevance score of the source passage |
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

License: **MIT, commercial use allowed** (code and the mmBERT weights). Naver's Provence / XProvence pruners, the closest alternatives, are CC BY-NC-ND 4.0 (non-commercial).
