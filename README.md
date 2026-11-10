<p align="center">
  <a href="https://youtu.be/KMvmdXCEOJU"><img src="media/youtube_cover_v15.jpg" width="100%" alt="LLMs need a Chopper: watch the video"></a><br>
  <a href="#quick-start"><img src="media/choice_universal_v6.png" width="49%" alt="Option 1, universal chopper: a small separate model (mmBERT-base 0.31B or mmBERT-small 0.14B), text in and text out, works with any answering model. Links to the quick start."></a>
  <a href="docs/DETAILS.md#a-host-with-latent-attention-minicpm3-4b"><img src="media/choice_integrated_v6.png" width="49%" alt="Option 2, integrated chopper: a 20% slice of MiniCPM3-4B itself, no second model, reads the model's own cache of the whole session. Links to how it works."></a>
</p>
<p align="center"><em>70-second video: <a href="https://youtu.be/KMvmdXCEOJU">youtu.be/KMvmdXCEOJU</a></em></p>

# Chopper 🔪 keep only the part of RAG that answers the question

Standard RAG (retrieval-augmented generation) pastes whole passages into your model's context, and they stay there.
**About 95% of those tokens don't help.** After a few lookups, a long session is mostly retrieval spam.

**Chopper** is a small model that reads the retrieved passages first and keeps only the fragments that matter. Your
model then sees ~100 tokens instead of ~3,000 per lookup, and a session can carry dozens of lookups.

```
question ─► retrieve 16 passages ─► 🔪 chop to ~3 fragments ─► model answers, or asks one follow-up search
```

- 🎯 **Clean context:** only useful fragments enter the session; the raw passages never stay.
- ⚡ **Less compute:** the main model reads 3–5% of the retrieved text.
- 🔗 **Multi-hop works:** the model writes a follow-up question when a fact is missing, and searches again.
- 🏠 **Built on one home GPU:** every dataset, label and model here was trained on a single RTX 3090.

## Results at a glance

| | |
|---|---|
| retrieved text the main model reads | **3–5%** (about 20–30× less); the answer is still in the kept text for 80% of HotpotQA and TriviaQA questions |
| multi-hop questions with every needed passage found (MuSiQue) | **.62 → .87** with one written follow-up search |
| correct answers end to end, MuSiQue / HotpotQA / TriviaQA (judged by gpt-5.6-luna, 4B answering model) | first search only .39 / .78 / .73 → **with the search loop .61 / .76 / .74** |
| "nothing here": pools whose answer was deleted (8 languages, business documents) | asks to search again **69%** of the time, keeps 1.5% of the text |
| chopping time per lookup (RTX 3090) | **155 ms** (separate mmBERT) · **277 ms** (built into MiniCPM3-4B) |
| 4-hop questions where every needed fact reaches the notes (MuSiQue, all paragraphs present) | **2% → 40%** with the new multi-hop chopper (0.31B); Claude Opus 5.5 as note-taker: 56% |

### Where multi-hop answers get lost

Read it like website traffic: 100 four-hop questions flow left to right, and at every hop the ones whose fact the notes missed
drop out. Top: the new multi-hop chopper (0.31B). Bottom: Claude Opus 5.5 taking the notes (~10,000B, rough estimate). Last stage:
Opus 5.5 answering from those notes. All hop counts, side by side, are in [the details](docs/DETAILS.md#7-multi-hop-where-answers-get-lost).

<p align="center"><img src="figures/hop_sankey_4hop.svg" width="100%" alt="Sankey funnel for 100 four-hop MuSiQue questions. Multi-hop chopper: 80% keep hop 1, 64% hop 2, 52% hop 3, 45% hop 4, 40% keep the answer, 28% answered correctly by Opus 5.5 from the notes. Opus 5.5 as note-taker: 92%, 77%, 64%, 58%, 56% and 44%."></p>

First long-conversation memory results (LoCoMo, LongMemEval) are in
[the details](docs/DETAILS.md#6-long-conversation-memory-locomo-and-longmemeval): there the 4B answering model is the
limit, and conversation-memory training is next ⏳.

## Pick your chopper

| | 🟧 **universal** | 🟪 **integrated** |
|---|---|---|
| what it is | a separate small encoder (mmBERT, 0.31B or 0.14B) | a 20% slice of MiniCPM3-4B's own weights + a small LoRA |
| works with | **any** answering model: text in, text out | MiniCPM3-4B |
| extra memory | 0.31B / 0.14B weights | 45M parameters; reads the model's own cache |
| weights | 🤗 [`wiiiktor/chopper_mmbert`](https://huggingface.co/wiiiktor/chopper_mmbert) | 🤗 [`wiiiktor/chopper_minicpm3`](https://huggingface.co/wiiiktor/chopper_minicpm3) |

Start with the universal one. The integrated one is the long-term bet: trained together with a model from the start, it
should become almost free. **If you pretrain models and want to build that, contact
[wiiiktor](https://github.com/wiiiktor).**

## Quick start

```bash
pip install -r requirements.txt

# your documents (a folder of .txt / .md, or a .jsonl with "title" and "text") + any Hugging Face causal LM
python operator-for-llm-with-mmbert.py --docs my_docs/ --model <hf model id> \
       --ask "Who founded the company that distributed the film UHF?"

# leave out --ask for an interactive session; it shows how many tokens plain RAG would have added
```

Weights download from Hugging Face on first use. Training data: 🤗 [`wiiiktor/chopper_demo`](https://huggingface.co/datasets/wiiiktor/chopper_demo).

## Want the details?

Everything else is on one page, **[docs/DETAILS.md](docs/DETAILS.md)**:
- [how it works](docs/DETAILS.md#how-it-works): double prefill, rollback, the answer-or-search decision;
- [datasets](docs/DETAILS.md#datasets) and [training](docs/DETAILS.md#training-hardware-time-memory) on one GPU;
- [all results](docs/DETAILS.md#results), including the [search loop](docs/DETAILS.md#3-second-retrieval-with-a-written-follow-up-question)
  and the [8-language rerank pilot](docs/DETAILS.md#5-rerank-pilot-8-languages-business-documents-no-answer-here);
- [rewiring a model into its own chopper](docs/DETAILS.md#rewiring) and [what comes next](docs/DETAILS.md#further-development);
- [credits, licenses and related work](docs/DETAILS.md#credits-and-licenses).

## Credits

Built on mmBERT (JHU CLSP), MiniCPM3-4B (OpenBMB), Qwen3 and Qwen3-Embedding (Qwen team), bge-reranker (BAAI), and the
MuSiQue, HotpotQA, TriviaQA, Wikipedia, LoCoMo and LongMemEval datasets. Our memory evaluation follows
[Memory Tax](https://arxiv.org/abs/2610.34073) (He et al., 2026) and compares with
[MemT-1.0](https://arxiv.org/abs/2610.26825) (MemTensor, 2026). Full table with licenses:
[docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).
