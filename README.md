<p align="center">
  <a href="https://youtu.be/KMvmdXCEOJU"><img src="media/youtube_cover_v15.jpg" width="100%" alt="LLMs need a Chopper: watch the video"></a><br>
  <a href="#quick-start"><img src="media/choice_universal_v6.png" width="49%" alt="Option 1, universal chopper: a small separate model (mmBERT-base 0.31B or mmBERT-small 0.14B), text in and text out, works with any answering model. Links to the quick start."></a>
  <a href="docs/DETAILS.md#a-host-with-latent-attention-minicpm3-4b"><img src="media/choice_integrated_v6.png" width="49%" alt="Option 2, integrated chopper: a 20% slice of MiniCPM3-4B itself, no second model, reads the model's own cache of the whole session. Links to how it works."></a>
</p>
<p align="center"><em>70-second video: <a href="https://youtu.be/KMvmdXCEOJU">youtu.be/KMvmdXCEOJU</a></em></p>

## Chopper 0.3B close to Opus 5.5 in cutting out the best pieces

### mmBERT is 32,000 times smaller

<p align="center"><img src="figures/size_cube.svg" width="100%" alt="One corner of a cube of dots. Each dot is 0.31B, the size of the mmBERT chopper; the multi-hop chopper is the single red dot on the corner; a 32 by 32 by 32 cube of such dots, about 32,000, stands for Claude Opus 5.5 at a rough 10,000B estimate (its size is not published)."></p>

### Qwen3-14B answering from different notes

400 MuSiQue questions, 100 each at 1, 2, 3 and 4 hops, each with its own 20 retrieved paragraphs. Qwen3-14B (4-bit, no fine-tuning,
no thinking mode) answers from the notes only; gpt-5.6-luna judges. Notes length = prompt tokens.

<p align="center"><img src="figures/qwen_notes_table.svg" width="100%" alt="Qwen3-14B correct answers (1/2/3/4 hops) and notes length. All notes retrieved from database: 2,620 tokens, 84/28/22/12. Multi-hop chopper notes: 244 tokens, 82/50/49/31. Opus 5.5's notes: 143 tokens, 90/64/55/43. Gold notes: 217 tokens, 86/63/52/37."></p>

### Qwen3-14B: all notes vs chopper notes vs Opus 5.5's notes

Read each diagram like website traffic: questions flow left to right, and at every hop the ones whose fact the notes missed drop
out; the last stage is Qwen3-14B answering correctly. Top: all 20 retrieved paragraphs. Middle: the multi-hop chopper's notes.
Bottom: Opus 5.5's notes.

<p align="center"><img src="figures/qwen_sankey_1hop.svg" width="100%" alt="1-hop questions: Qwen3-14B correct 84% with all notes, 82% with chopper notes, 90% with Opus notes"></p>
<p align="center"><img src="figures/qwen_sankey_2hop.svg" width="100%" alt="2-hop questions: Qwen3-14B correct 28% with all notes, 50% with chopper notes, 64% with Opus notes"></p>
<p align="center"><img src="figures/qwen_sankey_3hop.svg" width="100%" alt="3-hop questions: Qwen3-14B correct 22% with all notes, 49% with chopper notes, 55% with Opus notes"></p>
<p align="center"><img src="figures/qwen_sankey_4hop.svg" width="100%" alt="4-hop questions: Qwen3-14B correct 12% with all notes, 31% with chopper notes, 43% with Opus notes"></p>

In the middle and bottom diagrams the last stage counts only questions whose notes kept the whole chain; Qwen also gets some right
from partial notes, so its overall scores (table above) are higher. More: [docs/DETAILS.md §7](docs/DETAILS.md#7-multi-hop-where-answers-get-lost).

---

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
