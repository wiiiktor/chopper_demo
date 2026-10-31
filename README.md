[![LLMs need a Chopper: watch the video](media/video-screen-with-title.png)](https://youtu.be/KMvmdXCEOJU)

*70-second video: [youtu.be/KMvmdXCEOJU](https://youtu.be/KMvmdXCEOJU) · also in this repo as [media/chopper.mp4](media/chopper.mp4)*

# Chopper: cut retrieved text down to the fragments that answer the question

> **DRAFT 2026-10-31.** Sections marked ⏳ describe parts that are planned or still training. Every number in
> **Results** was measured; nothing in it is projected.

> 💡 **Affordable technology.** Everything in this repository was built and trained on **one 24 GB GPU** (an RTX 3090
> at home): the datasets, the 14B fragment labels, every chopper and the answerer. No cloud, no cluster.

## Clean your RAG inputs from spam tokens, and save your long work session

**The problem.** Standard RAG (retrieval-augmented generation) pastes whole retrieved passages into the model's
context, and they stay there. Typically ~95% of those tokens are irrelevant to the question. After a few retrieval
rounds, a long work session is mostly retrieval spam:
- it uses up the context window;
- it slows every following token;
- it buries the conversation the user actually cares about.

So a long session that needs many RAG lookups is not practical today.

**The fix.**
1. A small **chopper** model reads the retrieved passages first and cuts out only the fragments that matter
   (~3 × 35 tokens out of ~3,000).
2. The main model's context is then **rolled back** to the end of the conversation (its KV cache is cropped). The raw
   passages never stay in it.
3. Only the clean fragments are added, and the model either **answers** or writes **one follow-up search question**.

The context grows by ~100 tokens per lookup instead of ~3,000, so a session can carry dozens of lookups.

Why it matters:
- **A clean, long-lived context.** After each lookup only the useful fragments stay in the session. Plain RAG cannot
  take passages back out once they are in the context; rollback can.
- **Better answers.** Fragments beat full passages by 2–4× in exact-match accuracy for the same answerer.
- **Less compute.** The expensive model prefills ~100 tokens instead of ~3,000 per retrieval step.
- **Multi-hop retrieval works.** A written follow-up question retrieves the second hop that the first search missed.
  On MuSiQue, all-hops coverage goes from .62 to .87.

---

## Contents
1. [How it works](#how-it-works)
2. [What is in this repository](#what-is-in-this-repository)
3. [Quick start](#quick-start)
4. [Datasets](#datasets)
5. [Models](#models)
6. [Training: hardware, time, memory](#training-hardware-time-memory)
7. [Results](#results)
8. [The operator: double prefill, chopping, rollback, answer or search](#the-operator)
9. [Rewiring a Qwen-like model into its own chopper](#rewiring)
10. [Further development](#further-development)
11. [Credits and licenses](#credits-and-licenses)
12. [For AI assistants: machine-readable summary](#for-ai-assistants)

---

## How it works

```
conversation tail + question
   │
   ├─► 1. RETRIEVE   Qwen3-Embedding-0.6B top-100 → bge-reranker-v2-m3 → top 16 passages      (text only)
   │
   ├─► 2. CHOP       the chopper reads question + 16 passages (twice, see "double prefill")
   │                 → a KEEP probability per token → fragments (~3 × 35 tokens)
   │
   ├─► 3. ROLL BACK  the answering model's cache is cut back to the end of the conversation; it prefills ONLY the fragments
   │
   └─► 4. DECIDE     the answering model writes   <answer>   or   [search] <follow-up question>
                       └─ on [search]: the follow-up question becomes the next query → steps 1–4 again
```

Three versions of the chopper:

| | **mmBERT chopper** ✅ | **rewired chopper** ✅ | **chopper trained with the model** 🤝 |
|---|---|---|---|
| what it is | a separate encoder, fine-tuned: `jhu-clsp/mmBERT-base` (0.31B) or `jhu-clsp/mmBERT-small` (0.14B); 8k context, bidirectional | a ~10% **slice of an existing model's own weights**, cut out after training (rewired), + a small LoRA + KEEP head | a ~10% slice **trained together with the full model** from the start, with chopping as one of its training tasks |
| extra memory | 0.31B / 0.14B weights | ~15M parameters (no copy of the base weights) | ~15M: the slice is part of the model |
| reads the conversation | re-encodes what fits in its window | reads the model's **own cached keys/values**, so nothing is recomputed | the same: the model's own cache |
| works with | any answering model (text in, text out) | the model it was carved from | the model it was trained with |
| quality today | the better of the two measured | below mmBERT (see [Results 2](#2-the-chopper)) | not built yet: contact the repo owner ([wiiiktor](https://github.com/wiiiktor)) for cooperation |

> [!IMPORTANT]
> ## 🤝 Why the third one: a chopper trained with the model
>
> A 10% slice cut out of a finished model has to work with features that were never trained for cutting, and a LoRA
> cannot fully fix that: **our rewired slice loses to mmBERT-small**, an encoder of the same size trained end to end.
>
> A slice trained **together with the full model** would have its own features shaped for the job and would still
> read the full model's cache. It should come **almost for free** (about 10% of the model's compute, no second
> model) and at a **much higher quality** than either version here.
>
> It needs a pretraining run, which is beyond one GPU.
> **If you train models and want to build this, contact the repository owner
> ([wiiiktor](https://github.com/wiiiktor)) to cooperate on the project.**

---

## What is in this repository

Code lives here; weights and data live on Hugging Face (public): one model repo [`wiiiktor/chopper_demo`](https://huggingface.co/wiiiktor/chopper_demo) with a folder per model, and the dataset repo of the same name.

| where | what |
|---|---|
| `scripts/data/` | builds the datasets: pools (`mm_dump_pools.py`, `wk_*.py`) → retrieval (`mm_contexts.py`) → 14B fragment labels (`mm_qwen_labels.py`) → follow-up questions (`mm_subq.py`) |
| `scripts/chopper/` | `mm_train.py`: trains and evaluates the mmBERT chopper |
| `scripts/rewire/` | `keeper_rewire.py` rewires any Qwen/Llama-layout model, `keeper_train.py` trains its width-slice chopper, `keeper_lib.py` |
| `scripts/answer/` | the answer-or-search LoRA (`mm_qwen_answer.py`), forced follow-up questions, and the search loop |
| `scripts/plots/` | training curves (`plot_curves.py`) and the HTML chart page (`curves_page.py`) |
| `chains/` | the exact shell chains the results came from (they contain the original absolute paths) |
| `figures/` | the charts |
| 🤗 [`chopper_demo/mmbert-base/`](https://huggingface.co/wiiiktor/chopper_demo/tree/main/mmbert-base) | mmBERT-base chopper, 54,000 rows (all of dataset A, then continued on dataset B; see the ⚠️ in [Results 2](#2-the-chopper)): `encoder/` (a normal Transformers model) + `heads.safetensors` (KEEP + ANSWER/SEARCH) |
| 🤗 [`chopper_demo/mmbert-small/`](https://huggingface.co/wiiiktor/chopper_demo/tree/main/mmbert-small) | the size-matched mmBERT-small chopper, same layout and data |
| 🤗 [`chopper_demo/qwen3-1.7b-w10/`](https://huggingface.co/wiiiktor/chopper_demo/tree/main/qwen3-1.7b-w10) | `rewired/` Qwen3-1.7B (exact permutation, same outputs), `permutation.safetensors`, `chopper_lora.safetensors`, `keep_head.safetensors` |
| 🤗 [`chopper_demo/answerer-qwen3-1.7b-lora/`](https://huggingface.co/wiiiktor/chopper_demo/tree/main/answerer-qwen3-1.7b-lora) | the answer-or-`[search]` PEFT LoRA for the **plain** Qwen3-1.7B |
| 🤗 [`wiiiktor/chopper_demo` (dataset)](https://huggingface.co/datasets/wiiiktor/chopper_demo) | dataset A (benchmark chains) and dataset B (Wikipedia chains) |
| `operator-for-llm-with-mmbert.py` | the full loop over **your documents** with the mmBERT chopper and **any** Hugging Face causal LM (default Qwen3-1.7B + our answerer LoRA): retrieve → chop → roll back → answer or search |
| `operator-for-llm-slice.py` | the same loop with the 10% slice chopper inside the rewired Qwen3-1.7B (no second model; the slice reads the session from the model's own cache) |
| `scripts/operator_common.py` | what both operators share: corpus chunking, retrieval, the session cache with rollback, the answer loop |
| `requirements.txt` | Python packages for the operators |

`scripts/data/mm_dump_pools.py` builds the dataset A pools with the benchmark loaders of this project's larger
trainer, which is not part of this repository; the pools it wrote are in the dataset repo.

---

## Quick start

Run a work session over your own documents (a folder of `.txt` / `.md` files, or a `.jsonl` with `"title"` and
`"text"`). The weights download from Hugging Face on first use.

```bash
pip install -r requirements.txt

# mmBERT chopper + Qwen3-1.7B with the answerer LoRA (any other model: --model <hf id>)
python operator-for-llm-with-mmbert.py --docs my_docs/ --ask "Who founded the company that distributed the film UHF?"

# the 10% slice chopper inside the rewired Qwen3-1.7B
python operator-for-llm-slice.py --docs my_docs/

# without --ask: an interactive session; after each answer it prints how many tokens the session holds
# and how many plain RAG would have added
```

The research scripts behind every number in this README:

```bash
# 1. chop: evaluate the mmBERT chopper on the validation contexts
python scripts/chopper/mm_train.py <data dir> <out dir> --eval-only chopper.pt \
       --pred-thr 0.8 --merge-gap 5 --min-run 4

# 2. rewire a model and check that its outputs are unchanged
python scripts/rewire/keeper_rewire.py Qwen/Qwen3-1.7B <contexts.jsonl> perms.pt --n 128

# 3. train its 10% chopper
python scripts/rewire/keeper_train.py <data dir> perms.pt <out dir> --frac 0.1
```

---

## Datasets

### A. Benchmark chains (MuSiQue, HotpotQA, TriviaQA)

| | MuSiQue | HotpotQA | TriviaQA | total |
|---|---|---|---|---|
| training questions (all labelled) | 9,600 | 8,400 | 9,000 | **27,000** |
| validation (eval only) | 100 | 100 | 100 | 300 |

> ⚠️ These are **benchmark** datasets. Only their train splits are used for training and the validation rows are
> never trained on. Even so, a model trained on them should not be reported as zero-shot on those benchmarks.
> Dataset B exists for this reason.

Each training row contains:

| field | content | built by |
|---|---|---|
| `question`, `answer` | from the source dataset | — |
| `passages` | top 16 out of a pool of ~100+ distractor passages, each with its gold hop index or −1 | `mm_dump_pools.py`, `mm_contexts.py` |
| `allin` | 1 if every hop's gold passage is among the 16 (target **answer**), else 0 (target **search**) | `mm_contexts.py` |
| `spans` | fragment labels as character spans `[passage, start, end]` | `mm_qwen_labels.py` (Qwen3-14B-AWQ) |
| `subq` | the follow-up question for rows with a missing hop | `mm_subq.py` (Qwen3-14B-AWQ) |

**How the fragment labels are made.** Qwen3-14B-AWQ sees the question, **the gold answer** and the 16 passages. It
writes the reasoning chain as short steps and quotes, word for word, the shortest fragment behind each step. Those
quotes are located back in the text as character spans. 1.4% of the quotes cannot be located and are dropped.

> ⚠️ Because the labeller sees the answer, its fragments are an **upper bound**. Never use them as the chopper's
> output at evaluation time. The follow-up questions are written **without** the answer.

### B. Wikipedia chains (benchmark-free)

✅ Finished 2026-10-31, in the dataset repo under `wikipedia_chains/`: 18,000 2-hop + 9,000 1-hop training rows
with 14B fragment labels and 1,164 follow-up questions, and 300 validation rows (200 2-hop + 100 1-hop).

27,000 training rows (18,000 2-hop + 9,000 1-hop, the same 2 : 1 mix as dataset A) and 300 validation rows, built
from English Wikipedia (`wikimedia/wikipedia`, 2023-11-01 dump) with no benchmark question anywhere in the loop.

1. **Bridges** (`wk_chains.py`, CPU, ~20 min). The dump has no hyperlinks, so a bridge is an exact mention of another
   article's title in one of the first three paragraphs of article A. Mentions that are part of a longer name
   ("James Earl" inside "James Earl Jones"), dates, and single common words ("College", "August") are skipped.
   Each article B can be a bridge at most 15 times, so "United States" cannot dominate.
2. **Questions** (`wk_gen.py`, Qwen3-14B-AWQ). 2-hop: the answer is a short fact copied from B, and B is described only
   through A ("the label that released …"). 1-hop: a quiz question about one paragraph that makes sense on its own.
3. **Filter** (`wk_judge.py`). Rules reject a question if:
   - the answer leaks into A or into the question;
   - the question names B;
   - the answer is not in its passage;
   - the question says "the passage".

   Then Qwen3-4B must find the answer from the gold passages. Questions it can answer with no passage are counted,
   not removed.
4. **Pools** (`wk_pools.py`). 1,000 chunks per question:
   - the gold paragraphs;
   - hard negatives: the other paragraphs of A and B, and of up to 8 other articles A mentions;
   - random article leads as filler.

   The pools are in the dataset A format, so retrieval, the 14B fragment labels and the follow-up questions run
   unchanged (`mm_contexts.py`, `mm_qwen_labels.py`, `mm_subq.py`).

The whole chain is `wiki_chain.sh`. It generates in rounds of 20,000 2-hop and 6,000 1-hop candidates until the
quotas are filled, so the acceptance rate is measured on the way.

Measured acceptance (Qwen3-14B questions, Qwen3-4B filter), rounds 1 and 2: **2-hop 40.5% / 40.6%, 1-hop 79.0% /
80.2%**. Of the accepted questions, Qwen3-4B answers only **7–9% with no passage**, so retrieval is needed for
nearly all of them. The same filter accepted 7% of Qwen3-1.7B's 2-hop questions in a smoke test. Three rounds fill
the quotas (~2 h per round).

Time on one RTX 3090: questions ~6 h (3 rounds); retrieval ~4 h; 14B labels ~10 h (≈ 2,600 questions/h); follow-up
questions ~1 h. That is **~21 h for 27,000 rows**.

Planned later: 3-hop chains (A → B → C).

---

## Models

| model | base | trainable | trained on | status |
|---|---|---|---|---|
| **mmBERT-base chopper** | `jhu-clsp/mmBERT-base` (0.31B) | all weights + KEEP head + decision head | 9,000 rows of dataset A | ✅ |
| **answerer** (answer or `[search] <question>`) | `Qwen/Qwen3-1.7B` | LoRA r32 on all projections (35M) | 9,000 rows of dataset A, then continued on the other 18,000 | ✅ (continuation: 2 h 40 min, 4.0 GiB) |
| **rewired chopper, 10% width** | `Qwen/Qwen3-1.7B`: 1 of 8 KV groups + 512 of 6,144 MLP neurons per layer | LoRA r32 + KEEP head (14.9M) | 9,000 rows of dataset A, then continued on the other 18,000 | ✅ (continuation 3 h 35 min at ~0.85 s/step, 4.9 GiB) |
| mmBERT-base chopper, full data | `jhu-clsp/mmBERT-base` | all weights + heads | all 27,000 rows | ✅ (3090, 2 h 20 min incl. 3 evals, 3.4 GiB, shared GPU) |
| rewired choppers, other sizes | any Qwen / Llama-layout model (e.g. Qwen3-4B / 8B / 14B) | LoRA + KEEP head | datasets from this repo and/or your data, on a small GPU card | 🛠️ anyone can make one: `scripts/rewire/keeper_rewire.py` rewires the model, `scripts/rewire/keeper_train.py` trains its chopper |
| **mmBERT-small chopper** (size-matched control) | `jhu-clsp/mmBERT-small` (0.14B, about the size of the 1.7B's 10% slice) | all weights + heads | the same 9,000 rows, then continued on the other 18,000 | ✅ (9k: 24 min, 27k continuation: 1 h 35 min, 1.8 GiB) |
| **mmBERT-base and -small, continued on Wikipedia** (the versions on Hugging Face) | the 27k choppers above | all weights + heads | + the 27,000 rows of dataset B (54k in total) | ✅ (base 2 h 30 min, small 1 h 45 min, run side by side with the slice on one 3090) |

**Name:** "chopper" is the component that cuts fragments. Older checkpoints and output folders from before the rename are still called `reader` (`reader.pt`, `reader2/`), and the scripts load them under either name.

---

## Training: hardware, time, memory

Everything was built and trained on **one NVIDIA RTX 3090 (24 GB)**. No cloud.

| step | model | time | peak VRAM |
|---|---|---|---|
| retrieval contexts | Qwen3-Embedding-0.6B + bge-reranker-v2-m3 | 0.46 s / question (18k ≈ 2.3 h) | ~6 GB |
| fragment labels | Qwen3-14B-AWQ in vLLM | ~2,640 questions / h (18k ≈ 7 h) | ~21 GB |
| follow-up questions | Qwen3-14B-AWQ in vLLM | 9k in < 1 h | ~21 GB |
| mmBERT chopper | mmBERT-base, full fine-tune, 1 epoch, batch 4 | 0.24 s / step → 9k ≈ 36 min | 3.0 GiB |
| answerer | Qwen3-1.7B + LoRA, 1 epoch, accum 8 | 0.41 s / step → 9k ≈ 62 min | 3.9 GiB |
| rewiring | Qwen3-1.7B, activation scores on 128 contexts + permutation + checks | 25 s scoring, < 1 min in total | ~5 GB |
| rewired chopper | 10% slice + LoRA, 1 epoch, passages ~2.9k tokens read twice | 0.8 s / step → 9k ≈ 2 h | 4.9 GiB |

Software: PyTorch 2.12, transformers 5.10, peft, vLLM.

### Training curves

<img alt="Training curves: chopper KEEP loss, train F1 and balanced accuracy, val answer-in-kept and kept share for mmBERT-base, the 10% Qwen3-1.7B slice and mmBERT-small; answerer CE, token accuracy and val exact, against training rows seen (0–54,000)" src="figures/training_curves.png">

- The x axis is training rows seen. A run that continues on the other 18,000 rows is drawn after its 9,000-row
  parent; after 27,000 the runs continue on the 27,000 Wikipedia rows (dataset B). The dotted lines mark 9,000 and
  27,000. The train F1 jump at 27,000 is the easier Wikipedia labels, not a better chopper (see the ⚠️ in Results 2).
- Dashed green: mmBERT-small trained once on all 54,000 rows mixed (benchmark + Wikipedia), from the base model.
- Train lines are 500-row rolling means.
- Validation points are the mean over MuSiQue, HotpotQA and TriviaQA, at p > 0.5 with no span shaping, which is the
  setting every chopper logs during training.
- Regenerate with `scripts/plot_curves.py`; the plotted numbers are also in `figures/curves.json`.

### Chopping speed: 4,000 passage tokens, read twice (8,000 tokens)

> ⚠️ **Estimates from FLOP counts** (floating-point operations, matmuls + attention). They are calibrated on one
> measurement, the mmBERT-base 8k-token forward of 155 ms on the RTX 3090. Expect ±2×. On big GPUs the small slices
> will not reach full efficiency, so read anything under ~10 ms as "≈10 ms". Measured times will replace these.

**The 10% chopper never runs the LM head.** It outputs one KEEP probability per token through a hidden→1 linear layer,
not a next-token distribution over the vocabulary. For Qwen3-1.7B, the vocabulary projection is 2,048 × 151,936 =
311M weights. That is more than the whole 10% slice (0.14B) and would cost more compute than the slice itself.

| host model (10% slice) | work, 10¹² FLOP | T4 (2018) | RTX 3090 (2020) | RTX 4090 (2022) | A100 (2020) | H100 (2022) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (0.04B) | 1.6 | 200 ms | 80 ms | 36 ms | ~10 ms | ~10 ms |
| **Qwen3-1.7B (0.14B)** | 3.1 | 390 ms | **160 ms** | 70 ms | 17 ms | ~10 ms |
| Qwen3-4B (0.36B) | 8.2 | 1.0 s | 410 ms | 180 ms | 46 ms | 18 ms |
| Qwen3-8B (0.70B) | 13.4 | 1.7 s | 670 ms | 300 ms | 74 ms | 30 ms |
| Qwen3-14B (1.3B) ¹ | 24.3 | 3.0 s | 1.2 s | 540 ms | 135 ms | 54 ms |
| Qwen3-32B (3.1B) ¹ | 58.4 | 7.3 s | 2.9 s | 1.3 s | 320 ms | 130 ms |
| *for comparison: full Qwen3-1.7B, same 8k* | 30 | 3.8 s | 1.5 s | 670 ms | 170 ms | 70 ms |
| *for comparison: mmBERT-base chopper, 8k once (measured on the 3090)* | — | — | *155 ms* | — | — | — |

**First measurement:** the Qwen3-1.7B 10% chopper reads a typical context (cached conversation + ~2.9k passage
tokens × 2) in **233 ms median on the RTX 3090**, measured during an eval while training shared the GPU. The FLOP
estimate for that length is ~120 ms, so real runs are about 1.5–2× slower than the estimates, as expected for narrow
matrices.

Assumed effective throughput (bf16, fp32 accumulation): T4 8, RTX 3090 20, RTX 4090 45, A100 180, H100 450 TFLOPS
(10¹² FLOP per second). ¹ In bf16, 14B and 32B do not fit a 16–24 GB card; quantised weights add dequantisation time.

The doubled 10% read costs about 1/10 of the full model reading the same text. It is also 3–4× cheaper than the full
model reading the passages once, which is what plain RAG pays, and plain RAG then keeps those tokens in the context.

---

## Results

All results are on the 300 validation questions (100 per dataset). Every model is trained for one epoch. Three numbers
written as `a / b / c` are MuSiQue / HotpotQA / TriviaQA.

### How to read the metrics
- **Exact match:** the answerer's output equals the gold answer after lower-casing and removing articles and
  punctuation.
- **Character precision / recall / F1:** the chopper's kept characters compared with the characters the Qwen3-14B
  teacher quoted (spaces ignored, so the metric does not depend on the tokenizer).
  - Precision = the share of kept text that the teacher also kept.
  - Recall = the share of the teacher's text that the chopper kept.
  - F1 = 2·P·R / (P + R), the harmonic mean, which is pulled toward the lower of the two. A chopper cannot score
    well by keeping everything or by keeping almost nothing.
- **Answer-in-kept:** the share of questions whose gold answer string appears inside the kept text. It is the ceiling
  for an answerer that reads only the fragments. It matters more than F1 when the answer appears in several passages:
  the chopper may keep a different but equally good sentence than the teacher did.
- **All-hops:** the share of questions where **every** passage needed for the answer (each hop) is in the set given
  to the answerer. A MuSiQue question needs 2–4 passages; missing one makes it unanswerable.
  - **first 16** = the first retrieval only.
  - **16 ∪ 16** = the first 16 plus 16 from the second query, duplicates counted once (≤ 32 passages).
  - **8 + 8** = the top 8 of each, 16 in total: the same budget as no second search. It checks that a gain is not
    just from reading more passages.
- **SEARCH / ANSWER recall:** among questions that need a second search (a hop is missing from the first 16), the
  share where the answerer writes `[search]`; among those that don't, the share where it answers.

### 1. Fragments beat full passages

The same trained answerer gets different inputs (exact match on questions whose hops were all retrieved):

| notes the answerer reads | MuSiQue | HotpotQA | TriviaQA |
|---|---|---|---|
| Qwen3-14B fragments (upper bound) | **.55** | **.64** | **.72** |
| **mmBERT chopper fragments** | **.40** | **.57** | **.63** |
| nothing (question only) | .00 | .12 | .11 |
| all 16 passages in full *(RWKV-1.5B answerer)* | .02 | .19 | .26 |

### 2. The chopper

Decoded at p > 0.8, gaps ≤ 5 tokens merged, runs < 4 tokens dropped:

| chopper | trained on | character F1 | answer-in-kept | fragments / question |
|---|---|---|---|---|
| teacher: Qwen3-14B quotes (saw the answer) | — | 1 (by definition) | .62 / .80 / .83 | — |
| **mmBERT-base** (0.31B) | 9k | **.49 / .53 / .30** | **.52 / .80 / .80** | 3.0 / 3.9 / 7.8 |
| 10% Qwen3-1.7B (0.14B slice), **⅓ of the way through training** | 3k of 9k | .31 / .36 / .17 | .23 / .72 / .72 | 2.6 / 2.7 / 7.0 |
| same, threshold lowered to p > 0.5 | 3k of 9k | .25 / .32 / .14 | .32 / .88 / .83 | ~11 / 11 / 31 |
| **10% Qwen3-1.7B, end of the 9k epoch** | 9k | .29 / .42 / .17 | .22 / .65 / .71 | 2.9 / 2.9 / 3.8 |
| same, threshold lowered to p > 0.5 | 9k | .20 / .24 / .13 | .36 / .87 / .82 | ~17 / 17 / 26 (keeps 16% of the text) |
| **mmBERT-small (0.14B), size-matched** | 9k | .40 / .45 / .23 ¹ | .51 / .81 / .76 | 3.4 / 4.1 / 10.5 |
| **mmBERT-base, all data** | 27k | **.52 / .60 / .37** ¹ | **.54 / .83 / .79** | 2.9 / 3.4 / 5.5 |
| mmBERT-small, continued | 27k | .49 / .55 / .32 ¹ | .48 / .80 / .75 | 2.6 / 3.4 / 6.1 |
| 10% Qwen3-1.7B, continued | 27k | .31 / .48 / .19 | .24 / .70 / .72 | 3.1 / 3.0 / 3.8 |
| **mmBERT-base, + Wikipedia** (on HF) | 54k | **.56 / .66 / .44** ¹ | .44 / .80 / .69 | 2.3 / 2.6 / 3.7 |
| mmBERT-small, + Wikipedia (on HF) | 54k | .53 / .62 / .38 ¹ | .28 / .74 / .58 | 1.9 / 2.5 / 3.0 |

¹ token-level F1 from the training script's own eval. Scored at character level, mmBERT-small gives .40 / .45 / .23
(9k) and .49 / .55 / .32 (27k), the same to ±.01.

> ⚠️ **Continuing on the Wikipedia chains (dataset B) lowers the benchmark scores that matter most.** F1 rises, but
> answer-in-kept falls: mmBERT-base .54 / .83 / .79 → .44 / .80 / .69, mmBERT-small .48 / .80 / .75 → .28 / .74 / .58.
> The choppers learn the tighter Wikipedia fragments (they keep 2–4% of the text instead of 4–6%) and drop the answer
> more often on the benchmarks. The 54k versions are the ones in the standard Hugging Face folders; the 27k versions
> are revision `d560e98814` of the model repo. A mmBERT-small trained once on all 54k rows mixed is being evaluated
> for comparison ⏳.

On the 300 Wikipedia validation questions (2-hop / 1-hop), the same decode:

| chopper | trained on | token F1 | answer-in-kept | fragments / question |
|---|---|---|---|---|
| mmBERT-base | 27k (dataset A) | .50 / .57 | .91 / .96 | 4.0 / 2.2 |
| **mmBERT-base** | 54k (+ dataset B) | **.65 / .76** | .91 / .92 | 2.8 / 1.4 |
| mmBERT-small | 27k (dataset A) | .43 / .49 | .86 / .97 | 4.2 / 2.4 |
| mmBERT-small | 54k (+ dataset B) | .62 / .74 | .89 / .96 | 2.8 / 1.4 |

The choppers trained on benchmarks alone already keep the answer on 86–97% of the Wikipedia questions, so dataset B
transfers in that direction. Note that the 27k rows above were run again for this table at p > 0.8 with shaping: they
match the earlier table to ±.01.

**Three times the data buys mmBERT-base +.03 / +.07 / +.07 F1** at the same answer-in-kept, i.e. tighter fragments
(it keeps 4–6% of the text instead of 5–7%). Its validation curve is flat from 9k to 27k rows at p > 0.5, so more
of the same benchmark rows is unlikely to help much further.

**After 9,000 rows the 10% Qwen3-1.7B slice is clearly behind mmBERT-base (0.31B).** When allowed to keep more text
(p > 0.5), it keeps the answer about as often on HotpotQA and TriviaQA, but it needs ~16% of the text to do so,
against ~10% for mmBERT-base, and it is much weaker on MuSiQue. Its training loss levels off near 0.40, against ~0.25
for mmBERT-base (see the curves). Two runs will show whether this is a data limit or a capacity limit: the
continuation on the other 18,000 rows, and mmBERT-small, which has the same 0.14B size.

**mmBERT-small answers the size question: it is not the size.** At the same 0.14B and the same 9,000 rows it keeps the
answer almost as often as mmBERT-base (.51 / .81 / .76 vs .52 / .80 / .80) and far more often than the 10% Qwen slice
on MuSiQue (.51 vs .22). A small encoder trained end to end beats a 10% slice of a decoder whose other 90% stays frozen.

**After all 27,000 rows the gap holds.** The 10% slice improves a little (F1 +.02 to +.06), but it keeps the answer
less often than mmBERT-small: .24 / .70 / .72 against .48 / .80 / .75. Read on the full model's cache, it remains
the cheaper option to run, at 133 ms per read with no second model in memory, but it is not yet the better one.
One advantage these benchmarks cannot show: the slice attends to the **whole conversation** through the full model's
KV cache, at no extra cost, while mmBERT only sees what fits in its 8,192-token window next to the passages. Here
the "conversation" is just the question, so a test with long real sessions is still to do. To make the slice the
better chopper, the base model itself most likely has to be trained with the chopper from the start (see Further
development).

**Decision head:** mmBERT's own answer/search head never predicts *search*. That decision has to come from the
answering model, which does learn it (next table).

### 3. Answer or search

The Qwen3-1.7B answerer's decisions on MuSiQue (the dataset where a hop is often missing):

| notes | trained on | SEARCH recall | ANSWER recall | exact (M / H / T) |
|---|---|---|---|---|
| Qwen3-14B fragments | 9k | .61 | .95 | .55 / .64 / .72 |
| Qwen3-14B fragments | 27k | **.74** | .94 | .50 / .65 / .74 |
| mmBERT chopper fragments | 9k | .37 | .95 | .40 / .57 / .63 |
| mmBERT chopper fragments | 27k | **.50** | .95 | .42 / .66 / .60 |

Three times the data mainly teaches the answerer **when to search** (+.13 recall on both kinds of notes). Exact
match moves by a few points in either direction. With 100 questions per dataset, a few points is within noise.

### 4. Second retrieval with a written follow-up question

MuSiQue, all gold hops present among the passages:

| second query | first 16 ∪ new 16 | first 8 + new 8 (same budget) |
|---|---|---|
| none (first 16 only) | .62 | — |
| question + chopper fragments | .77 | .65 |
| question + the gold hop-1 passage | .76 | .69 |
| **follow-up question written from the chopper's fragments** | **.87** | **.81** |
| follow-up question written from the gold hop-1 passage | .93 | .89 |
| follow-up question written by the 1.7B answerer, **only when it chooses to search**, from chopper notes | .73 | .72 |
| same, from Qwen3-14B notes | .77 | .77 |
| **follow-up question written by the 1.7B answerer for every question** (forced `[search]`), from chopper notes | **.86** | **.80** |
| same, from Qwen3-14B notes | .87 | .86 |

HotpotQA and TriviaQA are already at .96–1.00 after the first search.

The 14B rows write a follow-up question for every question. The 1.7B rows only search when the answerer decides to,
and it searches on 37% (chopper notes) or 61% (14B notes) of the questions that need it. The forced rows separate the
two: **forced to write one every time, the 1.7B's follow-up questions are as good as the 14B's** (.86 vs .87). The gap
in the "only when it chooses" rows is entirely the decision to search, so that decision is what to train next.

### 5. Rewired chopper

The rewiring is **exact**. Checked at fp32 on Qwen3-1.7B:
- **Full model unchanged:** after the permutation, the largest logit change is 1.2e-4 (logit scale 49) and the
  top-1 token agrees 100%.
- **Cache sharing correct:** the chopper at 100% width, reading the full model's cached conversation, reproduces
  the full model's hidden states to 1.4e-4 (scale 82).

Chopping quality against the mmBERT chopper is in [section 2](#2-the-chopper) (the "10% Qwen3-1.7B" rows): it is below
mmBERT-small at the same size, which is why a slice trained with the model is the next step.

---

## The operator

Two scripts implement it: `operator-for-llm-with-mmbert.py` and `operator-for-llm-slice.py` (shared code in
`scripts/operator_common.py`).

| | with mmBERT | slice |
|---|---|---|
| chopper | mmBERT-base or -small, a separate 0.31B / 0.14B encoder | the 10% slice of the answering model + its LoRA and KEEP head |
| answering model | **any** Hugging Face decoder-only causal LM; default Qwen3-1.7B + our answerer LoRA, other models get a short instruction | the rewired Qwen3-1.7B (or another model you rewired and trained with `scripts/rewire/`) |
| answerer LoRA | used as trained | permuted into the rewired layout: same outputs as on the plain model (max logit difference 3e-5, fp32) |
| what the chopper reads | the question + 16 passages (8k-token window) | the **whole session** from the model's own cache + the 16 passages, twice |

**Assumptions about "any transformer"** (mmBERT operator): a Hugging Face decoder-only causal LM whose cache is a
`DynamicCache` with `crop()`, i.e. full attention (Llama, Qwen, Mistral and similar). Sliding-window-only or
recurrent caches cannot be rolled back this way. Our answerer LoRA exists for Qwen3-1.7B only, so other models
answer from an instruction and their `[search]` decisions are untested.

**Measured so far:** both operators ran end to end on a CPU over a small corpus (229 chunks from the validation
passages): sessions of 3 questions held 151–334 tokens in the cache where plain RAG would have added 2,200–6,350
passage tokens. The answers themselves have the answerer's known weakness: it rarely writes `[search]` when a hop is
missing (SEARCH recall .50, see Results 3).

```
state: answering model M with its KV cache of the conversation, at position P (the end of the conversation)

loop (max N searches):
  1. passages ← retrieve(query)                               # query = the question, later the follow-up question
  2. DOUBLE PREFILL: the chopper reads  [passages] [the same passages again]
       the second copy can attend to the whole first copy, which gives a causal model a two-way view;
       KEEP is read from the second copy
  3. fragments ← tokens with p(KEEP) > 0.8, gaps ≤ 5 merged, runs < 4 dropped
  4. ROLL BACK: crop M's cache to P (DynamicCache.crop); the chopper's own tensors are simply dropped
     (with the rewired chopper, M never prefilled the passages at all)
  5. M prefills  "Notes: - fragment … Question: …"  and generates
       → an answer                     : done
       → [search] <follow-up question> : query ← follow-up question; keep the notes; go to 1
```

With the **rewired chopper**, step 2 runs on ~10% of M's weights and reads M's cached conversation directly. The
only extra cost of a retrieval step is a ~10% model reading ~6k tokens.

**Why the rollback is the point.**
- With plain RAG, the session keeps everything it was ever shown: after 20 lookups that is ~60,000 tokens of passages,
  mostly irrelevant.
- With the operator, the cache is cropped back to P before the fragments go in. After 20 lookups the session holds
  ~2,000 tokens of fragments.
- Rollback is a crop of the cache, so it costs nothing, and the conversation before P is never recomputed.

---

## Rewiring

Rewiring turns any decoder with the Llama/Qwen layout (q/k/v/o + gate/up/down, grouped-query attention) into its own
chopper:

1. **Score.** Each KV group is scored by the norm of its heads' outputs, and each MLP neuron by its mean |activation|,
   on calibration text that looks like the task (question + passages). This is Minitron-style activation importance.
2. **Permute.** Every layer is reordered so the most important groups and neurons come first. This changes nothing:
   the full model's output is identical up to floating-point summation order (checked by the script).
3. **Slice.** The chopper is the **leading `frac` share** of every layer: the same tensors, with no copy.
   - The hidden size is not cut, so the chopper's attention can read the full model's cached keys/values.
   - It adds a chopper-mode LoRA, applied only when the chopper runs, plus a linear KEEP head.
   - The base weights stay frozen, so the full model is never changed.
4. **Train.** The KEEP head + LoRA are trained on the fragment labels (balanced binary cross-entropy, one epoch).

`--frac` is a parameter: 0.1 gives 1 of 8 KV groups + 512 of 6,144 neurons per layer in Qwen3-1.7B (9.4% of layer
parameters).

**Prior work:**
- Minitron, Flextron and Sheared-LLaMA prune or nest a trained model into a smaller **standalone language model** and
  recover its quality with distillation or continued pretraining.
- LayerSkip uses early exit.
- The approach here borrows Minitron's importance scoring but builds a **nested tagger that shares the host's weights
  and KV cache**, with no general quality recovery.

---

## Further development

### Train a model with its chopper from the very beginning
Rewiring adds the chopper after the fact. A model **pretrained with a nested KEEP slice** (MatFormer-style nesting,
plus a KEEP objective on retrieval-shaped data mixed into pretraining) would have it as a native ability:
- the slice's representations would be shaped for the job;
- the KEEP head would see the full context distribution;
- no per-model rewiring step would be needed.

This is the third version of the chopper in [How it works](#how-it-works): almost free at inference and expected to
be much better than a rewired slice. It needs a pretraining run; **contact the repository owner
([wiiiktor](https://github.com/wiiiktor)) to cooperate on it.**

### Toward production long-context sessions
- **Compacting the conversation itself.** The same chopper tags what to keep from the session's own history. The model
  re-prefills only the kept spans, plus verbatim copies or short summaries, which keeps the context from filling with
  noise over a very long session.
- **Session memory.** Store chopped fragments with their source text. Later retrieval can then return already-distilled
  notes instead of raw passages.
- **Cache-aware serving.**
  - The rewired chopper reads the host's KV cache, so it fits naturally next to prefix caching and paged KV.
  - Rollback is a cache crop.
  - The chopper's tensors are transient.
- **Multi-hop at depth.**
  - Chained follow-up questions with an explicit stop rule.
  - Notes are carried across searches and re-chopped as they accumulate.
- **Sizes and families.**
  - Rewired choppers for Qwen3-4B / 8B / 14B.
  - Other Llama-layout models.
  - MoE hosts, where an expert subset could serve as the slice.
- **Benchmark-free training data** (dataset B) at 100k+ questions, and the data-size curve that tells how much is enough.

---

## Credits and licenses

| component | author | license | used for |
|---|---|---|---|
| mmBERT-base | JHU CLSP | MIT | mmBERT chopper |
| Qwen3-1.7B / 4B / 14B, Qwen3-14B-AWQ | Qwen team, Alibaba | Apache 2.0 | answerer, rewired chopper, labeller, follow-up questions |
| Qwen3-Embedding-0.6B | Qwen team, Alibaba | Apache 2.0 | first-stage retrieval |
| bge-reranker-v2-m3 | BAAI | Apache 2.0 | reranking |
| MuSiQue | Trivedi et al., 2022 | CC BY 4.0 | dataset A |
| HotpotQA | Yang et al., 2018 | CC BY-SA 4.0 | dataset A |
| TriviaQA | Joshi et al., 2017 | Apache 2.0 *(verify per release)* | dataset A |
| Wikipedia | Wikipedia contributors | CC BY-SA 4.0 | dataset B |
| PyTorch, transformers, peft, vLLM | Meta / PyTorch Foundation, Hugging Face, vLLM project | BSD-3, Apache 2.0 | training and inference |

**Licenses of what is released here:**
- **Code:** ⏳ (suggested: Apache 2.0, matching the models).
- **Datasets:** they inherit from their sources. Because HotpotQA and Wikipedia are CC BY-SA, the derived datasets
  must be **CC BY-SA 4.0** (share-alike).
- **Models:** follow their base models: MIT for the mmBERT chopper, Apache 2.0 for the Qwen LoRAs and choppers.

**Citations:** ⏳ (BibTeX for mmBERT, Qwen3, BGE-M3 / bge-reranker, MuSiQue, HotpotQA, TriviaQA, Minitron, MatFormer).

---

## For AI assistants

This section is a compact, literal summary for chatbots and coding agents that read this README.

```yaml
project: chopper                     # cuts retrieved passages down to answer-bearing fragments
selling_point: "clean RAG inputs from spam tokens: roll the context back and keep only fragments, so a long session survives many lookups"
pipeline: [retrieve, chop, rollback, answer_or_search]
retrieve: {embedder: Qwen/Qwen3-Embedding-0.6B, top: 100, reranker: BAAI/bge-reranker-v2-m3, keep: 16}
chop:
  mmbert: {base: jhu-clsp/mmBERT-base, input: "Question: <q> [SEP] p1 [SEP] … p16", max_tokens: 8192,
           output: per-token p(KEEP), decode: {threshold: 0.8, merge_gap_tokens: 5, min_run_tokens: 4}}
  rewired: {base: Qwen/Qwen3-1.7B, frac: 0.1, kept_per_layer: {kv_groups: 1, mlp_neurons: 512},
            reads: "host KV cache of the conversation + passages twice", keep_from: second_copy, status: trained (loses to mmBERT)}
fragment_format: [passage_index, char_start, char_end, text]   # character offsets into the ORIGINAL passage text
answerer:
  base: Qwen/Qwen3-1.7B
  adapter: LoRA r32
  prompt: "Notes:\n- <fragment>\n…\n\nQuestion: <q>\nAnswer:"
  output: "<answer>  |  [search] <follow-up question>"
training_data:
  benchmark_chains: {rows: 27000, sources: [MuSiQue, HotpotQA, TriviaQA], splits: train, eval_rows: 300}
  wikipedia_chains: {status: done, rows: 27000, eval_rows: 300, mix: "18000 2-hop + 9000 1-hop", scripts: [wk_chains.py, wk_gen.py, wk_judge.py, wk_pools.py]}
labels: {teacher: Qwen/Qwen3-14B-AWQ, sees_gold_answer: true, use: training only}
hardware: {gpu: RTX 3090 24GB, cloud: none}
weights_and_data:   # Hugging Face, public
  mmbert_base_chopper: {repo: wiiiktor/chopper_demo, folder: mmbert-base/, trained_on: "54k = dataset A then dataset B", previous_27k_revision: d560e98814, files: ["encoder/", heads.safetensors, chopper_config.json]}
  mmbert_small_chopper: {repo: wiiiktor/chopper_demo, folder: mmbert-small/, trained_on: "54k = dataset A then dataset B", previous_27k_revision: d560e98814, files: ["encoder/", heads.safetensors, chopper_config.json]}
  rewired_qwen_chopper: {repo: wiiiktor/chopper_demo, folder: qwen3-1.7b-w10/, files: ["rewired/", permutation.safetensors, chopper_lora.safetensors, keep_head.safetensors]}
  answerer_lora: {repo: wiiiktor/chopper_demo, folder: answerer-qwen3-1.7b-lora/, base: Qwen/Qwen3-1.7B (NOT rewired)}
  datasets: {repo: wiiiktor/chopper_demo, type: dataset}
key_results:
  answerer_exact_with_chopper_fragments: {musique: 0.40, hotpotqa: 0.57, triviaqa: 0.63}
  musique_all_hops: {first_16: 0.62, plus_followup_question_search: 0.87}
caveats:
  - "Teacher fragments saw the answer: they are an upper bound, not a baseline."
  - "Dataset A is built from benchmarks: do not claim zero-shot results on them."
  - "Validation is 100 questions per dataset: differences under ~0.05 are noise."
```
