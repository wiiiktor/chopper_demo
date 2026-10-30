#!/usr/bin/env python
"""Write the hop-2 SUB-QUESTION for the <search> step (2026-10-29). Appending evidence to the question is a weak second
query (gold hop-1 passage → MuSiQue all-hops only .76); a written follow-up question that names the bridge entity
("Who is the spouse of Steve Hillage?") should retrieve hop 2 directly. Qwen3-14B-AWQ, WITHOUT the gold answer, reads the
question + what was found so far and writes one sub-question. Two evidence variants per val question:
  hop1gold — the gold hop-1 passage(s) (perfect reading);  chopper — the mmBERT chopper's kept fragments.
usage: .venv-vllm/bin/python mm_subq.py <mm dir> <chopper preds_val.jsonl> <out.jsonl>
       .venv-vllm/bin/python mm_subq.py --train <mm dir> <split tag> <out.jsonl>   (TRAIN targets: evidence = Qwen's
       fragments of that context, only for the contexts with a hop missing — those get "[search] <sub-question>")"""
import json, sys
from vllm import LLM, SamplingParams
TRAIN = sys.argv[1] == "--train"
if TRAIN:
    D_, TAGS, OUT = sys.argv[2:5]
    C = [json.loads(l) for l in open(f"{D_}/ctx_{TAGS}.jsonl")]
    QW = {(q["dataset"], q["row"]): q for q in map(json.loads, open(f"{D_}/qw_{TAGS}.jsonl"))}
    C = [c for c in C if c["allin"] != 1.0 and (c["dataset"], c["row"]) in QW]
    CC = {(c["dataset"], c["row"]): c for c in C}
    PRD = {k: {"fragments": [[0, 0, 0, CC[k]["passages"][kk]["text"][a:b]] for kk, a, b in QW[k]["spans"]]} for k in CC}
else:
    D_, PR, OUT = sys.argv[1:4]
    C = [json.loads(l) for l in open(f"{D_}/ctx_val.jsonl")]
    PRD = {(q["dataset"], q["row"]): q for q in map(json.loads, open(PR))}


def prompt(q, ev):
    return (f"Question: {q}\n\nWhat has been found so far:\n{ev}\n\n"
            "The question needs more than one step. Using what has been found, write the ONE next question that must be "
            "answered, naming the person, place or thing that was found (do not answer it, do not repeat the whole "
            "question). Reply with that question only.")


jobs = []
for c in C:
    key = (c["dataset"], c["row"])
    h1 = "\n".join(p["text"].strip() for p in c["passages"] if p["group"] == 0)[:3000]
    rd = "\n".join("- " + " ".join(f[3].split()) for f in PRD[key]["fragments"])[:3000] if key in PRD else ""
    for name, ev in ((("fragments", rd),) if TRAIN else (("hop1gold", h1), ("chopper", rd))):
        if ev:
            jobs.append((key, name, prompt(c["question"], ev)))
import os
llm = LLM(model="Qwen/Qwen3-14B-AWQ", max_model_len=6144, gpu_memory_utilization=float(os.environ.get("GPU_FRAC", "0.88")), seed=0)
outs = llm.chat([[{"role": "user", "content": p}] for _, _, p in jobs], SamplingParams(temperature=0.0, max_tokens=64),
                use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
R = {}
for (key, name, _), o in zip(jobs, outs):
    R.setdefault(key, {})[name] = o.outputs[0].text.strip().split("\n")[0]
with open(OUT, "w") as fo:
    for (ds, row), v in R.items():
        fo.write(json.dumps({"dataset": ds, "row": row, **v}, ensure_ascii=False) + "\n")
print("SUBQ DONE", len(R), flush=True)
