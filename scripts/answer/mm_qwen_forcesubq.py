#!/usr/bin/env python
"""How good are the Qwen3-1.7B answerer's sub-questions when it is NOT allowed to answer? (2026-10-30: the search step
with its own sub-questions gave MuSiQue all-hops .73-.77 vs .87 for Qwen3-14B's, but the 14B wrote one for EVERY question
while the 1.7B only searched when it chose to — this separates question quality from the search decision.)
Loads the trained LoRA (mm_qwen_answer.py --out …/lora), forces the output to start with "[search]" and lets it write
the sub-question greedily, for every val question; writes the mm_search_loop.py --subq format (key "chopper").
usage: .venv-rwkv7/bin/python mm_qwen_forcesubq.py <mm dir> <lora dir> <notes: qwen|<chopper preds_val.jsonl>> <out.jsonl>"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
D_, LORA, NOTES, OUT = sys.argv[1:5]
MODEL = "Qwen/Qwen3-1.7B"; dev = "cuda"
tk = AutoTokenizer.from_pretrained(MODEL)
M = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev), LORA).eval()
CV = [json.loads(l) for l in open(f"{D_}/ctx_val.jsonl")]
LV = {(q["dataset"], q["row"]): q for q in map(json.loads, open(f"{D_}/qw_val.jsonl"))}
PRD = {} if NOTES == "qwen" else {(q["dataset"], q["row"]): q for q in map(json.loads, open(NOTES))}


def prompt(q, notes):                                  # identical to mm_qwen_answer.py (max-notes 1500)
    s = "Notes:\n" + ("\n".join(f"- {' '.join(n.split())}" for n in notes) if notes else "(none)") + f"\n\nQuestion: {q}\nAnswer:"
    ids = tk.encode(s)
    return ids[-(1500 + 64):] if len(ids) > 1500 + 64 else ids


n = 0
with open(OUT, "w") as fo, torch.no_grad():
    for c in CV:
        k = (c["dataset"], c["row"])
        if NOTES == "qwen":
            notes = [c["passages"][p]["text"][a:b].strip() for p, a, b in LV[k]["spans"]] if k in LV else []
        else:
            notes = [f[3] for f in PRD[k]["fragments"]] if k in PRD else []
        ids = prompt(c["question"], notes) + tk.encode(" [search]")
        out = M.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=40, do_sample=False,
                         eos_token_id=tk.eos_token_id, pad_token_id=tk.eos_token_id)
        sq = tk.decode(out[0, len(ids):], skip_special_tokens=True).strip().split("\n")[0]
        fo.write(json.dumps({"dataset": c["dataset"], "row": c["row"], "chopper": sq}, ensure_ascii=False) + "\n"); n += 1
print("FORCED SUBQ DONE", n, flush=True)
