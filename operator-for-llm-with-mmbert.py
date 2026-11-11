#!/usr/bin/env python
"""Chopper operator with the mmBERT chopper: a long work session over your documents in which the answering model's
context grows only by the few fragments that matter, never by whole retrieved passages.

  question → retrieve 16 passages → the mmBERT chopper keeps ~3 short fragments → the model's cache is cropped back to
  the end of the session (rollback) → the model prefills only "Notes: … Question: …" → it answers, or writes
  [search] <follow-up question>, which starts the next round (up to --max-rounds)

The mmBERT chopper is independent of the answering model (text in, text out), so --model can be any Hugging Face
decoder-only causal LM (see scripts/operator_common.py for the assumptions). It answers from a short instruction, or
from your own answer-or-search LoRA (--answerer-lora <PEFT dir>, trained with scripts/answer/mm_qwen_answer.py).

usage:
  python operator-for-llm-with-mmbert.py --docs <folder of .txt/.md | file.jsonl> [--ask "question" ...]
         --model <hf model id> [--answerer-lora none|<dir>] [--chopper base|small]
         [--weights wiiiktor/chopper_mmbert] [--max-rounds 3] [--threshold <from chopper_config.json>] [--device auto]
Without --ask it opens an interactive prompt (empty line or "exit" ends the session)."""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from operator_common import Retriever, Session, device, fetch, kept_text, load_corpus, log, repl, shape

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True)
ap.add_argument("--ask", action="append")
ap.add_argument("--model", required=True, help="the answering model: any Hugging Face decoder-only causal LM")
ap.add_argument("--answerer-lora", default="none", help="none (answer from an instruction) or a PEFT dir trained for --model")
ap.add_argument("--chopper", default="base", choices=["base", "small"])
ap.add_argument("--weights", default="wiiiktor/chopper_mmbert", help="HF repo id or a local copy of it (mmbert-base/, mmbert-small/)")
ap.add_argument("--max-rounds", type=int, default=3)
ap.add_argument("--threshold", type=float, default=None, help="p(KEEP) cut-off; default = the one in the model's chopper_config.json")
ap.add_argument("--device", default="auto")
A = ap.parse_args()
dev = device(A.device); dt = torch.bfloat16 if dev == "cuda" else torch.float32

from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ---- corpus + retriever
chunks = load_corpus(A.docs)
log(f"{len(chunks):,} chunks from {A.docs}")
retrieve = Retriever(chunks, A.docs.rstrip("/") + ".chopper_index.pt", dev)

# ---- the mmBERT chopper: encoder + KEEP head
cd = fetch(A.weights, f"mmbert-{A.chopper}")
ctk = AutoTokenizer.from_pretrained(f"{cd}/encoder")
enc = AutoModel.from_pretrained(f"{cd}/encoder", dtype=dt).to(dev).eval()
HD = {k: v.to(dev).float() for k, v in load_file(f"{cd}/heads.safetensors").items()}
SEP = f" {ctk.sep_token} "
if A.threshold is None:
    import json
    A.threshold = json.load(open(f"{cd}/chopper_config.json"))["decode"]["pred_thr"]


@torch.no_grad()
def chop(question, passages):
    """Question: q [SEP] p1 [SEP] … p16 → per-token p(KEEP) → shaped runs → fragment texts"""
    head = f"Question: {question}"
    s = head + "".join(SEP + p.strip() for p in passages)
    e = ctk(s, return_offsets_mapping=True, truncation=True, max_length=8192, return_tensors="pt")
    off = e.pop("offset_mapping")[0].tolist()
    h = enc(**{k: v.to(dev) for k, v in e.items()}).last_hidden_state[0].float()
    p = torch.sigmoid(h @ HD["keep.weight"].T + HD["keep.bias"])[:, 0].cpu().numpy()
    body = np.array([a >= len(head) and b > a for a, b in off])
    q = shape((p > A.threshold) & body) & body
    out = []
    for a, b in kept_text(s, off, q):
        out += [t.strip() for t in s[a:b].split(ctk.sep_token) if len(t.strip()) > 2]
    return out


# ---- the answering model
mtk = AutoTokenizer.from_pretrained(A.model)
M = AutoModelForCausalLM.from_pretrained(A.model, dtype=dt).to(dev).eval()
lora = A.answerer_lora
if lora != "none":
    from peft import PeftModel
    M = PeftModel.from_pretrained(M, lora).eval()
log(f"answering model {A.model} ({'answerer LoRA' if lora != 'none' else 'instruction prompt'}), chopper mmBERT-{A.chopper}, {dev}")

sess = Session(M, mtk, dev, instruct=(lora == "none"))
repl(sess, retrieve, chop, A)
