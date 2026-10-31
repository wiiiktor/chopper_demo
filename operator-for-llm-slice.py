#!/usr/bin/env python
"""Chopper operator with the 10% SLICE chopper: the chopper is a ~10% width slice of the answering model itself
(rewired Qwen3-1.7B + a small LoRA + KEEP head), so it costs no second model and reads the session from the answering
model's own KV cache.

  question → retrieve 16 passages → the model's cache gets "Question: …\\n\\nPassages:\\n" → the slice reads the 16
  passages TWICE on top of that cache and keeps ~3 fragments from the second copy → ROLLBACK: the cache is cropped
  back to the end of the session (the full model never read the passages) → the model prefills only
  "Notes: … Question: …" → it answers, or writes [search] <follow-up question> for the next round

Needs a model whose slice was trained: by default Qwen/Qwen3-1.7B with the files in wiiiktor/chopper_demo
(qwen3-1.7b-w10/). For another Qwen / Llama-layout model, make them with scripts/rewire/keeper_rewire.py and
keeper_train.py and point --slice at that folder (perms .pt + keeper.pt).

The answer-or-search LoRA was trained on the plain Qwen3-1.7B; here it is permuted into the rewired layout, so it acts
exactly as on the plain model. While the slice reads, the LoRA is switched off, as in the slice's training.

usage:
  python operator-for-llm-slice.py --docs <folder of .txt/.md | file.jsonl> [--ask "question" ...]
         [--slice <dir with perms_*.pt + keeper.pt>] [--answerer-lora auto|none|<dir>] [--weights wiiiktor/chopper_demo]
         [--max-rounds 3] [--threshold 0.8] [--device auto]"""
import argparse, glob, os, re, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts")); sys.path.insert(0, os.path.join(HERE, "scripts", "rewire"))
from operator_common import Retriever, Session, device, fetch, kept_text, load_corpus, log, repl, shape
from keeper_lib import Keeper, geom, rewire

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True)
ap.add_argument("--ask", action="append")
ap.add_argument("--slice", default="", help="folder with perms_*.pt and keeper.pt (default: qwen3-1.7b-w10 from --weights)")
ap.add_argument("--answerer-lora", default="auto", help="auto = ours (Qwen3-1.7B only); none; or a PEFT dir for the plain model")
ap.add_argument("--weights", default="wiiiktor/chopper_demo")
ap.add_argument("--max-rounds", type=int, default=3)
ap.add_argument("--threshold", type=float, default=0.8)
ap.add_argument("--device", default="auto")
A = ap.parse_args()
dev = device(A.device); dt = torch.bfloat16 if dev == "cuda" else torch.float32

from transformers import AutoModelForCausalLM, AutoTokenizer

chunks = load_corpus(A.docs)
log(f"{len(chunks):,} chunks from {A.docs}")
retrieve = Retriever(chunks, A.docs.rstrip("/") + ".chopper_index.pt", dev)

# ---- the rewired model and its slice
sd_ = A.slice or fetch(A.weights, "qwen3-1.7b-w10", files=("perms_q17.pt", "keeper.pt"))
PR = torch.load(glob.glob(os.path.join(sd_, "perms*.pt"))[0], weights_only=False)
KP = torch.load(os.path.join(sd_, "keeper.pt"), map_location="cpu", weights_only=False)
tk = AutoTokenizer.from_pretrained(PR["model"])
M = AutoModelForCausalLM.from_pretrained(PR["model"], dtype=dt).to(dev).eval()
with torch.no_grad():
    rewire(M, PR["perms"])                                     # exact: the full model's outputs do not change
for p in M.parameters():
    p.requires_grad_(False)
K = Keeper(M, KP["g"], KP["n"], r=KP["args"]["r"], ckpt=False).to(dev).eval()
K.lora.load_state_dict(KP["lora"]); K.keep.load_state_dict(KP["keep"]); K.to(dtype=torch.float32)

# ---- the answerer LoRA, permuted into the rewired layout
lora = A.answerer_lora
if lora == "auto":
    lora = fetch(A.weights, "answerer-qwen3-1.7b-lora") if PR["model"] == "Qwen/Qwen3-1.7B" else "none"
host = M
if lora != "none":
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    H, hd, G, qpg, I = geom(M.config); sd = load_file(os.path.join(lora, "adapter_model.safetensors"))
    for k in list(sd):
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)_proj\.lora_([AB])", k)
        if not m:
            continue
        i, mod, ab = int(m.group(1)), m.group(2), m.group(3); pg, pn = PR["perms"][i]
        qi = torch.cat([torch.arange(j * qpg * hd, (j + 1) * qpg * hd) for j in pg.tolist()])
        ki = torch.cat([torch.arange(j * hd, (j + 1) * hd) for j in pg.tolist()])
        idx = {"q": qi, "k": ki, "v": ki, "o": qi, "gate": pn, "up": pn, "down": pn}[mod]
        if ab == "B" and mod in ("q", "k", "v", "gate", "up"):  # output rows follow the permuted rows
            sd[k] = sd[k][idx]
        elif ab == "A" and mod in ("o", "down"):                # input columns follow the permuted columns
            sd[k] = sd[k][:, idx]
    host = get_peft_model(M, LoraConfig.from_pretrained(lora)).eval()
    set_peft_model_state_dict(host, sd)
log(f"rewired {PR['model']}, slice {KP['g']} KV group(s) + {KP['n']} MLP neurons per layer, "
    f"{'answerer LoRA' if lora != 'none' else 'instruction prompt'}, {dev}")
sess = Session(host, tk, dev, instruct=(lora == "none"))
SEP_IDS = tk.encode("\n\nThe same passages again:\n")


def cache_kv():
    c = sess.cache
    return [(ly.keys, ly.values) for ly in c.layers] if hasattr(c, "layers") else [(c[i][0], c[i][1]) for i in range(len(c))]


@torch.no_grad()
def chop(question, passages):
    mark = sess.len
    off_ctx = host.disable_adapter() if host is not M else torch.no_grad()
    with off_ctx:
        sess.prefill(f"Question: {question}\n\nPassages:\n")    # the slice's training prompt, on top of the session
    body, p0 = "", []
    for k, p in enumerate(passages):
        body += ("\n\n" if k else "") + f"[{k + 1}] "; p0.append(len(body)); body += p.strip()
    e = tk(body, return_offsets_mapping=True, add_special_tokens=False)
    bi, off = e["input_ids"][:4096], e["offset_mapping"][:4096]
    x = torch.tensor([bi + SEP_IDS + bi], device=dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        kl = K(x, cache_kv(), sess.len)                       # passages ×2; KEEP from the second copy
    sess.crop(mark)                                            # ROLLBACK: the full model never read the passages
    p = torch.sigmoid(kl[-len(bi):].float()).cpu().numpy()
    q = shape(p > A.threshold)
    out = []
    for a, b in kept_text(body, off, q):
        out += [t.strip() for t in re.split(r"\n\n\[\d+\] ", body[a:b]) if len(t.strip()) > 2]
    return [re.sub(r"^\[\d+\] ", "", t) for t in out]


repl(sess, retrieve, chop, A)
