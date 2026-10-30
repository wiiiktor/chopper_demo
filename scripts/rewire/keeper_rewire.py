#!/usr/bin/env python
"""Step 1 of the width keeper: score, rewire, CHECK (see keeper_lib.py). Scores every KV group and MLP neuron of <model>
on calibration contexts (question + 16 passages, the keeper's own input), saves the permutation, and verifies on held-out
text that (a) the rewired full model gives the same logits and (b) the keeper at FULL width with zero LoRA reproduces
the full model exactly, including reading the full model's cached conversation (the mixed attention).
usage: .venv-rwkv7/bin/python keeper_rewire.py <model> <ctx.jsonl> <out perms.pt> [--n 128] [--max-len 4096] [--cpu]"""
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keeper_lib import importance, rewire, perms_from, Keeper, conv_cache, sizes

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("ctx"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=128); ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--cpu", action="store_true", help="fp32 on CPU (exactness check only, slow)")
A = ap.parse_args()
dev = "cpu" if A.cpu else "cuda"; dt = torch.float32 if A.cpu else torch.bfloat16
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
tk = AutoTokenizer.from_pretrained(A.model)
M = AutoModelForCausalLM.from_pretrained(A.model, dtype=dt).to(dev).eval()
C = [json.loads(l) for l in open(A.ctx)][:A.n + 1]


def text(c):
    return f"Question: {c['question']}\n\nPassages:\n" + "\n\n".join(f"[{k + 1}] {p['text'].strip()}" for k, p in enumerate(c["passages"]))


ids = lambda s, n=A.max_len: torch.tensor([tk.encode(s)[:n]], device=dev)
t0 = time.time()
imp = importance(M, (ids(text(c)) for c in C[:A.n]))
log(f"importance on {A.n} contexts: {time.time() - t0:.0f}s")
P = perms_from(imp)
for i in (0, len(imp) // 2, len(imp) - 1):                 # how concentrated is importance? share held by the top 10%
    g, n = imp[i]; ns = n.sort(descending=True).values; gs_ = g.sort(descending=True).values
    log(f"layer {i}: top-10% neurons hold {ns[:len(ns) // 10].sum() / ns.sum():.2f} of |act| | groups {[round(float(x), 1) for x in gs_]}")
probe = ids(text(C[A.n]), 512); qlen = len(tk.encode(f"Question: {C[A.n]['question']}\n\nPassages:\n"))
with torch.no_grad():
    ref = M(input_ids=probe).logits.float(); href = M.model(input_ids=probe).last_hidden_state.float()
    rewire(M, P)
    new = M(input_ids=probe).logits.float()
    log(f"CHECK rewired full model: max |Δlogit| {(new - ref).abs().max():.2e} (logit scale {ref.abs().max():.1f}) | "
        f"argmax agree {(new.argmax(-1) == ref.argmax(-1)).float().mean():.4f}")
    H, hd, G, qpg, I = __import__("keeper_lib").geom(M.config)
    K = Keeper(M, G, I, r=4).to(dev).eval()                 # full width, LoRA B = 0 → must equal the full model
    K.keep.weight.data.zero_()
    import keeper_lib
    hk = {}
    orig = K.keep.forward
    K.keep.forward = lambda x: (hk.__setitem__("h", x), orig(x))[1]
    K(probe[:, qlen:], conv_cache(M, probe[:, :qlen]), qlen)
    d = (hk["h"][0] - href[0, qlen:]).abs().max()
    log(f"CHECK keeper @ full width reading the cached conversation: max |Δhidden| {d:.2e} (scale {href.abs().max():.1f})")
for frac in (0.05, 0.1, 0.2, 0.3):
    g, n, f = sizes(M.config, frac); log(f"frac {frac}: {g} of {G} KV groups, {n} of {I} neurons per layer = {f:.3f} of layer params")
torch.save({"model": A.model, "imp": imp, "perms": P, "n_calib": A.n}, A.out)
log("REWIRE DONE", A.out)
