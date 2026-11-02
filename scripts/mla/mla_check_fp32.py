#!/usr/bin/env python
"""fp32 recheck of mla_rewire.py checks 2-4 (2026-11-01): in bf16 they showed max |Δlogit| 0.78 / 0.70 and |Δhidden| 6.0
of 356; in fp32 on the CPU the same comparisons must drop to rounding (~1e-3), or the latent cache / rewiring / keeper
path is wrong rather than just imprecise.
usage: .venv-rwkv7/bin/python mla_check_fp32.py <perms.pt> <probe ctx.jsonl> [--len 256]"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mla_lib as ML
from transformers import AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("perms"); ap.add_argument("probe"); ap.add_argument("--len", type=int, default=256)
A = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
torch.set_num_threads(16)
PR = torch.load(A.perms, weights_only=False); tk = AutoTokenizer.from_pretrained(PR["model"], trust_remote_code=True)
c = json.loads(open(A.probe).readline())
head = f"Question: {c['question']}\n\nPassages:\n"
ids = torch.tensor([tk.encode(head + "\n\n".join(f"[{k + 1}] {p['text'].strip()}" for k, p in enumerate(c["passages"])))[:A.len]])
M = ML.MiniCPM3(PR["model"], "cpu", torch.float32)
with torch.no_grad():
    full, _ = M.forward(ids)
    k = 100; _, cache = M.forward(ids[:, :k]); cont, _ = M.forward(ids[:, k:], cache, k)
    log(f"CHECK 2 fp32 latent cache: max |Δlogit| {(cont - full[:, k:]).abs().max():.2e} (scale {full.abs().max():.1f})")
    nc, _ = M.forward(ids[:, k:], None, k)                  # negative control: same tokens WITHOUT the cache must differ
    log(f"CHECK 2 control, no cache: max |Δlogit| {(nc - full[:, k:]).abs().max():.2e} (must be large, else check 2 is vacuous)")
    ML.rewire(M, PR["perms"]); new, _ = M.forward(ids)
    log(f"CHECK 3 fp32 rewired: max |Δlogit| {(new - full).abs().max():.2e} | argmax agree {(new.argmax(-1) == full.argmax(-1)).float().mean():.4f}")
    K = ML.Keeper(M, M.nh, M.I, r=4).eval(); ql = len(tk.encode(head))
    hfull, _ = M.forward(ids, logits=False); _, cq = M.forward(ids[:, :ql]); hk = K(ids[:, ql:], cq, ql, hidden=True)
    log(f"CHECK 4 fp32 keeper @ full width on the cached conversation: max |Δhidden| {(hk - hfull[:, ql:]).abs().max():.2e} "
        f"(scale {hfull.abs().max():.1f})")
