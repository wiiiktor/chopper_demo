#!/usr/bin/env python
"""Read-time speed-ups for the MLA width keeper (2026-11-02, user: "1 2 3 ok"). None of them change what the slice computes:
  A  as trained: the full model prefills the conversation, the slice + LoRA reads the passages ×2 (mla_keeper_train run())
  B  LoRA MERGED into the slice's own weight copies (q_a, kv_a whole; q_b/kv_b/o/gate/up/down sliced) → no adapter matmuls
  C  B + torch.compile(mode="max-autotune-no-cudagraphs"), passages padded to a 512-token bucket (padding at the END of a
     causal sequence cannot change the earlier tokens)
  D  B + torch.compile(mode="reduce-overhead") = CUDA graphs, one graph per bucket
Check: on every timed context, KEEP probabilities of B/C/D against A (max |Δp|, agreement of p>0.8) — must be ~bf16 noise.
Also times the conversation prefill alone (the host has that cache already in production).
usage: .venv-rwkv7/bin/python mla_speed.py <perms.pt> <keeper.pt> <ctx_val.jsonl> [--frac 0.2] [--n 40] [--bucket 512]"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mla_lib as ML

ap = argparse.ArgumentParser()
ap.add_argument("perms"); ap.add_argument("keeper"); ap.add_argument("ctx")
ap.add_argument("--frac", type=float, default=0.2); ap.add_argument("--r", type=int, default=32)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--bucket", type=int, default=512)
ap.add_argument("--max-body", type=int, default=4096); ap.add_argument("--modes", default="A,B,C,D")
ap.add_argument("--dev", default="cuda")
A = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
dev = A.dev; torch.manual_seed(0)
torch._dynamo.config.cache_size_limit = 512   # one graph per (conversation length, passage bucket)

from transformers import AutoTokenizer
PR = torch.load(A.perms, weights_only=False)
tk = AutoTokenizer.from_pretrained(PR["model"], trust_remote_code=True)
M = ML.MiniCPM3(PR["model"], dev); ML.rewire(M, PR["perms"])
h, n, fr = ML.sizes(M, A.frac)
K = ML.Keeper(M, h, n, r=A.r).to(dev).eval()
ck = torch.load(A.keeper, weights_only=False); K.lora.load_state_dict(ck["lora"]); K.keep.load_state_dict(ck["keep"])
log(f"slice {h}/{M.nh} heads + {n}/{M.I} neurons ({fr:.3f} of layer params), keeper step {ck['step']}")
SEP = tk.encode("\n\nThe same passages again:\n", add_special_tokens=False)

CTX = [json.loads(l) for l in open(A.ctx)][:A.n]


def inputs(c):                                              # as mla_keeper_train encode()
    conv = tk.encode(f"Question: {c['question']}\n\nPassages:\n")
    body = "\n\n".join(f"[{k + 1}] " + p["text"].strip() for k, p in enumerate(c["passages"]))
    bi = tk(body, add_special_tokens=False)["input_ids"][:A.max_body]
    return torch.tensor([conv], device=dev), torch.tensor([bi + SEP + bi], device=dev), len(bi)


IN = [inputs(c) for c in CTX]

# ---- merged weights: the slice's own copies, LoRA folded in (W' = W + B A, in fp32, stored bf16)
dq, dn, dv = M.dq, M.dn, M.dv
MW = []
with torch.no_grad():
    for i, l in enumerate(M.layers):
        lo = K.lora[i]; d = lambda k: (lo[k].B.float() @ lo[k].A.float())
        MW.append({"ln1": l["ln1"], "ln2": l["ln2"], "q_a_ln": l["q_a_ln"], "kv_a_ln": l["kv_a_ln"],
                   "q_a": (l["q_a"].float() + d("q_a")).to(M.dt), "kv_a": (l["kv_a"].float() + d("kv_a")).to(M.dt),
                   "q_b": (l["q_b"][:h * dq].float() + d("q_b")).to(M.dt), "kv_b": (l["kv_b"][:h * (dn + dv)].float() + d("kv_b")).to(M.dt),
                   "o": (l["o"][:, :h * dv].float() + d("o")).to(M.dt), "gate": (l["gate"][:n].float() + d("gate")).to(M.dt),
                   "up": (l["up"][:n].float() + d("up")).to(M.dt), "down": (l["down"][:, :n].float() + d("down")).to(M.dt)})
keep_w, keep_b = K.keep.weight.detach().float(), K.keep.bias.detach().float()


def merged_block(w, x, cos, sin, Cp, Pp):
    """one layer of the slice on merged weights; Cp/Pp = the conversation's cached latents of this layer"""
    T = x.shape[1]; a = ML.rms(x, w["ln1"], M.eps)
    qa = ML.rms(F.linear(a, w["q_a"]), w["q_a_ln"], M.eps)
    q = F.linear(qa, w["q_b"]).view(1, T, h, dq).transpose(1, 2); qn, qp = q.split([dn, M.dr], -1)
    qp = (qp.float() * cos + ML.rot_half(qp.float()) * sin).to(x.dtype)
    ckv = F.linear(a, w["kv_a"]); c, kpe = ckv.split([M.R, M.dr], -1)
    kpe = (kpe.float()[:, None] * cos + ML.rot_half(kpe.float()[:, None]) * sin)[:, 0].to(x.dtype)
    c = ML.rms(c, w["kv_a_ln"], M.eps)
    C, P = torch.cat([Cp, c], 1), torch.cat([Pp, kpe], 1)
    kv = F.linear(C, w["kv_b"]).view(1, C.shape[1], h, dn + dv).transpose(1, 2); kn, v = kv.split([dn, dv], -1)
    k = torch.cat([kn, P[:, None].expand(1, h, -1, M.dr)], -1)
    o = F.scaled_dot_product_attention(torch.cat([qn, qp], -1), k, v, attn_mask=ML.causal_lower_right(T, k.shape[2]), scale=dq ** -0.5)
    x = x + F.linear(o.transpose(1, 2).reshape(1, T, h * dv), w["o"]) * M.res
    b = ML.rms(x, w["ln2"], M.eps)
    return x + F.linear(F.silu(F.linear(b, w["gate"])) * F.linear(b, w["up"]), w["down"]) * M.res


def merged(x, cache, pos0):
    e = F.embedding(x, M.embed) * M.emb_scale
    cos, sin = M.rope(torch.arange(pos0, pos0 + x.shape[1], device=dev))
    for w, (Cp, Pp) in zip(MW, cache):
        e = merged_block(w, e, cos, sin, Cp, Pp)
    return F.linear(ML.rms(e, M.norm, M.eps).float(), keep_w, keep_b)[0, :, 0]


def pad(x):
    T = x.shape[1]; Tp = -(-T // A.bucket) * A.bucket
    return F.pad(x, (0, Tp - T), value=0), T


@torch.no_grad()
def read(mode, conv, x):
    _, cache = M.forward(conv, logits=False)
    if mode == "A":
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            return K(x, cache, conv.shape[1]).float()
    fn = {"B": merged, "C": FC, "D": FD}[mode]
    if mode == "B":
        return fn(x, cache, conv.shape[1])
    xp, T = pad(x)
    return fn(xp, cache, conv.shape[1])[:T].clone()


FC = FD = None
modes = A.modes.split(",")
if "C" in modes:
    FC = torch.compile(merged, mode="max-autotune-no-cudagraphs", dynamic=False)
if "D" in modes:
    FD = torch.compile(merged, mode="reduce-overhead", dynamic=False)

sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
with torch.no_grad():                                       # conversation prefill alone
    tt = []
    for conv, x, nb in IN:
        sync(); t0 = time.time(); M.forward(conv, logits=False); sync(); tt.append(time.time() - t0)
    log(f"conversation prefill alone (full 4B model, ~{np.mean([c.shape[1] for c, _, _ in IN]):.0f} tokens): {1000 * np.median(tt[min(2, len(tt) - 1):]):.0f} ms median")
REF = {}
for mode in modes:
    t_w = time.time()
    for conv, x, nb in IN:                                  # warm-up / compile every bucket once
        read(mode, conv, x)
    sync(); log(f"mode {mode}: warm-up (incl. compile) {time.time() - t_w:.0f}s")
    tt, dp, ag = [], [], []
    for j, (conv, x, nb) in enumerate(IN):
        sync(); t0 = time.time(); kl = read(mode, conv, x); sync(); tt.append(time.time() - t0)
        p = torch.sigmoid(kl[-nb:]).cpu()
        if mode == "A":
            REF[j] = p
        else:
            dp.append((p - REF[j]).abs().max().item()); ag.append(((p > 0.8) == (REF[j] > 0.8)).float().mean().item())
    chk = f" | vs A: max |Δp| {max(dp):.4f}, p>0.8 agreement {np.mean(ag):.5f} (min {min(ag):.5f})" if dp else ""
    log(f"MODE {mode}: read {1000 * np.median(tt):.0f} ms median ({1000 * np.percentile(tt, 90):.0f} p90) over {len(tt)} contexts, "
        f"~{np.mean([x.shape[1] for _, x, _ in IN]):.0f} passage tokens{chk}")
log("MLA SPEED DONE")
