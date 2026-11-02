#!/usr/bin/env python
"""Step 1 of the MLA width keeper (see mla_lib.py): CHECK the pure-torch MiniCPM3-4B, score, rewire, and measure how much a
ZERO-SHOT slice keeps — before any training.
  check 1  mla_lib forward == the repo's own code (trust_remote_code, no cache), logits on a probe text
  check 2  latent-cache prefill + continuation == one forward over the whole text (the cache the keeper reads)
  check 3  rewired model == original model
  check 4  Keeper at FULL width with zero LoRA, reading the cached conversation == the full model's hidden states
  drift    the slice at frac 0.05 / 0.1 / 0.2 (importance-ordered vs a RANDOM order of the same size) reads the passages
           on top of the full model's cached question; against the full model on the same tokens: cosine of the final
           hidden state, next-token top-1 agreement, KL(full ‖ slice). --qwen <perms.pt> runs the same drift for the
           Qwen GQA slice (keeper_lib) so the two architectures compare on identical contexts.
usage: .venv-rwkv7/bin/python mla_rewire.py <model dir> <calib ctx.jsonl> <probe ctx.jsonl> <out perms.pt>
       [--n 128] [--max-len 4096] [--probe-n 16] [--probe-len 2048] [--qwen <a Qwen perms .pt, optional comparison>]"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mla_lib as ML

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("calib"); ap.add_argument("probe"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=128); ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--probe-n", type=int, default=16); ap.add_argument("--probe-len", type=int, default=2048)
ap.add_argument("--qwen", default=""); ap.add_argument("--ref-id", default="openbmb/MiniCPM3-4B", help="hub id for the repo-code reference (a snapshot dir breaks its remote-code lookup)"); ap.add_argument("--skip-ref", action="store_true")
A = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
dev = "cuda"


def text(c):
    return f"Question: {c['question']}\n\nPassages:\n" + "\n\n".join(f"[{k + 1}] {p['text'].strip()}" for k, p in enumerate(c["passages"]))


def head(c):
    return f"Question: {c['question']}\n\nPassages:\n"


from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(A.model, trust_remote_code=True)
CAL = [json.loads(l) for l in open(A.calib)][:A.n]; PRB = [json.loads(l) for l in open(A.probe)][:A.probe_n]
enc = lambda s, n: torch.tensor([tk.encode(s)[:n]], device=dev)

# ---- check 1: against the repo's own modelling code
probe = enc(text(PRB[0]), 256)
if not A.skip_ref:
    try:
        from transformers import AutoModelForCausalLM
        R = AutoModelForCausalLM.from_pretrained(A.ref_id, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
        with torch.no_grad():
            ref = R(input_ids=probe, use_cache=False).logits.float()
        del R; torch.cuda.empty_cache()
    except Exception as e:
        log(f"CHECK 1 SKIPPED: the repo's code does not run here ({type(e).__name__}: {str(e)[:200]})"); ref = None
else:
    ref = None
M = ML.MiniCPM3(A.model, dev)
with torch.no_grad():
    mine, _ = M.forward(probe)
if ref is not None:
    log(f"CHECK 1 mla_lib vs repo code: max |Δlogit| {(mine - ref).abs().max():.3f} (scale {ref.abs().max():.1f}) | "
        f"argmax agree {(mine.argmax(-1) == ref.argmax(-1)).float().mean():.4f}")
nll = F.cross_entropy(mine[0, :-1], probe[0, 1:]).item()
log(f"probe next-token CE {nll:.3f} on {probe.shape[1]} tokens (a sane LM is ~2-3 on Wikipedia text)")

# ---- check 2: latent cache
with torch.no_grad():
    k = 100; _, cache = M.forward(probe[:, :k]); cont, _ = M.forward(probe[:, k:], cache, k)
log(f"CHECK 2 latent cache: max |Δlogit| prefill+continue vs one pass {(cont - mine[:, k:]).abs().max():.3f}")

# ---- importance + rewire (check 3)
t0 = time.time()
imp = ML.importance(M, (enc(text(c), A.max_len) for c in CAL))
log(f"importance on {len(CAL)} contexts: {time.time() - t0:.0f}s")
for i in (0, M.L // 2, M.L - 1):
    a, b = imp[i]; bs = b.sort(descending=True).values; as_ = a.sort(descending=True).values
    log(f"layer {i}: top-10% neurons hold {bs[:len(bs) // 10].sum() / bs.sum():.2f} of |act| | top-4 heads hold "
        f"{as_[:4].sum() / as_.sum():.2f} of head-output norm (uniform 0.10)")
P = ML.perms_from(imp)
ML.rewire(M, P)
with torch.no_grad():
    new, _ = M.forward(probe)
log(f"CHECK 3 rewired: max |Δlogit| {(new - mine).abs().max():.3f} | argmax agree {(new.argmax(-1) == mine.argmax(-1)).float().mean():.4f}")

# ---- check 4: full-width keeper over the cached conversation
with torch.no_grad():
    K = ML.Keeper(M, M.nh, M.I, r=4).to(dev).eval()
    ql = len(tk.encode(head(PRB[0])))
    hfull, _ = M.forward(probe, logits=False); _, cq = M.forward(probe[:, :ql])
    hk = K(probe[:, ql:], cq, ql, hidden=True)
log(f"CHECK 4 keeper @ full width on the cached conversation: max |Δhidden| {(hk - hfull[:, ql:]).abs().max():.3f} "
    f"(scale {hfull.abs().max():.1f})")
del K


@torch.no_grad()
def drift(full_hidden_logits, slice_hidden_logits):
    """per token: cosine of the final hidden, top-1 agreement, KL(full‖slice) — summed over the probe contexts"""
    cs, ag, kl, n = 0.0, 0.0, 0.0, 0
    for c in PRB:
        ids = enc(text(c), A.probe_len); ql = len(tk.encode(head(c)))
        hf, lf = full_hidden_logits(ids); hs, ls = slice_hidden_logits(ids, ql)
        hf, lf = hf[0, ql:].float(), lf[0, ql:].float(); hs, ls = hs.float(), ls.float()
        cs += F.cosine_similarity(hf, hs, dim=-1).sum().item(); ag += (lf.argmax(-1) == ls.argmax(-1)).sum().item()
        kl += F.kl_div(ls.log_softmax(-1), lf.log_softmax(-1), log_target=True, reduction="sum").item(); n += hf.shape[0]
    return cs / n, ag / n, kl / n


def mla_full(ids):
    h, _ = M.forward(ids, logits=False); return h, F.linear(h / M.logit_div, M.lm_head)


def mla_slice(h_, n_):
    K = ML.Keeper(M, h_, n_, r=4).to(dev).eval()            # LoRA B = 0: the bare slice

    def f(ids, ql):
        _, cq = M.forward(ids[:, :ql])
        with torch.no_grad():
            h = K(ids[:, ql:], cq, ql, hidden=True)[0]
        return h, F.linear(h / M.logit_div, M.lm_head)
    return f


out = {"model": A.model, "imp": imp, "perms": P, "n_calib": len(CAL), "drift": {}}
for frac in (0.05, 0.1, 0.2):
    h_, n_, fr = ML.sizes(M, frac)
    c1, a1, k1 = drift(mla_full, mla_slice(h_, n_))
    RP = [(torch.randperm(M.nh), torch.randperm(M.I)) for _ in range(M.L)]
    ML.rewire(M, RP)                                        # same size, random members
    c0, a0, k0 = drift(mla_full, mla_slice(h_, n_))
    ML.rewire(M, [(a.argsort(), b.argsort()) for a, b in RP])   # undo: back to the importance order
    log(f"DRIFT MiniCPM3 frac {frac}: {h_} of {M.nh} heads + {n_} of {M.I} neurons = {fr:.3f} of layer params | "
        f"importance slice cos {c1:.3f} top-1 {a1:.3f} KL {k1:.2f} | random slice cos {c0:.3f} top-1 {a0:.3f} KL {k0:.2f}")
    out["drift"][frac] = {"h": h_, "n": n_, "share": fr, "imp": (c1, a1, k1), "random": (c0, a0, k0)}
torch.save(out, A.out)
log("MLA REWIRE DONE", A.out)

if A.qwen:                                                  # the same drift for the Qwen GQA slice
    del M; torch.cuda.empty_cache()
    import keeper_lib as KL
    from transformers import AutoModelForCausalLM
    PQ = torch.load(A.qwen, weights_only=False)
    tk = AutoTokenizer.from_pretrained(PQ["model"])
    Q = AutoModelForCausalLM.from_pretrained(PQ["model"], dtype=torch.bfloat16).to(dev).eval()
    for p in Q.parameters():
        p.requires_grad_(False)
    KL.rewire(Q, PQ["perms"])

    def q_full(ids):
        with torch.no_grad():
            h = Q.model(input_ids=ids).last_hidden_state
        return h, Q.lm_head(h)

    def q_slice(g, n):
        def f(ids, ql):
            K = KL.Keeper(Q, g, n, r=4, ckpt=False).to(dev).eval()
            with torch.no_grad():
                cv = KL.conv_cache(Q, ids[:, :ql]); hk = {}
                orig = K.keep.forward; K.keep.forward = lambda x: (hk.__setitem__("h", x), orig(x))[1]
                K(ids[:, ql:], cv, ql); h = hk["h"][0].to(torch.bfloat16)
            return h, Q.lm_head(h)
        return f
    for frac in (0.05, 0.1, 0.2):
        g, n, fr = KL.sizes(Q.config, frac)
        c1, a1, k1 = drift(q_full, q_slice(g, n))
        log(f"DRIFT Qwen3-1.7B frac {frac}: {g} of {Q.config.num_key_value_heads} KV groups + {n} neurons = {fr:.3f} | "
            f"importance slice cos {c1:.3f} top-1 {a1:.3f} KL {k1:.2f}")
        out["drift"][f"qwen_{frac}"] = {"g": g, "n": n, "share": fr, "imp": (c1, a1, k1)}
    torch.save(out, A.out)
    log("QWEN DRIFT DONE")
