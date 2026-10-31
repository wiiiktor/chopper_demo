#!/usr/bin/env python
"""Step 2 of the width keeper: train + evaluate (see keeper_lib.py). The FULL rewired model prefills only the
conversation ("Question: …\\n\\nPassages:\\n"); the frac-width keeper reads the 16 passages TWICE on top of the full
model's cached keys/values and predicts KEEP per token of the second copy. Targets: Qwen3-14B fragment quotes (the
mmBERT chopper's labels, mm_qwen_labels.py), balanced BCE, one epoch, chopper-mode LoRA + KEEP head only — the base
weights are frozen and shared with the full model.
Eval (val, tokenizer-independent so it compares with the mmBERT chopper): CHARACTER P/R/F1 against Qwen's spans,
answer string inside the kept text, fragments/q, tokens/fragment, kept share — at a threshold sweep and with the
chopper's decode shaping; the same character metrics for the mmBERT chopper's preds (--baseline). Dumps preds_val.jsonl
in the chopper's format (fragments [passage, c0, c1, text]) so mm_qwen_answer.py / mm_search_loop.py can consume it.
usage: .venv-rwkv7/bin/python keeper_train.py <mm dir> <perms.pt> <out dir> [--frac 0.1] [--train-splits train,train2]"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keeper_lib import rewire, Keeper, conv_cache, sizes

ap = argparse.ArgumentParser()
ap.add_argument("mm"); ap.add_argument("perms"); ap.add_argument("out")
ap.add_argument("--frac", type=float, default=0.1); ap.add_argument("--r", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--head-lr", type=float, default=1e-3)
ap.add_argument("--accum", type=int, default=4); ap.add_argument("--max-body", type=int, default=4096)
ap.add_argument("--train-splits", default="train,train2"); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--eval-every", type=int, default=3000); ap.add_argument("--eval-n", type=int, default=0)
ap.add_argument("--pred-thr", type=float, default=0.8); ap.add_argument("--merge-gap", type=int, default=5)
ap.add_argument("--min-run", type=int, default=4)
ap.add_argument("--baseline", default="")
ap.add_argument("--val-split", default="val", help="ctx_/qw_ tag to evaluate on (val, wval)")
ap.add_argument("--eval-only", default="")
ap.add_argument("--init", default="", help="keeper.pt to CONTINUE from (LoRA + head; fresh optimiser and schedule)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); dev = os.environ.get("KDEV", "cuda"); random.seed(0); torch.manual_seed(0)
LOG = open(os.path.join(A.out, "train.log"), "a")


def log(s):
    s = time.strftime("%H:%M:%S ") + s; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


PR = torch.load(A.perms, weights_only=False)
tk = AutoTokenizer.from_pretrained(PR["model"])
M = AutoModelForCausalLM.from_pretrained(PR["model"], dtype=torch.bfloat16).to(dev).eval()
for p in M.parameters():
    p.requires_grad_(False)
rewire(M, PR["perms"])
g, n, fr = sizes(M.config, A.frac)
K = Keeper(M, g, n, r=A.r).to(dev)
nl = sum(p.numel() for p in K.lora.parameters())
SEP_IDS = tk.encode("\n\nThe same passages again:\n")


def load(split):
    C = [json.loads(l) for l in open(f"{A.mm}/ctx_{split}.jsonl")]
    L = {(d["dataset"], d["row"]): d for d in map(json.loads, open(f"{A.mm}/qw_{split}.jsonl"))}
    return [(c, L[(c["dataset"], c["row"])]) for c in C if (c["dataset"], c["row"]) in L]


def encode(c, lab):
    """conversation ids, body string + per-passage char starts, body token ids/offsets, per-token KEEP target"""
    conv = tk.encode(f"Question: {c['question']}\n\nPassages:\n")
    body, p0, lead = "", [], []
    for k, p in enumerate(c["passages"]):
        if k:
            body += "\n\n"
        body += f"[{k + 1}] "; p0.append(len(body)); lead.append(len(p["text"]) - len(p["text"].lstrip()))
        body += p["text"].strip()
    keepc = np.zeros(len(body) + 1, dtype=bool)
    for k, c0, c1 in (lab["spans"] if lab else []):
        a_, b_ = p0[k] + c0 - lead[k], p0[k] + c1 - lead[k]
        keepc[max(a_, p0[k]):max(b_, p0[k])] = True
    e = tk(body, return_offsets_mapping=True, add_special_tokens=False)
    bi, off = e["input_ids"][:A.max_body], e["offset_mapping"][:A.max_body]
    y = torch.tensor([float(keepc[a_:b_].any()) if b_ > a_ else 0.0 for a_, b_ in off])
    return conv, body, p0, lead, bi, off, y


def run(c, lab):
    conv, body, p0, lead, bi, off, y = encode(c, lab)
    ci = torch.tensor([conv], device=dev); x = torch.tensor([bi + SEP_IDS + bi], device=dev)
    with torch.autocast(dev, dtype=torch.bfloat16):
        kl = K(x, conv_cache(M, ci), len(conv))
    return kl[-len(bi):].float(), y, (body, p0, lead, off)


def runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def shape(q, gap, mn):
    q = q.copy()
    if gap:
        rr = runs(q)
        for (a0, b0), (a1, b1) in zip(rr, rr[1:]):
            if a1 - b0 <= gap:
                q[b0:a1] = True
    if mn > 1:
        for a0, b0 in runs(q):
            if b0 - a0 < mn:
                q[a0:b0] = False
    return q


def gold_chars(c, lab):
    S = set()
    for k, a, b in lab["spans"]:
        S |= {(k, i) for i in range(a, b) if not c["passages"][k]["text"][i].isspace()}
    return S


def frag_chars(c, frags):
    """[(passage, c0, c1)] in ORIGINAL passage coordinates → set of non-space chars"""
    S = set()
    for k, a, b in frags:
        t = c["passages"][k]["text"]; S |= {(k, i) for i in range(a, min(b, len(t))) if not t[i].isspace()}
    return S


def to_frags(q, meta):
    """kept token mask → fragments in original passage coordinates + text"""
    body, p0, lead, off = meta; cm = np.zeros(len(body) + 1, dtype=bool)
    for (a_, b_), k in zip(off, q):
        if k:
            cm[a_:b_] = True
    cm[1:-1] |= cm[:-2] & cm[2:]
    out = []
    for a_, b_ in runs(cm):
        k = max(0, int(np.searchsorted(np.array(p0), a_, side="right")) - 1)
        a0, b0 = max(a_, p0[k]), b_
        if b0 > a0:
            out.append([k, int(a0 - p0[k] + lead[k]), int(b0 - p0[k] + lead[k]), body[a0:b0]])
    return out


def score(acc, c, lab, frags, ntok_kept, ntok, nfrag):
    G, Pc = gold_chars(c, lab), frag_chars(c, [f[:3] for f in frags])
    acc["tp"] += len(G & Pc); acc["fp"] += len(Pc - G); acc["fn"] += len(G - Pc)
    acc["ans"] += c["answer"].lower() in " | ".join(f[3] for f in frags).lower()
    acc["n"] += 1; acc["kept"] += ntok_kept; acc["tot"] += ntok; acc["frag"] += nfrag


def report(tag, R):
    for ds, r in R.items():
        P = r["tp"] / max(1, r["tp"] + r["fp"]); Rc = r["tp"] / max(1, r["tp"] + r["fn"])
        log(f"EVAL {tag} | {ds:8s} ({r['n']} q) | CHAR P {P:.3f} R {Rc:.3f} F1 {2 * P * Rc / max(1e-9, P + Rc):.3f} | answer in kept "
            f"{r['ans'] / r['n']:.2f} | kept {r['kept'] / max(1, r['tot']):.3f} | fragments/q {r['frag'] / r['n']:.1f} "
            f"tok/fragment {r['kept'] / max(1, r['frag']):.1f}")


sync = lambda: torch.cuda.synchronize() if dev == "cuda" else None
VA = load(A.val_split)
if A.eval_n:
    VA = VA[:A.eval_n]
new_acc = lambda: {"tp": 0, "fp": 0, "fn": 0, "ans": 0, "n": 0, "kept": 0, "tot": 0, "frag": 0}


@torch.no_grad()
def evaluate(tag):
    K.eval(); TH = (0.5, 0.7, 0.8, 0.9); RS = {t: {} for t in TH}; RD = {}; PRED = []; tt = []
    for c, lab in VA:
        sync(); t0 = time.time()
        kl, y, meta = run(c, lab)
        sync(); tt.append(time.time() - t0)
        p = torch.sigmoid(kl).cpu().numpy()
        for th in TH:
            q = p > th; fr_ = to_frags(q, meta)
            score(RS[th].setdefault(c["dataset"], new_acc()), c, lab, fr_, int(q.sum()), len(q), int(q[0]) + int((q[1:] & ~q[:-1]).sum()))
        q = shape(p > A.pred_thr, A.merge_gap, A.min_run); fr_ = to_frags(q, meta)
        score(RD.setdefault(c["dataset"], new_acc()), c, lab, fr_, int(q.sum()), len(q), int(q[0]) + int((q[1:] & ~q[:-1]).sum()))
        PRED.append({"dataset": c["dataset"], "row": c["row"], "p_answer": 1.0, "fragments": fr_})
    for th in TH:
        report(f"{tag} raw p>{th}", RS[th])
    report(f"{tag} SHAPED p>{A.pred_thr} gap{A.merge_gap} min{A.min_run}", RD)
    log(f"EVAL {tag} | keeper read (conversation cache + 16 passages ×2) {1000 * np.median(tt):.0f} ms median")
    with open(os.path.join(A.out, "preds_val.jsonl"), "w") as f_:
        for x in PRED:
            f_.write(json.dumps(x, ensure_ascii=False) + "\n")
    K.train()


def wsnorm(t):
    """whitespace-collapsed text + map from each of its chars to the original index"""
    out, idx, sp = [], [], False
    for i, ch in enumerate(t):
        if ch.isspace():
            if out and not sp:
                out.append(" "); idx.append(i)
            sp = True
        else:
            out.append(ch); idx.append(i); sp = False
    return "".join(out), idx


@torch.no_grad()
def baseline():
    """the mmBERT chopper's dumped fragments, scored with the same character metric"""
    if not os.path.exists(A.baseline):
        return
    BP = {(q["dataset"], q["row"]): q for q in map(json.loads, open(A.baseline))}; R = {}
    for c, lab in VA:
        b = BP.get((c["dataset"], c["row"]))
        if not b:
            continue
        fr_ = []; NM = [wsnorm(pp["text"]) for pp in c["passages"]]
        for k, _, _, t0 in b["fragments"]:                  # a fragment can run across mmBERT's "<eos>" separator
            for t in filter(None, (x.strip() for x in t0.split("<eos>"))):
                tn = " ".join(t.split())
                for kk in [k] + [x for x in range(len(NM)) if x != k]:
                    i = NM[kk][0].find(tn)
                    if i >= 0:
                        fr_.append([kk, NM[kk][1][i], NM[kk][1][i + len(tn) - 1] + 1, t]); break
        ntok = sum(len(f[3].split()) for f in fr_)             # words, not tokens (other tokenizer): kept = word share
        score(R.setdefault(c["dataset"], new_acc()), c, lab, fr_, ntok, sum(len(p["text"].split()) for p in c["passages"]), len(fr_))
        R[c["dataset"]]["ans"] += (c["answer"].lower() in " | ".join(f[3] for f in b["fragments"]).lower()) - \
            (c["answer"].lower() in " | ".join(f[3] for f in fr_).lower())   # answer-in-kept on ALL its text, as mm_train
    report(f"BASELINE mmBERT {os.path.basename(os.path.dirname(A.baseline))}", R)


if A.init:
    ck0 = torch.load(A.init, weights_only=False); K.lora.load_state_dict(ck0["lora"]); K.keep.load_state_dict(ck0["keep"])
    log(f"continuing from {A.init} (step {ck0['step']}, splits {ck0['args']['train_splits']})")
log(f"WIDTH KEEPER {PR['model']} frac {A.frac}: {g} of {M.config.num_key_value_heads} KV groups, {n} of "
    f"{M.config.intermediate_size} neurons per layer = {fr:.3f} of the layer params | LoRA r {A.r}: {nl / 1e6:.1f}M trainable "
    f"| conversation read from the FULL model's cache, passages ×2")
baseline()
if A.eval_only:
    ck = torch.load(A.eval_only, weights_only=False); K.lora.load_state_dict(ck["lora"]); K.keep.load_state_dict(ck["keep"])
    evaluate(f"{ck['step']}"); raise SystemExit
TR = [x for sp in A.train_splits.split(",") for x in load(sp)]; random.shuffle(TR)
if A.smoke:
    TR = TR[:A.smoke]
opt = torch.optim.AdamW([{"params": K.lora.parameters(), "lr": A.lr}, {"params": K.keep.parameters(), "lr": A.head_lr}], weight_decay=0.0)
K.train(); t0 = time.time(); win = []; total = len(TR); upd = math.ceil(total / A.accum)
for step, (c, lab) in enumerate(TR, 1):
    u = (step - 1) // A.accum; f = min(1.0, (u + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u / upd)))
    opt.param_groups[0]["lr"] = A.lr * f; opt.param_groups[1]["lr"] = A.head_lr * f
    kl, y, meta = run(c, lab); y = y.to(dev)
    b = F.binary_cross_entropy_with_logits(kl, y, reduction="none")
    lk = 0.5 * (b[y > 0].mean() + b[y == 0].mean()) if (y > 0).any() else b.mean()
    (lk / A.accum).backward()
    with torch.no_grad():
        q_ = kl.detach() > 0; t_ = y > 0
        tp_, fp_, fn_ = int((q_ & t_).sum()), int((q_ & ~t_).sum()), int((~q_ & t_).sum())
    win.append((lk.item(), len(meta[3]), tp_, fp_, fn_, len(y) - tp_ - fp_ - fn_))
    if step % A.accum == 0 or step == total:
        torch.nn.utils.clip_grad_norm_(list(K.parameters()), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step % 100 == 0 or A.smoke:
        w = np.array(win); win.clear()
        tp_, fp_, fn_, tn_ = w[:, 2].sum(), w[:, 3].sum(), w[:, 4].sum(), w[:, 5].sum()
        bal = 0.5 * (tp_ / max(1, tp_ + fn_) + tn_ / max(1, tn_ + fp_)); f1_ = 2 * tp_ / max(1, 2 * tp_ + fp_ + fn_)
        log(f"step {step}/{total} | keepL {w[:, 0].mean():.3f} | train KEEP@0.5 bal-acc {bal:.3f} F1 {f1_:.3f} | passage tokens {w[:, 1].mean():.0f} (×2) | "
            f"{(time.time() - t0) / step:.2f}s/step | peak {torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0:.1f}GiB")
    if step % A.eval_every == 0 or step == total:
        torch.save({"lora": K.lora.state_dict(), "keep": K.keep.state_dict(), "args": vars(A), "step": step, "g": g, "n": n,
                    "perms": A.perms}, os.path.join(A.out, "keeper.pt"))
        evaluate(f"step {step}")
log("KEEPER TRAIN DONE")
