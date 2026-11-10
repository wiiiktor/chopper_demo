#!/usr/bin/env python
"""mmBERT chopper, pass 1 (2026-10-29, user: "switch to mmbert with conversation tail and top 16 snippets, and decisions
search / answer"; "use mmbert a few times, starting with snippets, finishing with a lot of short fragments after a longer
conversation tail"). jhu-clsp/mmBERT-base (0.31B, MIT, 8k context, bidirectional) reads ONE input:
    "Question: <conversation tail>" [SEP] passage 1 [SEP] … passage 16          (≤ --max-len tokens)
and predicts
  * KEEP per token  — target: Qwen3-14B fragment quotes (mm_qwen_labels.py, character spans → tokens by offsets)
  * a decision at [CLS] — ANSWER if every hop's gold passage is among the 16 (ctx allin), else SEARCH
Losses: balanced BCE on KEEP (mean over kept + mean over not-kept tokens, /2) + class-balanced CE on the decision.
Eval on val: KEEP P/R/F1 vs Qwen (threshold sweep), fragments/q and tokens/fragment, answer string inside the kept
text, decision accuracy per class. One epoch over the train contexts (project rule), batch --accum.
usage: .venv-gen/bin/python mm_train.py <mm dir> <out dir> [--lr 3e-5] [--accum 4] [--max-len 8192] [--eval-every 1000]"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("mm"); ap.add_argument("out")
ap.add_argument("--base", default="jhu-clsp/mmBERT-base")
ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
ap.add_argument("--accum", type=int, default=4); ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hop-w", type=float, default=1.0, help="weight of kept tokens in later-hop gold passages (group >= 1); 2026-11-08: MuSiQue keeps hop 1 98%% of the time but hop 2 only 58%%")
ap.add_argument("--keep-w", type=float, default=0.5, help="KEEP loss share of the kept tokens (2026-11-06: 0.5 = balanced, which with ~5% kept tokens weighs a kept token ~19x and tells every row it holds an answer)")
ap.add_argument("--train-splits", default="train", help="comma list of ctx_/qw_ file tags to train on (e.g. train,train2)")
ap.add_argument("--val-split", default="val", help="ctx_/qw_ tag to evaluate on (val, wval)")
ap.add_argument("--eval-only", default="", help="chopper.pt (or an older reader.pt) to load; evaluate on val and dump preds, no training")
ap.add_argument("--recycle", type=int, default=0, help="2026-11-10 multi-hop chopper: re-run the LAST N encoder layers a second time with the "
                "first pass's KEEP probability added to every token (zero-initialised projection, AlphaFold-style recycling); loss on both passes, "
                "the second pass is the output. 0 = one pass (the old chopper)")
ap.add_argument("--rec-w1", type=float, default=0.5, help="weight of the first pass's KEEP loss when recycling")
ap.add_argument("--init", default="", help="chopper.pt (or an older reader.pt) to CONTINUE training from (fresh optimiser and schedule)")
ap.add_argument("--merge-gap", type=int, default=0, help="decode: fill gaps of ≤ this many tokens between kept tokens")
ap.add_argument("--min-run", type=int, default=1, help="decode: drop kept runs shorter than this many tokens")
ap.add_argument("--pred-thr", type=float, default=0.5, help="KEEP threshold for the dumped fragments (2026-10-29: p>0.5 "
                "keeps ~12% where Qwen keeps ~2%; p>0.9 = P .36 R .53)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
dev = "cuda"; torch.manual_seed(A.seed); random.seed(A.seed)
LOG = open(os.path.join(A.out, "train.log"), "a")


def log(s):
    s = time.strftime("%H:%M:%S ") + s; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


tk = AutoTokenizer.from_pretrained(A.base)
SEP = f" {tk.sep_token} " if tk.sep_token else "\n\n"


def load(split):
    C = [json.loads(l) for l in open(f"{A.mm}/ctx_{split}.jsonl")]
    L = {(d["dataset"], d["row"]): d for d in map(json.loads, open(f"{A.mm}/qw_{split}.jsonl"))}
    return [(c, L[(c["dataset"], c["row"])]) for c in C if (c["dataset"], c["row"]) in L]


def encode(c, lab):
    """one input string, its tokens (with char offsets), per-token KEEP targets, the decision target"""
    s = f"Question: {c['question']}"; p0 = []
    for p in c["passages"]:
        s += SEP; p0.append(len(s)); s += p["text"].strip()
    lead = [len(p["text"]) - len(p["text"].lstrip()) for p in c["passages"]]
    keepc = np.zeros(len(s) + 1, dtype=bool)
    for k, c0, c1 in lab["spans"]:
        a_, b_ = p0[k] + c0 - lead[k], p0[k] + c1 - lead[k]
        keepc[max(a_, 0):max(b_, 0)] = True
    e = tk(s, return_offsets_mapping=True, truncation=True, max_length=A.max_len)
    y = [float(keepc[a_:b_].any()) if b_ > a_ else 0.0 for a_, b_ in e["offset_mapping"]]
    hopc = np.zeros(len(s) + 1, dtype=bool)
    for k, c0, c1 in lab["spans"]:
        if c["passages"][k].get("group", -1) >= 1:
            hopc[max(p0[k] + c0 - lead[k], 0):max(p0[k] + c1 - lead[k], 0)] = True
    h = [float(hopc[a_:b_].any()) if b_ > a_ else 0.0 for a_, b_ in e["offset_mapping"]]
    dec = 1 if c["allin"] == 1.0 else 0                   # 1 = ANSWER, 0 = SEARCH
    return s, e, torch.tensor(y), dec, torch.tensor(h)


class Chopper(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(A.base, dtype=torch.bfloat16)
        H = self.enc.config.hidden_size
        self.keep = nn.Linear(H, 1); self.dec = nn.Linear(H, 2)
        self.rec = nn.Linear(1, H)                     # recycling: first-pass KEEP probability -> hidden-size offset, starts at zero
        nn.init.zeros_(self.rec.weight); nn.init.zeros_(self.rec.bias)

    def forward(self, ids, both=False):
        if not A.recycle:
            h = self.enc(input_ids=ids).last_hidden_state[0].float()
            k, d = self.keep(h)[:, 0], self.dec(h[0])
            return (k, d, None) if both else (k, d)
        import transformers.models.modernbert.modeling_modernbert as MB
        e = self.enc; x = e.embeddings(input_ids=ids); pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        mk = {"config": e.config, "inputs_embeds": x, "attention_mask": None}
        am = {"full_attention": MB.create_bidirectional_mask(**mk), "sliding_attention": MB.create_bidirectional_sliding_window_mask(**mk)}
        pe = {t: e.rotary_emb(x, pos, t) for t in set(e.config.layer_types)}
        cut = len(e.layers) - A.recycle
        for i, L in enumerate(e.layers):
            if i == cut:
                x_cut = x
            x = L(x, attention_mask=am[L.attention_type], position_embeddings=pe[L.attention_type])
        k1 = self.keep(e.final_norm(x)[0].float())[:, 0]
        x = x_cut + self.rec(torch.sigmoid(k1).detach().unsqueeze(-1)).to(x_cut.dtype).unsqueeze(0)
        for L in e.layers[cut:]:
            x = L(x, attention_mask=am[L.attention_type], position_embeddings=pe[L.attention_type])
        h = e.final_norm(x)[0].float()
        k2, d = self.keep(h)[:, 0], self.dec(h[0])
        return (k2, d, k1) if both else (k2, d)


M = Chopper().to(dev)
M.enc.gradient_checkpointing_enable() if hasattr(M.enc, "gradient_checkpointing_enable") else None
opt = torch.optim.AdamW([{"params": M.enc.parameters(), "lr": A.lr},
                         {"params": list(M.keep.parameters()) + list(M.dec.parameters()) + list(M.rec.parameters()), "lr": A.head_lr}], weight_decay=0.0)
TR, VA = [x for sp in A.train_splits.split(",") for x in load(sp)], load(A.val_split)
random.shuffle(TR)
if A.smoke:
    TR, VA = TR[:A.smoke], VA[:max(6, A.smoke)]
pa = np.mean([c["allin"] == 1.0 for c, _ in TR]); W = torch.tensor([0.5 / max(1 - pa, 1e-3), 0.5 / max(pa, 1e-3)], device=dev, dtype=torch.float)
log(f"mmBERT chopper: {len(TR)} train / {len(VA)} val contexts | ANSWER share {pa:.2f} | lr {A.lr} head {A.head_lr} accum {A.accum} "
    f"max-len {A.max_len} | {sum(p.numel() for p in M.parameters()) / 1e6:.0f}M params")


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


def runs(m):
    """(start, end) of every run of True"""
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


@torch.no_grad()
def evaluate(step):
    M.eval(); TH = (0.3, 0.5, 0.7, 0.9); sw = {t: [0, 0, 0, 0] for t in TH}; R = {}; PRED = []
    for c, lab in VA:
        s, e, y, dec, _ = encode(c, lab)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kl, dl = M(torch.tensor([e["input_ids"]], device=dev))
        p = torch.sigmoid(kl.float()).cpu().numpy(); t_ = y.numpy() > 0; body = np.array([a_ >= len(f"Question: {c['question']}") for a_, _ in e["offset_mapping"]])
        r = R.setdefault(c["dataset"], {"tp": 0, "fp": 0, "fn": 0, "kept": 0, "tot": 0, "frag": 0, "ans": 0, "n": 0, "dok": [0, 0, 0, 0]})
        for th in TH:
            q = (p > th) & body; w = sw[th]
            w[0] += int((q & t_).sum()); w[1] += int((q & ~t_).sum()); w[2] += int((~q & t_).sum()); w[3] += int(q.sum())
        q = (p > A.pred_thr) & body
        if A.merge_gap or A.min_run > 1:               # decode-time span shaping (no retraining)
            q = shape(q, A.merge_gap, A.min_run) & body
        r["tp"] += int((q & t_).sum()); r["fp"] += int((q & ~t_).sum()); r["fn"] += int((~q & t_).sum())
        r["kept"] += int(q.sum()); r["tot"] += int(body.sum()); r["n"] += 1
        r["frag"] += int(q[0]) + int((q[1:] & ~q[:-1]).sum())
        cm = np.zeros(len(s) + 1, dtype=bool)          # kept characters → contiguous runs of the ORIGINAL text
        for (a_, b_), k in zip(e["offset_mapping"], q):
            if k:
                cm[a_:b_] = True
        cm[1:-1] |= cm[:-2] & cm[2:]                   # bridge single gaps (the space between two kept tokens)
        kept_txt = " | ".join(s[a_:b_] for a_, b_ in runs(cm))
        r["ans"] += int(c["answer"].lower() in kept_txt.lower())
        pd = int(dl.argmax()); r["dok"][2 * dec + pd] += 1          # [S→S, S→A, A→S, A→A]
        pspan = []                                     # kept character runs per passage (for the <search> step)
        try:
            p0 = [len(f"Question: {c['question']}")]
            for pp in c["passages"]:
                p0.append(p0[-1] + len(SEP) + len(pp["text"].strip()))
            for a_, b_ in runs(cm):
                k = max(0, int(np.searchsorted(np.array(p0[:-1]) + len(SEP), a_, side="right")) - 1)
                pspan.append([k, int(a_), int(b_), s[a_:b_]])
        except Exception as ex_:
            log(f"pred dump failed: {ex_!r}")
        PRED.append({"dataset": c["dataset"], "row": c["row"], "p_answer": float(torch.softmax(dl.float(), 0)[1]),
                     "decision_gold": dec, "fragments": pspan})
    out = {}
    for name, r in R.items():
        P = r["tp"] / max(1, r["tp"] + r["fp"]); Rc = r["tp"] / max(1, r["tp"] + r["fn"]); F1 = 2 * P * Rc / max(1e-9, P + Rc)
        d = r["dok"]; accS = d[0] / max(1, d[0] + d[1]); accA = d[3] / max(1, d[2] + d[3])
        log(f"EVAL step {step} | {name} ({r['n']} q) | KEEP P {P:.3f} R {Rc:.3f} F1 {F1:.3f} | kept {r['kept'] / max(1, r['tot']):.3f} "
            f"| fragments/q {r['frag'] / r['n']:.1f} tok/fragment {r['kept'] / max(1, r['frag']):.1f} | answer in kept {r['ans'] / r['n']:.2f} "
            f"| decision: SEARCH recall {accS:.2f} ({d[0] + d[1]}) ANSWER recall {accA:.2f} ({d[2] + d[3]})")
        out[name] = dict(P=P, R=Rc, F1=F1, ans=r["ans"] / r["n"], accS=accS, accA=accA)
    with open(os.path.join(A.out, "preds_val.jsonl"), "w") as f_:
        for x in PRED:
            f_.write(json.dumps(x, ensure_ascii=False) + "\n")
    log(f"EVAL step {step} | sweep " + " | ".join(f"p>{t}: P {w[0] / max(1, w[0] + w[1]):.2f} R {w[0] / max(1, w[0] + w[2]):.2f}" for t, w in sw.items()))
    M.train(); return out


if A.eval_only:
    ck = torch.load(A.eval_only, map_location="cpu", weights_only=False)
    M.enc.load_state_dict(ck["enc"]); M.keep.load_state_dict(ck["keep"]); M.dec.load_state_dict(ck["dec"])
    if "rec" in ck:
        M.rec.load_state_dict(ck["rec"]); A.recycle = ck["args"].get("recycle", 0)
    log(f"eval only: {A.eval_only} (step {ck['step']}) | KEEP threshold {A.pred_thr}")
    evaluate(f"{ck['step']} @p>{A.pred_thr}"); raise SystemExit
if A.init:
    ck = torch.load(A.init, map_location="cpu", weights_only=False)
    M.enc.load_state_dict(ck["enc"]); M.keep.load_state_dict(ck["keep"]); M.dec.load_state_dict(ck["dec"])
    if "rec" in ck:
        M.rec.load_state_dict(ck["rec"])
    log(f"continuing from {A.init} (step {ck['step']}, splits {ck['args'].get('train_splits')})")
hist = []; t0 = time.time(); win = []
M.train(); opt.zero_grad(set_to_none=True)
total = len(TR); upd = math.ceil(total / A.accum)
for step, (c, lab) in enumerate(TR, 1):
    u = (step - 1) // A.accum
    f = min(1.0, (u + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u / upd)))
    opt.param_groups[0]["lr"] = A.lr * f; opt.param_groups[1]["lr"] = A.head_lr * f
    s, e, y, dec, hw = encode(c, lab); y = y.to(dev); hw = 1 + (A.hop_w - 1) * hw.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kl, dl, k1 = M(torch.tensor([e["input_ids"]], device=dev), both=True)
    kl, dl = kl.float(), dl.float()

    def keep_loss(kx):
        b = F.binary_cross_entropy_with_logits(kx.float(), y, reduction="none")
        return A.keep_w * (b * hw)[y > 0].sum() / hw[y > 0].sum() + (1 - A.keep_w) * b[y == 0].mean() if (y > 0).any() else b.mean()
    lk = keep_loss(kl) + (A.rec_w1 * keep_loss(k1) if k1 is not None else 0)
    ld = F.cross_entropy(dl.unsqueeze(0), torch.tensor([dec], device=dev), weight=W)
    ((lk + ld) / A.accum).backward()
    with torch.no_grad():                              # train-window KEEP quality at p>0.5 (chart: does it flatten?)
        q_ = kl.detach() > 0; t_ = y > 0
        tp_, fp_, fn_ = int((q_ & t_).sum()), int((q_ & ~t_).sum()), int((~q_ & t_).sum()); tn_ = int(len(y)) - tp_ - fp_ - fn_
    win.append((float(lk), float(ld), len(e["input_ids"]), tp_, fp_, fn_, tn_))
    if step % A.accum == 0 or step == total:
        torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step % 100 == 0 or A.smoke:
        w = np.array(win); win.clear()
        tp_, fp_, fn_, tn_ = w[:, 3].sum(), w[:, 4].sum(), w[:, 5].sum(), w[:, 6].sum()
        bal = 0.5 * (tp_ / max(1, tp_ + fn_) + tn_ / max(1, tn_ + fp_)); f1_ = 2 * tp_ / max(1, 2 * tp_ + fp_ + fn_)
        log(f"step {step}/{total} | keepL {w[:, 0].mean():.3f} decL {w[:, 1].mean():.3f} | train KEEP@0.5 bal-acc {bal:.3f} F1 {f1_:.3f} | len {w[:, 2].mean():.0f} | "
            f"{(time.time() - t0) / step:.2f}s/step | peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
    if step % A.eval_every == 0 or step == total:
        torch.save({"keep": M.keep.state_dict(), "dec": M.dec.state_dict(), "enc": M.enc.state_dict(), "rec": M.rec.state_dict(), "args": vars(A), "step": step},
                   os.path.join(A.out, "chopper.pt"))
        if not os.path.lexists(os.path.join(A.out, "reader.pt")):   # old name, for chains written before the rename
            os.symlink("chopper.pt", os.path.join(A.out, "reader.pt"))
        hist.append({"step": step, **evaluate(step)})
        json.dump(hist, open(os.path.join(A.out, "evals.json"), "w"), indent=1)
log("MM TRAIN DONE")
