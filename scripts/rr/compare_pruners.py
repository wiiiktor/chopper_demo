#!/usr/bin/env python3
"""Chopper vs context pruners (user 2026-11-15: "what are our results against Provence and LLMLingua-2?").

Same holdout as chopper_as_reranker.py: rrval_notag, every STRIDE-th row (stride 5 = 400 queries, 8 languages, 16 passages
each), gold = character spans of luna-accepted answer sentences. Every method gives a CHARACTER MASK of what it keeps per
passage (and a relevance score when it has one), so all are scored the same way:
  - pruning:   gold recall (kept gold chars / gold chars), precision (gold chars / kept chars), F1, kept share of the input,
               full-coverage rate (rows with recall >= 0.9)
  - reranking: nDCG@10, Hit@1 from the relevance score (chopper, XProvence, Provence)
Methods
  chopper        released universal mmBERT-base (`Chopper.run`, all 16 passages in one pass, default threshold)
  xprovence@t    naver/xprovence-reranker-bgem3-v2 (568M, bge-reranker-v2-m3 base, 16 languages, CC BY-NC-ND), one passage at
                 a time, sentence-level, threshold 0.1 (Provence default), 0.3 (their "no drop") and 0.7 (their "high compression"); always_select_title
                 =False because these passages have no title sentence. Kept sentences are mapped back by spaCy xx_sent_ud_sm
                 (the splitter XProvence itself uses).
  provence@0.1   naver/provence-reranker-debertav3-v1 (English only, 512 tokens): scored on English rows only (by_lang 'en').
  llmlingua2@r   microsoft/llmlingua-2-xlm-roberta-large-meetingbank, QUESTION-AGNOSTIC token dropping, rate 0.33 (its usual
                 setting) and rate = the chopper's measured kept share; kept words are aligned back to the passage in order.
Runs in .venv-prune (transformers 4.57: XProvence's remote code predates transformers 5).

usage: compare_pruners.py <out.json> [stride] [limit] [methods comma list]
"""
import json, math, re, sys, time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "release/integrations/chopper-rag")
from chopper_rag.core import Chopper  # noqa: E402

PACK = "/media/wii/bigdata/rr/pack"
OUT = sys.argv[1]
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
METHODS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["chopper", "xprovence", "provence", "llmlingua2"]
dev = "cuda"

C = [json.loads(l) for l in open(f"{PACK}/ctx_rrval_notag.jsonl")][::STRIDE]
Q = {(d["dataset"], d["row"]): d for d in map(json.loads, open(f"{PACK}/qw_rrval_notag.jsonl"))}
if LIMIT:
    C = C[:LIMIT]
rows = []
for c in C:
    P = [p["text"] for p in c["passages"]]
    gold = [np.zeros(len(p), bool) for p in P]
    for k, a, b in Q[(c["dataset"], c["row"])]["spans"]:
        gold[k][a:b] = True
    rows.append({"q": c["question"], "P": P, "gold": gold, "lang": c["dataset"].split("_")[-1]})
print(f"{len(rows)} rows", flush=True)
masks, rel, secs = {}, {}, {}


def sync():
    torch.cuda.synchronize()


def locate_in_order(passage, pieces):
    """mark each piece (sentence or word) at its next occurrence after the previous one"""
    m = np.zeros(len(passage), bool); cur = 0
    for s in pieces:
        s = s.strip()
        if not s:
            continue
        i = passage.find(s, cur)
        if i < 0:
            i = passage.find(s)
        if i >= 0:
            m[i:i + len(s)] = True; cur = i + len(s)
    return m


# ------------------------------------------------------------------ chopper
if "chopper" in METHODS:
    ch = Chopper.from_pretrained(variant="base", device=dev)
    for name, snap in (("chopper", "none"), ("chopper_sent", "sentence")):   # raw (published numbers) + package default since 0.1.7
        ch.snap = snap
        M, R = [], []; sync(); t = time.time()
        for r in rows:
            out = ch.run(r["q"], r["P"])
            mk = [np.zeros(len(p), bool) for p in r["P"]]
            for c in out.chops:
                mk[c.passage_index][c.start:c.end] = True
            M.append(mk); R.append(list(out.passage_scores))
        sync(); secs[name] = (time.time() - t) / len(rows)
        masks[name], rel[name] = M, R
    del ch; torch.cuda.empty_cache(); print("chopper done", flush=True)

nlp = None
if "xprovence" in METHODS or "provence" in METHODS:
    import spacy
    nlp = spacy.load("xx_sent_ud_sm")


def provence_like(name, thr, only_en=False):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name, trust_remote_code=True).to(dev).eval()
    M, R = [], []; sync(); t = time.time(); n = 0
    for r in rows:
        if only_en and r["lang"] != "en":
            M.append(None); R.append(None); continue
        with torch.no_grad():
            o = m.process([r["q"]], [r["P"]], threshold=thr, always_select_title=False, enable_warnings=False)
        pc = o["pruned_context"][0]; sc = o["reranking_score"][0]
        mk = []
        for p, kept in zip(r["P"], pc):
            sents = [s.text for s in nlp(p).sents if s.text.strip() and s.text.strip() in kept]
            mk.append(locate_in_order(p, sents))
        M.append(mk); R.append([float(x) for x in sc]); n += 1
    sync()
    del m; torch.cuda.empty_cache()
    return M, R, (time.time() - t) / max(n, 1)


if "xprovence" in METHODS:
    for thr in (0.1, 0.3, 0.7):
        k = f"xprovence@{thr}"
        masks[k], rel[k], secs[k] = provence_like("naver/xprovence-reranker-bgem3-v2", thr)
        print(k, "done", flush=True)
if "provence" in METHODS:
    k = "provence@0.1"
    masks[k], rel[k], secs[k] = provence_like("naver/provence-reranker-debertav3-v1", 0.1, only_en=True)
    print(k, "done", flush=True)

# ------------------------------------------------------------------ LLMLingua-2 (question-agnostic)
if "llmlingua2" in METHODS:
    from llmlingua import PromptCompressor
    lc = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map=dev)
    share = None
    if "chopper" in masks:
        kept = sum(m.sum() for mk in masks["chopper"] for m in mk); tot = sum(len(p) for r in rows for p in r["P"])
        share = max(0.02, float(kept / tot))
    for rate in [0.33] + ([round(share, 3)] if share else []):
        k = f"llmlingua2@{rate}"; M = []; sync(); t = time.time()
        for r in rows:
            mk = []
            for p in r["P"]:
                o = lc.compress_prompt(p, rate=rate, force_tokens=["\n"], drop_consecutive=True)
                mk.append(locate_in_order(p, re.findall(r"\S+", o["compressed_prompt"])))
            M.append(mk)
        sync(); secs[k] = (time.time() - t) / len(rows); masks[k] = M
        print(k, "done", flush=True)


# ------------------------------------------------------------------ metrics
def prune_metrics(M, idx):
    g = k = gk = tot = 0; full = []
    for i in idx:
        r, mk = rows[i], M[i]
        rg = rk = 0
        for gm, m in zip(r["gold"], mk):
            rg += gm.sum(); rk += (gm & m).sum(); k += m.sum(); tot += len(m)
        g += rg; gk += rk
        if rg:
            full.append(rk / rg >= 0.9)
    rec = gk / max(g, 1); prec = gk / max(k, 1)
    return {"recall": round(float(rec), 4), "precision": round(float(prec), 4),
            "F1": round(float(2 * rec * prec / max(rec + prec, 1e-9)), 4), "kept_share": round(float(k / max(tot, 1)), 4),
            "full_coverage": round(float(np.mean(full)), 4) if full else None, "n": len(idx)}


def rank_metrics(S, idx):
    nd, h1 = [], []
    for i in idx:
        gp = [j for j, gm in enumerate(rows[i]["gold"]) if gm.any()]
        if not gp:
            continue
        order = list(np.argsort(-np.array(S[i]), kind="stable"))
        ranks = [order.index(j) + 1 for j in gp]
        dcg = sum(1 / math.log2(x + 1) for x in ranks if x <= 10)
        idcg = sum(1 / math.log2(x + 2) for x in range(min(len(gp), 10)))
        nd.append(dcg / idcg); h1.append(min(ranks) == 1)
    return {"nDCG@10": round(float(np.mean(nd)), 4), "Hit@1": round(float(np.mean(h1)), 4)} if nd else None


langs = sorted({r["lang"] for r in rows})
res = {"overall": {}, "by_lang": defaultdict(dict), "sec_per_row": secs}
for k, M in masks.items():
    idx = [i for i in range(len(rows)) if M[i] is not None]
    res["overall"][k] = prune_metrics(M, idx) | ({"rank": rank_metrics(rel[k], idx)} if k in rel else {})
    for L in langs:
        li = [i for i in idx if rows[i]["lang"] == L]
        if li:
            res["by_lang"][L][k] = prune_metrics(M, li)["F1"]
# the English-only Provence row is compared with the others on the same English rows
en = [i for i in range(len(rows)) if rows[i]["lang"] == "en"]
res["english_rows"] = {k: prune_metrics(M, en) | ({"rank": rank_metrics(rel[k], en)} if k in rel else {})
                       for k, M in masks.items() if en}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
