#!/usr/bin/env python3
"""Can a sharper first filter cut the luna bill? (2026-11-05, user: "i could save 1400 usd using perplexity?") Scores the
pairs luna already judged with pplx-embed-v2-late-0.6b (MaxSim) and compares ranking against Qwen3-Reranker-0.6B:
luna_cost_1k.jsonl (1,000 pairs, 8 languages, all with reranker p > 0.3; seed passages excluded) and luna_gold_pl.jsonl
(160 Polish pairs stratified over the whole reranker range). Prints AUC and, at fixed recall of luna's YES, the share of
pairs a filter would still send to luna.   usage: rr_pplx_vs_rr.py <rr dir>"""
import json, sys
import numpy as np
from sentence_transformers import MultiVectorEncoder
D = sys.argv[1]
m = MultiVectorEncoder("perplexity-ai/pplx-embed-v2-late-0.6b", device="cuda")
ROWS = {}
for l in ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]:
    for r in map(json.loads, open(f"{D}/rows_{l}.jsonl")):
        ROWS[r["rid"]] = r


def score(X):
    out = []
    for i in range(0, len(X), 16):
        b = X[i:i + 16]
        Q = m.encode_query([ROWS[x["rid"]]["query"] for x in b]); P = m.encode_document([ROWS[x["rid"]]["passages"][x["i"]]["text"] for x in b])
        out += [float(m.similarity([q], [p])[0][0]) for q, p in zip(Q, P)]
    return np.array(out)


def auc(s, y):
    r = s.argsort().argsort() + 1; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sent_at(s, y, rec):
    return (s >= np.sort(s[y])[int((1 - rec) * y.sum())]).mean()


for name, X in (("1k pairs, p>0.3, non-seed", [x for x in map(json.loads, open(f"{D}/luna_cost_1k.jsonl")) if x["luna"] is not None and not x["seed"]]),
                ("160 pl, stratified", [x for x in map(json.loads, open(f"{D}/luna_gold_pl.jsonl")) if x["luna"] is not None])):
    y = np.array([x["luna"] for x in X]); rr = np.array([x["rr"] for x in X]); pp = score(X)
    print(f"{name}: n {len(X)} YES {y.mean():.3f} | AUC reranker {auc(rr, y):.3f} pplx {auc(pp, y):.3f}", flush=True)
    for rec in (.9, .8):
        print(f"  pairs still sent to luna at recall {rec}: reranker {sent_at(rr, y, rec):.3f} pplx {sent_at(pp, y, rec):.3f}", flush=True)
