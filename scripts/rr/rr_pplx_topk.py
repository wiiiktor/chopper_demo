#!/usr/bin/env python3
"""What would a top-K first filter cost in luna's YES answers? (2026-11-05) On v2 rows every non-seed passage of a row is
ranked by pplx-embed-v2-late-0.6b and by Qwen3-Reranker-0.6B. Recall is weighted by reading probability: rr-read pairs (p>0.3)
count 1, audit pairs (5% random of the rest) count 20, so the estimate covers passages below the current cutoff too.
Prints, per K, the share of estimated YES inside the top K, and the current p>0.3 rule for reference.
usage: rr_pplx_topk.py <rr dir> [lang ...] (env N=rows per language)"""
import json, os, sys
import numpy as np
from sentence_transformers import MultiVectorEncoder
D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en"]; N = int(os.environ.get("N", "400"))
m = MultiVectorEncoder("perplexity-ai/pplx-embed-v2-late-0.6b", device="cuda")
for lang in LANGS:
    L = {}
    for x in map(json.loads, open(f"{D}/luna_v2_{lang}.jsonl")):
        L[(x["rid"], x["i"])] = x
    RR = json.load(open(f"{D}/rr_v2_{lang}.json")); K = (2, 3, 4, 6, 8); tot = 0.; n = 0; reads = 0
    hit = {("pplx", k): 0. for k in K} | {("rr", k): 0. for k in K}; cut = 0.
    for r in map(json.loads, open(f"{D}/rows_v2_{lang}.jsonl")):
        if (r["rid"], r["seed"]) not in L or r["qtype"] == "no_answer":
            continue
        idx = [i for i in range(len(r["passages"])) if i != r["seed"]]
        q = m.encode_query([r["query"]]); P = m.encode_document([r["passages"][i]["text"] for i in idx])
        p = np.array([float(m.similarity(q, [d])[0][0]) for d in P]); s = np.array([RR[r["rid"]][i] for i in idx])
        op, os_ = np.argsort(-p), np.argsort(-s)
        for j, i in enumerate(idx):
            x = L.get((r["rid"], i))
            if not x or not x["v"]:
                continue
            w = 20. if x["how"] == "audit" else 1.
            tot += w; cut += w * (s[j] > .3)
            for k in K:
                hit[("pplx", k)] += w * (j in op[:k]); hit[("rr", k)] += w * (j in os_[:k])
        reads += (s > .3).sum(); n += 1
        if n >= N:
            break
    print(f"{lang}: {n} rows | est. non-seed YES per row {tot / n:.2f} | p>0.3 rule: {reads / n:.2f} reads/row, recall {cut / tot:.3f}", flush=True)
    for k in K:
        print(f"  top-{k}: recall pplx {hit[('pplx', k)] / tot:.3f} reranker {hit[('rr', k)] / tot:.3f}", flush=True)
