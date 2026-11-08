#!/usr/bin/env python3
"""Fair retest of rr_pplx_vs_rr.py: MaxSim scales with query length, so a global threshold across queries is not how a
late-interaction model is used. Within-row ranking on v2 rows: for rows with luna YES and NO among non-seed read passages,
how often does each model rank a YES above a NO (pairwise AUC per row, averaged)?   usage: rr_pplx_within.py <rr dir> [N rows]"""
import json, sys
import numpy as np
from sentence_transformers import MultiVectorEncoder
D = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
m = MultiVectorEncoder("perplexity-ai/pplx-embed-v2-late-0.6b", device="cuda")
for lang in ["pl", "en"]:
    L = {}
    for x in map(json.loads, open(f"{D}/luna_v2_{lang}.jsonl")):
        if x["how"] != "seed":
            L.setdefault(x["rid"], {})[x["i"]] = x["v"]
    RR = json.load(open(f"{D}/rr_v2_{lang}.json")); A, B, n = [], [], 0
    for r in map(json.loads, open(f"{D}/rows_v2_{lang}.jsonl")):
        v = L.get(r["rid"], {})
        if not any(v.values()) or all(v.values()):
            continue
        idx = sorted(v); y = np.array([v[i] for i in idx])
        q = m.encode_query([r["query"]]); P = m.encode_document([r["passages"][i]["text"] for i in idx])
        p = np.array([float(m.similarity(q, [d])[0][0]) for d in P]); s = np.array([RR[r["rid"]][i] for i in idx])
        pw = lambda z: np.mean([(a > b) + .5 * (a == b) for a in z[y] for b in z[~y]])
        A.append(pw(s)); B.append(pw(p)); n += 1
        if n >= N:
            break
    print(f"{lang}: {n} rows | within-row AUC reranker {np.mean(A):.3f} pplx {np.mean(B):.3f}", flush=True)
