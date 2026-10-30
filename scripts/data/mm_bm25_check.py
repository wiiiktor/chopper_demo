#!/usr/bin/env python
"""BM25 vs dense vs hybrid on the mm val pools (2026-10-29, user: "bm25?"): all-hops@k and per-hop recall@k, BM25 computed
per pool (Okapi, k1 1.5, b 0.75, idf over the pool's 1,000 chunks), dense = the saved Qwen3-Embedding vectors, hybrid =
reciprocal-rank fusion (60). usage: .venv-gen/bin/python mm_bm25_check.py <mm dir>"""
import sys, json, re, math
from collections import Counter
import numpy as np, torch
D_ = sys.argv[1]
T = json.load(open(f"{D_}/chunks_val.json")); P = [json.loads(l) for l in open(f"{D_}/pools_val.jsonl")]
E = torch.load(f"{D_}/emb_val.pt").float()
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
etk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
emo = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).cuda().eval()
TASK = "Given a question, retrieve passages that contain facts needed to answer it"
with torch.no_grad():
    b = etk([f"Instruct: {TASK}\nQuery:{p['question']}" for p in P], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    Q = F.normalize(emo(**b).last_hidden_state[:, -1].float(), dim=-1).cpu()
W = re.compile(r"\w+", re.U)
STOP = set("the a an of in on at to for and or is was were are be by with as from that this which who what when where how did does do".split())
toks = lambda s: [w for w in W.findall(s.lower()) if w not in STOP]
KS = (8, 16, 32)
res = {}
for pi, p in enumerate(P):
    ch = p["chunks"]; docs = [toks(T[c]) for c in ch]; N = len(docs); avg = np.mean([len(d) for d in docs])
    dfc = Counter(w for d in docs for w in set(d)); q = toks(p["question"])
    sc = np.zeros(N)
    for i, d in enumerate(docs):
        tf = Counter(d); L = len(d)
        for w in set(q):
            if w in tf:
                idf = math.log(1 + (N - dfc[w] + 0.5) / (dfc[w] + 0.5))
                sc[i] += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (1 - 0.75 + 0.75 * L / avg))
    bm = list(np.argsort(-sc)); de = list((E[torch.tensor(ch)] @ Q[pi]).argsort(descending=True).numpy())
    rb = {i: r for r, i in enumerate(bm)}; rd = {i: r for r, i in enumerate(de)}
    hy = sorted(range(N), key=lambda i: -(1 / (60 + rb[i]) + 1 / (60 + rd[i])))
    G = [g for g in p["groups"] if g]
    for meth, rk in (("bm25", bm), ("dense", de), ("hybrid", hy)):
        pos = {int(i): r for r, i in enumerate(rk)}; first = [min(pos[x] for x in g) for g in G]
        r_ = res.setdefault((p["dataset"], meth), {k: [] for k in KS}); h2 = res.setdefault((p["dataset"], meth, "hop2"), {k: [] for k in KS})
        for k in KS:
            r_[k].append(all(f < k for f in first))
            if len(first) > 1:
                h2[k].append(first[1] < k)
for key, v in sorted(res.items()):
    if any(len(x) for x in v.values()):
        print(" ".join(map(str, key)).ljust(26), " ".join(f"@{k} {np.mean(x):.2f}" for k, x in v.items() if x))
