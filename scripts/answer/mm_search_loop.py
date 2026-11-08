#!/usr/bin/env python
"""One <search> step after the mmBERT chopper (2026-10-29, user: "decisions search / answer"; "bm25?"). For every val
question: query 2 = question + the KEPT FRAGMENTS (chopper predictions, or Qwen's fragments as the teacher upper bound, or
the top-1 passage as the plain-RAG 2-hop baseline) → dense (Qwen3-Embedding) top-100 ∪ BM25 top-50 over the same pool →
bge-reranker on query 2 → new top-16. Reports all-hops (every gold group present) for: the first 16; first 16 ∪ new 16;
and a 16-passage budget (first 8 + new 8), overall and on the questions the chopper sent to SEARCH.
usage: .venv-gen/bin/python mm_search_loop.py <mm dir> <chopper preds_val.jsonl> [--K 16]"""
import sys, json, re, math, time
from collections import Counter
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
D_, PR = sys.argv[1], sys.argv[2]
K = int(sys.argv[sys.argv.index("--K") + 1]) if "--K" in sys.argv else 16
dev = "cuda"; log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
T = json.load(open(f"{D_}/chunks_val.json")); E = torch.load(f"{D_}/emb_val.pt").to(dev).float()
POOL = {(p["dataset"], p["row"]): p for p in map(json.loads, open(f"{D_}/pools_val.jsonl"))}
CTX = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{D_}/ctx_val.jsonl"))}
QW = {(q["dataset"], q["row"]): q for q in map(json.loads, open(f"{D_}/qw_val.jsonl"))}
PRD = {(q["dataset"], q["row"]): q for q in map(json.loads, open(PR))}
SUBQ = {}                                              # --subq <mm_subq.py output>: written hop-2 sub-questions
if "--subq" in sys.argv:
    SUBQ = {(q["dataset"], q["row"]): q for q in map(json.loads, open(sys.argv[sys.argv.index("--subq") + 1]))}
etk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
emo = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(dev).eval()
rtk = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rmo = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.bfloat16).to(dev).eval()
TASK = "Given a question, retrieve passages that contain facts needed to answer it"
W = re.compile(r"\w+", re.U)
STOP = set("the a an of in on at to for and or is was were are be by with as from that this which who what when where how did does do".split())
toks = lambda s: [w for w in W.findall(s.lower()) if w not in STOP]


@torch.no_grad()
def qemb(q):
    b = etk([f"Instruct: {TASK}\nQuery:{q}"], truncation=True, max_length=512, return_tensors="pt").to(dev)
    return F.normalize(emo(**b).last_hidden_state[:, -1].float(), dim=-1)[0]


@torch.no_grad()
def rerank(q, docs, bs=50):
    s = []
    for i in range(0, len(docs), bs):
        b = rtk([[q, d] for d in docs[i:i + bs]], padding=True, truncation=True, max_length=640, return_tensors="pt").to(dev)
        s.append(rmo(**b).logits.view(-1).float())
    return torch.cat(s).cpu()


def bm25(q, docs):
    N = len(docs); avg = np.mean([len(d) for d in docs]); dfc = Counter(w for d in docs for w in set(d)); sc = np.zeros(N)
    for i, d in enumerate(docs):
        tf = Counter(d); L = len(d)
        for w in set(toks(q)):
            if w in tf:
                sc[i] += math.log(1 + (N - dfc[w] + 0.5) / (dfc[w] + 0.5)) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * L / avg))
    return np.argsort(-sc)


def search(p, q2):
    ch = p["chunks"]; de = (E[torch.tensor(ch, device=dev)] @ qemb(q2)).argsort(descending=True).tolist()[:100]
    bm = [int(i) for i in bm25(q2, [toks(T[c]) for c in ch])[:50]]
    cand = list(dict.fromkeys(de + bm)); rs = rerank(q2, [T[ch[i]] for i in cand])
    return [cand[k] for k in rs.argsort(descending=True).tolist()][:K]


def allhops(p, S):
    G = [g for g in p["groups"] if g]
    return float(all(any(i in S for i in g) for g in G)) if G else float("nan")


DUMP = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else ""   # 2026-11-08: write the chopper's 8+8 pools for the end-to-end test
DF = open(DUMP, "w") if DUMP else None
DSRC = sys.argv[sys.argv.index("--dump-src") + 1] if "--dump-src" in sys.argv else "chopper"   # e.g. subq-chopper
R = {}; t0 = time.time()
for n, (key, c) in enumerate(CTX.items()):
    p = POOL[key]; first = [x["pos"] for x in c["passages"]]
    pr = PRD.get(key); qw = QW.get(key)
    frag_r = " ".join(f[3] for f in (pr["fragments"] if pr else []))[:1500]
    frag_q = " ".join(c["passages"][k]["text"][a:b] for k, a, b in (qw["spans"] if qw else []))[:1500]
    top1 = c["passages"][0]["text"][:1500]
    hop1 = " ".join(x["text"] for x in c["passages"] if x["group"] == 0)[:1500]   # oracle: the gold hop-1 passage
    # (fair ceiling: Qwen saw the gold ANSWER when quoting, so "qwen" can carry answer knowledge into query 2)
    row = {"first16": allhops(p, set(first)), "search": pr is not None and pr["p_answer"] < 0.5}
    sq = dict(SUBQ.get(key, {}))
    if "reader" in sq:                                 # files written before the reader→chopper rename
        sq.setdefault("chopper", sq.pop("reader"))
    SRC = [("chopper", frag_r), ("qwen", frag_q), ("top1", top1), ("hop1gold", hop1)]
    SRC += [(f"subq-{k}", sq[k]) for k in ("hop1gold", "chopper") if k in sq]  # "" = answered, no search
    for name, frag in SRC:
        if name.startswith("subq") and not frag:       # the answerer did not ask to search: keep the first 16
            row[f"{name} 16∪16"] = row[f"{name} 8+8"] = row["first16"]; continue
        new = search(p, frag if name.startswith("subq") else c["question"] + " " + frag)
        row[f"{name} 16∪16"] = allhops(p, set(first) | set(new))
        row[f"{name} 8+8"] = allhops(p, set(first[:8]) | set([i for i in new if i not in first[:8]][:8]))
        if DF and name == DSRC:
            S = first[:8] + [i for i in new if i not in first[:8]][:8]
            grp = {i: gi for gi, g in enumerate(p["groups"]) for i in g}
            DF.write(json.dumps({**{k: c[k] for k in ("dataset", "split", "row", "question", "answer")}, "groups": p["groups"], "allin": row[f"{name} 8+8"],
                                 "passages": [{"pos": i, "cid": p["chunks"][i], "text": T[p["chunks"][i]], "group": grp.get(i, -1)} for i in S]}, ensure_ascii=False) + "\n")
    R.setdefault(c["dataset"], []).append(row)
    if (n + 1) % 50 == 0:
        log(f"{n + 1}/{len(CTX)} | {(time.time() - t0) / (n + 1):.2f}s/q")
for ds, rows in R.items():
    for sub, rr in (("all", rows), ("chopper→SEARCH", [r for r in rows if r["search"]])):
        if not rr:
            continue
        m = lambda k: np.nanmean([r.get(k, float("nan")) for r in rr])
        log(f"SEARCH {ds:8s} {sub:13s} n={len(rr):3d} | first16 {m('first16'):.2f} | "
            + " | ".join(f"{nm}: 16∪16 {m(nm + ' 16∪16'):.2f} 8+8 {m(nm + ' 8+8'):.2f}" for nm in ("chopper", "qwen", "top1", "hop1gold", "subq-hop1gold", "subq-chopper") if any(nm + " 8+8" in r for r in rr)))
log("SEARCH LOOP DONE")
