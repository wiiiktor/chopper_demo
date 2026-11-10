#!/usr/bin/env python
"""Multi-hop chopper v2 data (user 2026-11-10, points 1b + 3):
 1b  labels = UNION of the Qwen3-14B teacher's quotes (mm_qwen_labels.py; it writes the reasoning chain, so it quotes bridging clauses)
     and the answer-string hop sentences, plus the TITLE of every paragraph that holds a label -> ctx_mhtrain2 / qw_mhtrain2.
 3   the same questions in RETRIEVED form: 60 candidates (own 20 MuSiQue paragraphs + 40 from other questions), our search
     (Qwen3-Embedding dense + BM25 + bge-reranker) picks 16 in rank order; gold hop paragraphs the search missed are inserted at random
     places (training rows); labels carried by paragraph text -> ctx_mhret / qw_mhret.
usage: mh_v2_pack.py <pack dir> <teacher labels jsonl>"""
import json, math, random, re, sys, time
from collections import Counter
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
PK, TL = sys.argv[1:3]; rng = random.Random(0); dev = "cuda"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
C = [json.loads(l) for l in open(f"{PK}/ctx_mhtrain.jsonl")]
Q = {q["row"]: q for q in map(json.loads, open(f"{PK}/qw_mhtrain.jsonl"))}
T = {x["row"]: x for x in map(json.loads, open(TL))}
st = Counter()
with open(f"{PK}/ctx_mhtrain2.jsonl", "w") as fc, open(f"{PK}/qw_mhtrain2.jsonl", "w") as fq:
    for c in C:
        sp = {tuple(x) for x in Q[c["row"]]["spans"]} | {tuple(x) for x in T.get(c["row"], {}).get("spans", [])}
        for k in {k for k, a, b in sp}:
            sp.add((k, 0, len(c["passages"][k]["text"].split("\n", 1)[0])))
        st["teacher spans added"] += len(sp) - len(Q[c["row"]]["spans"])
        c2 = dict(c, split="mhtrain2"); fc.write(json.dumps(c2, ensure_ascii=False) + "\n")
        fq.write(json.dumps({"dataset": c["dataset"], "split": "mhtrain2", "row": c["row"], "spans": sorted(map(list, sp)), "steps": [], "miss": []}) + "\n")
log("labels merged", dict(st))

etk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
emo = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(dev).eval()
rtk = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rmo = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.bfloat16).to(dev).eval()
TASK = "Given a question, retrieve passages that contain facts needed to answer it"
W = re.compile(r"\w+", re.U); STOP = set("the a an of in on at to for and or is was were are be by with as from that this which who what when where how did does do".split())
toks = lambda s: [w for w in W.findall(s.lower()) if w not in STOP]
L2 = {q["row"]: q for q in map(json.loads, open(f"{PK}/qw_mhtrain2.jsonl"))}
ALLP = [p["text"] for c in C for p in c["passages"]]


@torch.no_grad()
def emb(texts, q=False):
    out = []
    for i in range(0, len(texts), 32):
        b = [f"Instruct: {TASK}\nQuery:{t}" if q else t for t in texts[i:i + 32]]
        e = etk(b, truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
        out.append(F.normalize(emo(**e).last_hidden_state[:, -1].float(), dim=-1))
    return torch.cat(out)


t0 = time.time()
with open(f"{PK}/ctx_mhret.jsonl", "w") as fc, open(f"{PK}/qw_mhret.jsonl", "w") as fq, torch.no_grad():
    for n, c in enumerate(C):
        own = [p["text"] for p in c["passages"]]; cand = own + [ALLP[i] for i in rng.sample(range(len(ALLP)), 40)]
        cand = list(dict.fromkeys(cand)); E = emb(cand); qe = emb([c["question"]], q=True)[0]
        de = (E @ qe).argsort(descending=True).tolist()[:30]
        dc = [toks(t) for t in cand]; N_ = len(dc); avg = np.mean([len(d) for d in dc]); dfc = Counter(w for d in dc for w in set(d)); sc = np.zeros(N_)
        for i, d in enumerate(dc):
            tf = Counter(d)
            for w in set(toks(c["question"])):
                if w in tf:
                    sc[i] += math.log(1 + (N_ - dfc[w] + 0.5) / (dfc[w] + 0.5)) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(d) / avg))
        pool = list(dict.fromkeys(de + list(np.argsort(-sc)[:20])))
        b = rtk([[c["question"], cand[i]] for i in pool], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
        top = [pool[j] for j in rmo(**b).logits.view(-1).float().argsort(descending=True).tolist()[:16]]
        gold = [own.index(c["passages"][g[0]]["text"]) for g in c["groups"]]
        st["gold hops retrieved"] += sum(g in top for g in gold); st["gold hops"] += len(gold)
        for g in gold:
            if g not in top:
                top = [i for i in top if i not in gold or i == g][:15]; top.insert(rng.randrange(len(top) + 1), g)
        txt = [cand[i] for i in top]; byt = {t: k for k, t in enumerate(txt)}
        sp = [[byt[c["passages"][k]["text"]], a, b2] for k, a, b2 in L2[c["row"]]["spans"] if c["passages"][k]["text"] in byt]
        fc.write(json.dumps({"dataset": c["dataset"], "split": "mhret", "row": c["row"], "question": c["question"], "answer": c["answer"], "groups": [],
                             "allin": 1.0, "passages": [{"pos": k, "cid": -1, "text": t, "group": -1} for k, t in enumerate(txt)]}, ensure_ascii=False) + "\n")
        fq.write(json.dumps({"dataset": c["dataset"], "split": "mhret", "row": c["row"], "spans": sp, "steps": [], "miss": []}) + "\n")
        if n % 2000 == 0:
            log(f"{n}/{len(C)} | {(time.time() - t0) / (n + 1):.2f}s/row")
log("RETPACK DONE", dict(st), f"retrieval kept {st['gold hops retrieved'] / max(1, st['gold hops']):.2f} of gold hop paragraphs")
