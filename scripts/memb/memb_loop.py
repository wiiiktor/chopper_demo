#!/usr/bin/env python
"""Retrieval + chopper side of the memory-benchmark loop (2026-11-08), the tool_eval_init/fetch pair adapted to long histories.
Search = the tool loop's: Qwen3-Embedding dense top-100 + BM25 top-50 over the question's OWN history, bge-reranker-v2-m3, best K.
Chopper = mmBERT (mm_train.py --eval-only) or the MiniCPM3 20% slice (mla_keeper_train.py --eval-only), both at KEEP p>.5.
  embed <bench dir>                                   chunk embeddings -> emb.pt
  init  <bench dir> <state> <work> <mm|slice> <ckpt>  first 16 by the question -> chopper notes -> state (answerer = tool_eval_answer.py)
  fetch <bench dir> <state> <work> <mm|slice> <ckpt>  every pending search(query): 8 best unseen passages -> chopper cuts with the query -> tool turn"""
import json, math, os, re, subprocess, sys, time
from collections import Counter
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
MODE, BD = sys.argv[1], sys.argv[2]; dev = "cuda"; ROOT = "/home/wii/Projects/certain"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
T = json.load(open(f"{BD}/chunks.json"))
etk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
emo = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(dev).eval()
if MODE == "embed":
    out = []
    order = sorted(range(len(T)), key=lambda i: len(T[i]))
    with torch.no_grad():
        for b in range(0, len(T), 64):
            ix = order[b:b + 64]; e = etk([T[i] for i in ix], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
            out.append((ix, F.normalize(emo(**e).last_hidden_state[:, -1].float(), dim=-1).half().cpu()))
            if b % 6400 == 0:
                log(f"embed {b}/{len(T)}")
    E = torch.zeros(len(T), out[0][1].shape[1], dtype=torch.float16)
    for ix, e in out:
        E[torch.tensor(ix)] = e
    torch.save(E, f"{BD}/emb.pt"); log("EMBED DONE"); raise SystemExit
ST, WD, KIND, CK = (sys.argv[3:7] + ["", "", ""])[:4]; WD and os.makedirs(WD, exist_ok=True)
E = torch.load(f"{BD}/emb.pt").to(dev).float(); H = json.load(open(f"{BD}/hay.json"))
rtk = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rmo = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.bfloat16).to(dev).eval()
TASK = "Given a question, retrieve passages that contain facts needed to answer it"
W = re.compile(r"\w+", re.U)
STOP = set("the a an of in on at to for and or is was were are be by with as from that this which who what when where how did does do i my me you your".split())
toks = lambda s: [w for w in W.findall(s.lower()) if w not in STOP]
TK = {}


@torch.no_grad()
def search(hay, q, k, skip=()):
    ch = [c for c in H[hay] if c not in skip]
    if not ch:
        return []
    e = etk([f"Instruct: {TASK}\nQuery:{q}"], truncation=True, max_length=512, return_tensors="pt").to(dev)
    qe = F.normalize(emo(**e).last_hidden_state[:, -1].float(), dim=-1)[0]
    de = [ch[i] for i in (E[torch.tensor(ch, device=dev)] @ qe).argsort(descending=True).tolist()[:100]]
    docs = [TK.setdefault(c, toks(T[c])) for c in ch]; N = len(docs); avg = np.mean([len(d) for d in docs]) or 1
    dfc = Counter(w for d in docs for w in set(d)); sc = np.zeros(N)
    for i, d in enumerate(docs):
        tf = Counter(d); L = len(d)
        for w in set(toks(q)):
            if w in tf:
                sc[i] += math.log(1 + (N - dfc[w] + 0.5) / (dfc[w] + 0.5)) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * L / avg))
    cand = list(dict.fromkeys(de + [ch[i] for i in np.argsort(-sc)[:50]]))
    s = []
    for i in range(0, len(cand), 50):
        b = rtk([[q, T[c]] for c in cand[i:i + 50]], padding=True, truncation=True, max_length=640, return_tensors="pt").to(dev)
        s.append(rmo(**b).logits.view(-1).float())
    return [cand[j] for j in torch.cat(s).argsort(descending=True).tolist()[:k]]


def chop(rows, tag):
    """rows = [(query, [chunk ids])] -> per row the kept fragment texts (and the chunk each came from)"""
    d = f"{WD}/{tag}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/ctx_fetch.jsonl", "w") as f, open(f"{d}/qw_fetch.jsonl", "w") as g:
        for n, (q, cs) in enumerate(rows):
            f.write(json.dumps({"dataset": "memb", "split": "val", "row": n, "question": q, "answer": "", "groups": [], "allin": 1.0,
                                "passages": [{"pos": i, "cid": c, "text": T[c], "group": -1} for i, c in enumerate(cs)]}, ensure_ascii=False) + "\n")
            g.write(json.dumps({"dataset": "memb", "split": "val", "row": n, "spans": [], "steps": [], "miss": []}) + "\n")
    if KIND == "mm":
        cmd = [f"{ROOT}/.venv-gen/bin/python", f"{ROOT}/scripts/mm_train.py", d, f"{d}/chop", "--base", "jhu-clsp/mmBERT-base", "--train-splits", "fetch",
               "--eval-only", CK, "--val-split", "fetch", "--pred-thr", "0.5", "--merge-gap", "5", "--min-run", "4"]
    else:
        cmd = [f"{ROOT}/.venv-rwkv7/bin/python", f"{ROOT}/scripts/mla_keeper_train.py", d, "/media/wii/bigdata/mla/perms_minicpm3.pt", f"{d}/chop", "--frac", "0.2",
               "--train-splits", "fetch", "--val-split", "fetch", "--eval-only", CK, "--pred-thr", "0.5", "--baseline", "none"]
    subprocess.run(cmd, check=True, stdout=open(f"{d}/chop.out", "w"), stderr=subprocess.STDOUT)
    P = {x["row"]: x for x in map(json.loads, open(f"{d}/chop/preds_val.jsonl"))}
    return [[(" ".join(f[3].replace("<eos>", " ").split()), cs[f[0]]) for f in P[n]["fragments"]] if n in P else [] for n, (q, cs) in enumerate(rows)]


def notes(fr):
    return "\n".join(f"- {t}" for t, _ in fr if t) or "(none)"


if MODE in ("rag", "gold"):                      # paper 2610.34073 references: naive RAG = raw top-10 chunks, no chopper; gold = only the evidence chunks
    B = json.loads(open("/media/wii/bigdata/rr/tool_train.jsonl").readline())
    Q = [json.loads(l) for l in open(f"{BD}/qs.jsonl")]
    with open(ST, "w") as f:
        for q in Q:
            if MODE == "gold" and not q["ev"]:
                continue
            cs = search(q["hay"], q["question"], 10) if MODE == "rag" else sorted(q["ev"])
            ql = f"Question: {q['question']}" + (f"\n(Today is {q['date']}.)" if q["date"] else "")
            f.write(json.dumps({"key": [q["bench"], q["qid"]], "q": q, "system": B["system"], "tools": B["tools"], "seen": [],
                                "ev16": [c for c in q["ev"] if c in cs], "evkept": [c for c in q["ev"] if c in cs],
                                "messages": [{"role": "user", "content": "Notes:\n" + "\n\n".join(T[c] for c in cs) + f"\n\n{ql}"}]}, ensure_ascii=False) + "\n")
    log(f"{MODE.upper()} {os.path.basename(BD)} written")
elif MODE == "init":
    B = json.loads(open("/media/wii/bigdata/rr/tool_train.jsonl").readline())
    Q = [json.loads(l) for l in open(f"{BD}/qs.jsonl")]
    if os.environ.get("SMOKE"):
        Q = Q[:int(os.environ["SMOKE"])]
    first = [search(q["hay"], q["question"], 16) for q in Q]; log(f"first pools for {len(Q)} questions")
    FR = chop([(q["question"], f) for q, f in zip(Q, first)], "init")
    with open(ST, "w") as f:
        for q, fs, fr in zip(Q, first, FR):
            ql = f"Question: {q['question']}" + (f"\n(Today is {q['date']}.)" if q["date"] else "")
            f.write(json.dumps({"key": [q["bench"], q["qid"]], "q": q, "system": B["system"], "tools": B["tools"], "seen": [],
                                "ev16": [c for c in q["ev"] if c in fs], "evkept": sorted({c for _, c in fr if c in q["ev"]}),
                                "messages": [{"role": "user", "content": f"Notes:\n{notes(fr)}\n\n{ql}"}]}, ensure_ascii=False) + "\n")
    ev = [q for q in Q if q["ev"]]
    S = [json.loads(l) for l in open(ST)]
    log(f"INIT {os.path.basename(BD)} {KIND}: all evidence in first 16 {np.mean([len(s['ev16']) == len(s['q']['ev']) for s in S if s['q']['ev']]):.2f} | "
        f"any {np.mean([bool(s['ev16']) for s in S if s['q']['ev']]):.2f} | evidence kept by chopper (any) {np.mean([bool(s['evkept']) for s in S if s['q']['ev']]):.2f} | "
        f"notes words {np.mean([len(s['messages'][0]['content'].split()) for s in S]):.0f} (n={len(ev)} with evidence)")
elif MODE == "fetch":
    S = [json.loads(l) for l in open(ST)]; todo = [s for s in S if s.get("pending")]
    rows = []
    for s in todo:
        new = search(s["q"]["hay"], s["pending"], 8, set(s["seen"])); s["seen"] += new; rows.append((s["pending"], new))
    FR = chop(rows, f"fetch{int(time.time())}") if rows else []
    for s, fr in zip(todo, FR):
        s["evkept"] = sorted(set(s["evkept"]) | {c for _, c in fr if c in s["q"]["ev"]})
        s["messages"].append({"role": "tool", "content": notes(fr) if fr else "(no relevant passages found)"}); s["pending"] = None
    with open(ST, "w") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in S)
    log(f"fetched {len(todo)}")
sys.stdout.flush(); os._exit(0)
