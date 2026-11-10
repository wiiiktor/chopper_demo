#!/usr/bin/env python
"""Conversation-memory rows, stage 2 (2026-11-09): cm_gen.py histories -> mmBERT chopper rows + a held-out memory benchmark.
Each session is cut into ~110-word passages of consecutive turns with a <src>date | session k</src> tag (memb_build.chunk, the
LoCoMo / LongMemEval format). For every question: the question's own history is searched like the benchmarks (Qwen3-Embedding
dense top-100 + BM25 top-50, bge-reranker-v2-m3, best 16). KEEP spans = every planted evidence sentence + its passage's tag
(update questions keep the old AND the new sentence: the dates decide). An evidence passage the search missed replaces the
16th passage in TRAINING rows (the chopper needs the positive); held-out rows stay as retrieved. Absent questions = no-answer
rows (no spans, allin 0). Also writes the held-out histories in the memb format (chunks.json, hay.json, qs.jsonl) so
memb_loop.py / memb_judge.py can run the same arms on them.
Writes <pack>/ctx_cm.jsonl + qw_cm.jsonl (train), ctx_cmval/qw_cmval (held out), ctx_rrfull20/qw_rrfull20 (20% of rrfull),
<memb>/cm/{chunks.json,hay.json,qs.jsonl}.
usage: cm_pack.py <hist dir> <pack dir> <memb dir> [--hold 150]   (SMOKE=N histories)"""
import glob, json, math, os, random, re, sys, time
from collections import Counter
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
HD, PK, MB = sys.argv[1:4]; HOLD = int(sys.argv[sys.argv.index("--hold") + 1]) if "--hold" in sys.argv else 150
SMOKE = int(os.environ.get("SMOKE", "0")); dev = "cuda"; rng = random.Random(0)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
sys.argv = [sys.argv[0], MB]                                      # memb_build's chunker, without running its benchmark build
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memb", "memb_build.py")).read()
exec(SRC.split("def strat(")[0])                                  # defines sents(), chunk()
H = [json.loads(l) for f in sorted(glob.glob(f"{HD}/hist_*.jsonl")) if "_smoke" not in f or SMOKE for l in open(f)]
if SMOKE:
    H = H[:SMOKE]
rng.shuffle(H); hold = {h["hid"] for h in H[:HOLD if not SMOKE else max(1, len(H) // 3)]}

# passages, with where each planted sentence landed
C, HAY, PL = [], {}, {}                                           # PL[(hid, s, t)] = [chunk ids holding that turn's text]
for h in H:
    ids = []
    for s, ss in enumerate(h["sessions"]):
        turns = [("User" if t["role"] == "user" else "Assistant", t["text"], (s, i)) for i, t in enumerate(ss["turns"])]
        for tx, evs in chunk(turns, f"<src>{ss['date']} | session {s + 1}</src>\n"):
            for e in evs:
                PL.setdefault((h["hid"],) + tuple(e), []).append(len(C))
            ids.append(len(C)); C.append(tx)
    HAY[h["hid"]] = ids
log(f"{len(H)} histories ({len(hold)} held out), {len(C)} passages")

etk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
emo = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(dev).eval()
rtk = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rmo = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.bfloat16).to(dev).eval()
E = torch.zeros(len(C), 1024, dtype=torch.float16, device=dev); order = sorted(range(len(C)), key=lambda i: len(C[i]))
with torch.no_grad():
    for b in range(0, len(C), 64):
        ix = order[b:b + 64]; e = etk([C[i] for i in ix], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
        E[torch.tensor(ix, device=dev)] = F.normalize(emo(**e).last_hidden_state[:, -1].float(), dim=-1).half()
log("embedded")
TASK = "Given a question, retrieve passages that contain facts needed to answer it"
W = re.compile(r"\w+", re.U); STOP = set("the a an of in on at to for and or is was were are be by with as from that this which who what when where how did does do i my me you your".split())
toks = lambda s: [w for w in W.findall(s.lower()) if w not in STOP]; TK = {}


@torch.no_grad()
def search(hay, q, k=16):
    ch = HAY[hay]; e = etk([f"Instruct: {TASK}\nQuery:{q}"], truncation=True, max_length=512, return_tensors="pt").to(dev)
    qe = F.normalize(emo(**e).last_hidden_state[:, -1].float(), dim=-1)[0]
    de = [ch[i] for i in (E[torch.tensor(ch, device=dev)].float() @ qe).argsort(descending=True).tolist()[:60]]
    docs = [TK.setdefault(c, toks(C[c])) for c in ch]; N = len(docs); avg = np.mean([len(d) for d in docs]) or 1
    dfc = Counter(w for d in docs for w in set(d)); sc = np.zeros(N); qt = set(toks(q))
    for i, d in enumerate(docs):
        tf = Counter(d); L = len(d)
        for w in qt:
            if w in tf:
                sc[i] += math.log(1 + (N - dfc[w] + 0.5) / (dfc[w] + 0.5)) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * L / avg))
    cand = list(dict.fromkeys(de + [ch[i] for i in np.argsort(-sc)[:30]]))
    b = rtk([[q, C[c]] for c in cand], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
    s = rmo(**b).logits.view(-1).float()
    return [cand[j] for j in s.argsort(descending=True).tolist()[:k]]


W_ = lambda t: set(re.findall(r"[a-z0-9]+", t.lower())) - STOP


def owners(h):
    """planted sentence -> the question it was planted for (2026-11-10 fix: 36% of evidence turns held several planted
    sentences and every one of them was labelled for every question on that turn). New histories store q["say"]; old ones are
    assigned by word overlap with question + answer among the questions that share the turn."""
    own = {}
    for qi, q in enumerate(h["qs"]):
        for sent in q.get("say", []):
            own[sent] = qi
    for s_, t_ in {tuple(e) for q in h["qs"] for e in q["ev"]}:
        cand = [qi for qi, q in enumerate(h["qs"]) if [s_, t_] in q["ev"]]
        for sent in h["sessions"][s_]["turns"][t_].get("plant", []):
            if sent not in own and cand:
                own[sent] = max(cand, key=lambda qi: len(W_(sent) & W_(h["qs"][qi]["q"] + " " + h["qs"][qi]["a"])))
    return own


def spans(h, q, pool):
    """[[passage k, c0, c1]] for every planted evidence sentence OF THIS QUESTION found in the pool + that passage's tag"""
    out, hit = [], 0; qi = h["qs"].index(q); own = OWN[h["hid"]]
    for s, t in q["ev"]:
        mine = [x for x in h["sessions"][s]["turns"][t].get("plant", []) if own.get(x) == qi]
        if not mine:                                       # lost every overlap tie on this turn: keep its best match
            pl = h["sessions"][s]["turns"][t].get("plant", [])
            mine = [max(pl, key=lambda x: len(W_(x) & W_(q["q"] + " " + q["a"])))] if pl else []
        for sent in mine:
            for k, c in enumerate(pool):
                if c in PL.get((h["hid"], s, t), []):
                    i = C[c].find(sent)
                    if i >= 0:
                        tg = C[c].index("</src>\n") + 7
                        out += [[k, 0, tg], [k, i, i + len(sent)]]; hit += 1
    return sorted(set(map(tuple, out))), hit


os.makedirs(PK, exist_ok=True); os.makedirs(f"{MB}/cm", exist_ok=True)
OWN = {h["hid"]: owners(h) for h in H}
st = Counter(); t0 = time.time(); n = 0
sfx = "_smoke" if SMOKE else ""
F_ = {sp: (open(f"{PK}/ctx_{sp}{sfx}.jsonl", "w"), open(f"{PK}/qw_{sp}{sfx}.jsonl", "w")) for sp in ("cm", "cmval")}
QV = open(f"{MB}/cm/qs{sfx}.jsonl", "w")
for h in H:
    sp = "cmval" if h["hid"] in hold else "cm"
    for q in h["qs"]:
        qq = q["q"]; pool = search(h["hid"], qq)
        evc = [c for s, t in q["ev"] for c in PL.get((h["hid"], s, t), [])]
        st[f"{sp} evidence in 16"] += bool(evc) and any(c in pool for c in evc); st[f"{sp} with evidence"] += bool(evc)
        miss = [c for c in dict.fromkeys(evc) if c not in pool]
        if sp == "cm" and miss:                                # training rows: missed evidence passages go in at random places
            pool = [c for c in pool][:16 - len(miss)]
            for c in miss:
                pool.insert(rng.randrange(len(pool) + 1), c)
        sps, hit = spans(h, q, pool)
        if q["kind"] != "absent" and not hit and sp == "cm":
            st["dropped: evidence not placeable"] += 1; continue
        row = {"dataset": "cm", "split": sp, "row": n, "question": qq, "answer": q["a"], "groups": [], "allin": 0.0 if q["kind"] == "absent" else 1.0,
               "qtype": q["kind"], "passages": [{"pos": k, "cid": c, "text": C[c], "group": -1} for k, c in enumerate(pool)]}
        F_[sp][0].write(json.dumps(row, ensure_ascii=False) + "\n")
        F_[sp][1].write(json.dumps({"dataset": "cm", "split": sp, "row": n, "spans": [list(x) for x in sps], "steps": [], "miss": []}) + "\n")
        st[f"{sp} rows"] += 1; st[f"{sp} {q['kind']}"] += 1; n += 1
        if sp == "cmval":
            QV.write(json.dumps({"qid": f"{h['hid']}_{n}", "bench": "cm", "cat": q["kind"], "question": qq, "gold": q["a"], "date": q["qdate"],
                                 "hay": h["hid"], "ev": sorted(set(evc))}, ensure_ascii=False) + "\n")
    if n and n % 2000 < len(h["qs"]):
        log(f"{n} rows | {(time.time() - t0) / n:.2f}s/row")
for a, b in F_.values():
    a.close(); b.close()
QV.close()
json.dump(C, open(f"{MB}/cm/chunks{sfx}.json", "w"), ensure_ascii=False); json.dump({k: v for k, v in HAY.items() if k in hold}, open(f"{MB}/cm/hay{sfx}.json", "w"))
if not SMOKE and not os.path.exists(f"{PK}/ctx_rrfull20.jsonl"):   # 20% of the current recipe's rows, so the new rows do not wipe the rerank skill
    R = random.Random(1); keep = {}
    with open(f"{PK}/ctx_rrfull20.jsonl", "w") as f:
        for l in open(f"{PK}/ctx_rrfull.jsonl"):
            if R.random() < 0.2:
                c = json.loads(l); keep[(c["dataset"], c["row"])] = 1; f.write(l)
    with open(f"{PK}/qw_rrfull20.jsonl", "w") as f:
        for l in open(f"{PK}/qw_rrfull.jsonl"):
            d = json.loads(l)
            if (d["dataset"], d["row"]) in keep:
                f.write(l)
    st["rrfull20 rows"] = len(keep)
log("PACK DONE " + " | ".join(f"{k} {v}" for k, v in sorted(st.items())))
