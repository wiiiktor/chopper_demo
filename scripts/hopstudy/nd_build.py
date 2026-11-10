#!/usr/bin/env python3
"""New multi-hop chopper data (user 2026-11-10: "i approve all 1 2 3 4 datasets"), datasets 1 + 2, with HUMAN gold sentence labels:
  hpq  HotpotQA distractor train (CC BY-SA 4.0): 10 paragraphs, supporting_facts = (title, sentence id); questions already in our
       54k rows are skipped.
  w2   2WikiMultiHopQA train (Apache-2.0): 10 paragraphs, supporting facts; comparison / bridge-comparison / compositional /
       inference types, stratified.
  mqu  MuSiQue full train, UNANSWERABLE twins (CC BY 4.0): a needed paragraph was removed -> no spans, allin 0 (the chopper should
       keep nothing and let the answerer search).
Labels = every supporting sentence + its paragraph title (the multi-hop rule: every fact the answerer needs, self-contained).
Also writes a 2WikiMultiHopQA DEV hop study (<hs2 dir>: ctx_hs/qw_hs/rows.json, 100 per type) to test generalisation beyond MuSiQue.
usage: nd_build.py <newdata dir> <pack dir> <hs2 dir> [--n-hpq 15000] [--n-w2 16000] [--n-mqu 6000]"""
import ast, json, os, random, sys
import pyarrow.parquet as pq
ND, PK, HS2 = sys.argv[1:4]; os.makedirs(HS2, exist_ok=True)
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
NH, NW, NM = arg("--n-hpq", 15000), arg("--n-w2", 16000), arg("--n-mqu", 6000)
rng = random.Random(0)
USED = set()
for f in ("train", "train2", "train3"):
    USED |= {json.loads(l)["question"].strip() for l in open(f"/media/wii/bigdata/mm/ctx_{f}.jsonl")}
P = lambda x: x if isinstance(x, (dict, list)) else (json.loads(x) if x.strip().startswith(("[", "{\"")) else ast.literal_eval(x))


def row(ds, n, q, a, paras, sup, allin=1.0):
    """paras = [(title, [sentences])]; sup = {(title, sent_id)} -> ctx + qw rows"""
    ps, spans, groups = [], [], []
    for k, (t, ss) in enumerate(paras):
        text = t + "\n" + " ".join(s.strip() for s in ss); ps.append({"pos": k, "cid": -1, "text": text, "group": -1})
        pos = len(t) + 1; hit = False
        for i, s in enumerate(ss):
            s2 = s.strip()
            if (t, i) in sup and s2:
                j = text.find(s2, pos)
                if j >= 0:
                    spans.append([k, j, j + len(s2)]); hit = True
            pos += len(s2) + 1
        if hit:
            spans.append([k, 0, len(t)]); groups.append([k]); ps[k]["group"] = len(groups) - 1
    c = {"dataset": ds, "split": ds, "row": n, "question": q, "answer": a, "groups": groups, "allin": allin, "passages": ps}
    return c, {"dataset": ds, "split": ds, "row": n, "spans": spans, "steps": [], "miss": []}


def write(name, rows):
    with open(f"{PK}/ctx_{name}.jsonl", "w") as fc, open(f"{PK}/qw_{name}.jsonl", "w") as fq:
        for c, q in rows:
            fc.write(json.dumps(c, ensure_ascii=False) + "\n"); fq.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(name, len(rows), "rows", flush=True)


# 1a HotpotQA
H = [r for f in sorted(os.listdir(f"{ND}/hotpot_qa/distractor")) if f.startswith("train") for r in pq.read_table(f"{ND}/hotpot_qa/distractor/{f}").to_pylist()]
H = [r for r in H if r["question"].strip() not in USED]; rng.shuffle(H); out = []
for n, r in enumerate(H[:NH]):
    ctx, sf = P(r["context"]), P(r["supporting_facts"])
    paras = list(zip(ctx["title"], ctx["sentences"])); sup = set(zip(sf["title"], sf["sent_id"]))
    out.append(row("hpq", n, r["question"], r["answer"], paras, sup))
write("hpq", out)

# 1b 2WikiMultiHopQA (train, stratified by type) + dev hop study
def w2rows(path):
    by = {}
    for r in pq.read_table(path).to_pylist():
        by.setdefault(r["type"], []).append(r)
    return by


by = w2rows(f"{ND}/2WikiMultihopQA/train.parquet"); out = []; types = sorted(by)
for v in by.values():
    rng.shuffle(v)
pick = [r for i in range(NW) for t in [types[i % len(types)]] if i // len(types) < len(by[t]) for r in [by[t][i // len(types)]]]
for n, r in enumerate(pick[:NW]):
    ctx, sf = P(r["context"]), P(r["supporting_facts"])
    out.append(row("w2", n, r["question"], r["answer"], [(t, ss) for t, ss in ctx], {(t, i) for t, i in sf}))
write("w2", out)
dv = w2rows(f"{ND}/2WikiMultihopQA/dev.parquet"); rows = []; n = 0
with open(f"{HS2}/ctx_hs.jsonl", "w") as fc, open(f"{HS2}/qw_hs.jsonl", "w") as fq:
    for t in sorted(dv):
        rng.shuffle(dv[t])
        for r in dv[t][:100]:
            ctx, sf = P(r["context"]), P(r["supporting_facts"])
            c, q = row(f"w2_{t}", n, r["question"], r["answer"], [(tt, ss) for tt, ss in ctx], {(tt, i) for tt, i in sf})
            fc.write(json.dumps(c, ensure_ascii=False) + "\n"); fq.write(json.dumps(q, ensure_ascii=False) + "\n")
            rows.append({"row": n, "type": t, "q": r["question"], "a": r["answer"], "gold": [g[0] for g in c["groups"]]}); n += 1
json.dump(rows, open(f"{HS2}/rows.json", "w"), ensure_ascii=False); print("2Wiki dev hop study", n, "rows", flush=True)

# 2 MuSiQue unanswerable twins
M = [json.loads(l) for l in open(f"{ND}/musique/musique_full_v1.0_train.jsonl")]
M = [x for x in M if not x["answerable"]]; rng.shuffle(M); out = []
for n, x in enumerate(M[:NM]):
    out.append(row("mqu", n, x["question"], x["answer"], [(p["title"], [p["paragraph_text"]]) for p in x["paragraphs"]], set(), allin=0.0))
write("mqu", out)
