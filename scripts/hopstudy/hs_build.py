#!/usr/bin/env python3
"""Hop study (user 2026-11-09: "stats should focus on hop number, not a benchmark ... 1 2 3 4 hop questions separately, with
influence of the answerer and qwen teacher; would we gain from a near perfect teacher?"). One source for every hop count:
MuSiQue dev (answerable). 1-hop = the natural first sub-question of a 2-hop question (no "X >> rel" form) with its own answer;
2/3/4-hop = full questions. Pool = MuSiQue's own 20 paragraphs ("title\\ntext"), so every gold hop paragraph is present: the
study isolates the chopper, the teacher and the answerer from retrieval.
"Perfect notes" (the near-perfect-teacher proxy): for every hop, the sentences of its gold paragraph that contain that hop's
answer (else the whole paragraph), with the title, built from MuSiQue's own decomposition, no model involved.
Writes <out>/ctx_hs.jsonl + qw_hs.jsonl (mm_train / mla eval format, spans = perfect sentences) and <out>/rows.json (hops,
per-hop gold paragraph index + answer, perfect notes).
usage: hs_build.py <musique dev jsonl> <out dir> [--n 100]"""
import json, os, random, re, sys
SRC, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "hs"          # 2026-11-10: also builds multi-hop TRAINING rows (--tag mhtrain)
SKIP = set()                                                       # --skip-used: questions already in the 54k chopper rows are left out
if "--skip-used" in sys.argv:
    for f in ("train", "train2", "train3"):
        SKIP |= {json.loads(l)["question"].strip() for l in open(f"/media/wii/bigdata/mm/ctx_{f}.jsonl")}
os.makedirs(OUT, exist_ok=True); rng = random.Random(0)
D = [json.loads(l) for l in open(SRC)]; D = [x for x in D if x["answerable"] and x["question"].strip() not in SKIP]
sents = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
norm = lambda t: " ".join(t.lower().split())
by = {1: [], 2: [], 3: [], 4: []}
for x in D:
    k = int(x["id"][0]); dec = x["question_decomposition"]
    by[k].append({"q": x["question"], "a": x["answer"], "hops": [(d["paragraph_support_idx"], d["answer"]) for d in dec], "x": x})
    if k == 2 and ">>" not in dec[0]["question"] and "#" not in dec[0]["question"]:
        by[1].append({"q": dec[0]["question"], "a": dec[0]["answer"], "hops": [(dec[0]["paragraph_support_idx"], dec[0]["answer"])], "x": x})
rows, n = [], 0
with open(f"{OUT}/ctx_{TAG}.jsonl", "w") as fc, open(f"{OUT}/qw_{TAG}.jsonl", "w") as fq:
    for k in (1, 2, 3, 4):
        rng.shuffle(by[k])
        for r in by[k][:N]:
            P = r["x"]["paragraphs"]; texts = [f"{p['title']}\n{p['paragraph_text']}" for p in P]
            spans, perfect = [], []
            for j, (pi, ha) in enumerate(r["hops"]):
                if TAG != "hs":                            # 2026-11-10: every hop fact keeps its paragraph TITLE (= its subject): 10% of hop
                    spans.append([pi, 0, len(P[pi]["title"])])   # sentences only say "He was drafted by ..." and are useless to the answerer alone
                t = texts[pi]; hit = [s for s in sents(P[pi]["paragraph_text"]) if norm(ha) in norm(s)] or [P[pi]["paragraph_text"]]
                perfect.append(f"{P[pi]['title']}: " + " ".join(hit))
                for s in hit:
                    i = t.find(s)
                    if i >= 0:
                        spans.append([pi, i, i + len(s)])
            c = {"dataset": f"hop{k}", "split": TAG, "row": n, "question": r["q"], "answer": r["a"], "groups": [[pi] for pi, _ in r["hops"]],
                 "allin": 1.0, "passages": [{"pos": i, "cid": -1, "text": t, "group": next((j for j, (pi, _) in enumerate(r["hops"]) if pi == i), -1)}
                                            for i, t in enumerate(texts)]}
            fc.write(json.dumps(c, ensure_ascii=False) + "\n")
            fq.write(json.dumps({"dataset": f"hop{k}", "split": TAG, "row": n, "spans": spans, "steps": [], "miss": []}) + "\n")
            rows.append({"row": n, "hops": k, "q": r["q"], "a": r["a"], "aliases": r["x"].get("answer_aliases", []) if k > 1 else [],
                         "gold": [[pi, ha] for pi, ha in r["hops"]], "perfect": perfect}); n += 1
json.dump(rows, open(f"{OUT}/rows{'' if TAG == 'hs' else '_' + TAG}.json", "w"), ensure_ascii=False)
print({k: min(N, len(v)) for k, v in by.items()}, "rows", n)
