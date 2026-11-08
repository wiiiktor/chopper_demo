#!/usr/bin/env python3
"""More no-answer rows for the rerank chopper (2026-11-06; only 4.5% of the pilot's rows were no-answer and the chopper kept text on 68 of 69 of them). Two training mixes at 30% no-answer / 70% answer:
rr30down (user: just remove answer rows): every real no-answer row + answer rows sampled down to 70%. rr30made: answer rows + no-answer rows made from other answer rows: the query stays, every passage
with an answer goes, and the gaps are filled with no-span passages of another row of the same language, area and source kind, tags matching, so pool size and tags give nothing away. Fillers were not
judged against this query, an accepted risk. Test sets: rrval_made (500 holdout rows made the same way) for what is kept when no answer exists.   usage: rr_noans_pack.py <pack dir>"""
import json, random, sys
from collections import defaultdict
P = sys.argv[1]; R = random.Random(7)
load = lambda n: list(zip(map(json.loads, open(f"{P}/ctx_{n}.jsonl")), map(json.loads, open(f"{P}/qw_{n}.jsonl"))))
tagged = lambda c: c["passages"][0]["text"].startswith("<src>")


def bucket(rows):
    B = defaultdict(list)
    for c, q in rows:
        pos = {k for k, *_ in q["spans"]}
        for i, p in enumerate(c["passages"]):
            if i not in pos:
                B[(c["dataset"], c["area"], c["kind"], tagged(c))].append(p)
    return B


def make(c, q, B, n):
    pos = {k for k, *_ in q["spans"]}; keep = [p for i, p in enumerate(c["passages"]) if i not in pos]
    pool = B[(c["dataset"], c["area"], c["kind"], tagged(c))]; have = {p["text"] for p in keep}
    while len(keep) < len(c["passages"]):
        p = R.choice(pool)
        if p["text"] not in have:
            keep.append(p); have.add(p["text"])
    R.shuffle(keep)
    d = dict(c); d.update(passages=keep, allin=0.0, qtype="made_no_answer", row=n, answer="")
    return d, {"dataset": c["dataset"], "split": "rr", "row": n, "spans": [], "steps": [], "miss": []}


def write(name, rows):
    R.shuffle(rows)
    with open(f"{P}/ctx_{name}.jsonl", "w") as f, open(f"{P}/qw_{name}.jsonl", "w") as g:
        for c, q in rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n"); g.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(name, len(rows), "rows, no spans", sum(not q["spans"] for _, q in rows), flush=True)


TR = load("rrtrain"); ANS = [x for x in TR if x[0]["qtype"] != "no_answer" and x[1]["spans"]]; NO = [x for x in TR if x[0]["qtype"] == "no_answer"]
B = bucket(TR); print("answer", len(ANS), "no-answer", len(NO), flush=True)
write("rr30down", NO + R.sample(ANS, round(len(NO) * 7 / 3)))
A = R.sample(ANS, 14500); N = round(len(A) * 3 / 7)
made = [make(c, q, B, 10_000_000 + i) for i, (c, q) in enumerate(R.sample(ANS, N - len(NO)))]
write("rr30made", A + NO + made)
V = load("rrval_notag"); VA = [x for x in V if x[0]["qtype"] != "no_answer" and x[1]["spans"]]; BV = bucket(V)
write("rrval_made", [make(c, q, BV, 10_000_000 + i) for i, (c, q) in enumerate(R.sample(VA, 500))])
