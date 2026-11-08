#!/usr/bin/env python3
"""No-answer rows by deleting the answer, pool intact (2026-11-06). rr30made cut no-answer kept text to 0.4% but also cut benchmark answers (MuSiQue .46 -> .31, HotpotQA .80 -> .69): its filler passages came from
other rows, so a pool that looks incoherent may be what it learned to spot. Here every passage stays in place and on topic, only its material sentences go, so the pool looks like any other pool.
rr30del = 14,500 answer rows + real no-answer rows + deleted rows, 30% no-answer; rrval_del = 500 holdout rows made the same way.   usage: rr_del_pack.py <pack dir>"""
import json, random, sys
P = sys.argv[1]; R = random.Random(11)
load = lambda n: list(zip(map(json.loads, open(f"{P}/ctx_{n}.jsonl")), map(json.loads, open(f"{P}/qw_{n}.jsonl"))))


def delete(c, q, n):
    ps = [dict(p) for p in c["passages"]]; by = {}
    for k, a, b in q["spans"]:
        if not (a == 0 and ps[k]["text"][:5] == "<src>"):     # the tag stays, only sentences go
            by.setdefault(k, []).append((a, b))
    for k, S in by.items():
        t = ps[k]["text"]
        for a, b in sorted(S, reverse=True):
            t = t[:a].rstrip() + (" " if t[b:b + 1] not in ("\n", "") else "") + t[b:].lstrip(" ")
        ps[k]["text"] = t
    d = dict(c); d.update(passages=ps, allin=0.0, qtype="deleted_no_answer", row=n, answer="")
    return d, {"dataset": c["dataset"], "split": "rr", "row": n, "spans": [], "steps": [], "miss": []}


def write(name, rows):
    R.shuffle(rows)
    with open(f"{P}/ctx_{name}.jsonl", "w") as f, open(f"{P}/qw_{name}.jsonl", "w") as g:
        for c, q in rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n"); g.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(name, len(rows), "rows, no spans", sum(not q["spans"] for _, q in rows), flush=True)


TR = load("rrtrain"); ANS = [x for x in TR if x[0]["qtype"] != "no_answer" and x[1]["spans"]]; NO = [x for x in TR if x[0]["qtype"] == "no_answer"]
A = R.sample(ANS, 14500); N = round(len(A) * 3 / 7)
write("rr30del", A + NO + [delete(c, q, 20_000_000 + i) for i, (c, q) in enumerate(R.sample(ANS, N - len(NO)))])
V = load("rrval_notag"); VA = [x for x in V if x[0]["qtype"] != "no_answer" and x[1]["spans"]]
write("rrval_del", [delete(c, q, 20_000_000 + i) for i, (c, q) in enumerate(R.sample(VA, 500))])
