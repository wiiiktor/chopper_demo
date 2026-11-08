#!/usr/bin/env python3
"""luna verdicts for the memory-benchmark loop (2026-11-08). Every final answer in each state file is judged against the reference with
judge_mh_luna's QA prompt (unanswerable questions: reference = "not mentioned", so a refusal is correct and an invented answer is wrong).
Cached on (question, gold, pred) in <memb dir>/luna_cache.jsonl. Prints luna-correct per benchmark x category, searches/question and
evidence recall (first 16 / kept by the chopper, any evidence chunk); writes <state>.judged.json.
usage: memb_judge.py <memb dir> <state.jsonl> [...]"""
import json, os, sys
import concurrent.futures as cf
from collections import defaultdict
sys.path.insert(0, "/home/wii/Projects/certain/scripts")
from judge_mh_luna import QA, ask
from openai import OpenAI
D, STS = sys.argv[1], sys.argv[2:]
CF = f"{D}/luna_cache.jsonl"; CACHE = {}
if os.path.exists(CF):
    CACHE = {(x["q"], x["gold"], x["pred"]): x["v"] for x in map(json.loads, open(CF))}
cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip()); stats = {"in": 0, "out": 0, "unparsed": 0, "errors": 0}
S = {st: [json.loads(l) for l in open(st)] for st in STS}
key = lambda s: (s["q"]["question"] + (f" (asked on {s['q']['date']})" if s["q"]["date"] else ""), s["q"]["gold"], s.get("pred", "") or "(no answer)")
todo = list(dict.fromkeys(key(s) for ss in S.values() for s in ss if key(s) not in CACHE))
print(f"{len(todo)} new verdicts", flush=True)
with cf.ThreadPoolExecutor(16) as ex, open(CF, "a") as f:
    for k, v in zip(todo, ex.map(lambda k: ask(cl, QA.format(q=k[0], gold=k[1], alias="", pred=k[2]), stats), todo)):
        CACHE[k] = v
        if v is not None:
            f.write(json.dumps({"q": k[0], "gold": k[1], "pred": k[2], "v": v}, ensure_ascii=False) + "\n")
print(f"luna tokens in {stats['in']} out {stats['out']} | unparsed {stats['unparsed']} errors {stats['errors']}", flush=True)
for st, ss in S.items():
    R = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for s in ss:
        for c in (s["q"]["cat"], "ALL"):
            r = R[(s["q"]["bench"], c)]; r[0] += 1; r[1] += bool(CACHE.get(key(s))); r[2] += len(s.get("queries", []))
            if s["q"]["ev"]:
                r[3] += 1; r[4] += bool(s.get("ev16")); r[5] += bool(s.get("evkept"))
    out = []
    for (b, c), r in sorted(R.items()):
        out.append({"bench": b, "cat": c, "n": r[0], "luna": r[1] / r[0], "searches": r[2] / r[0], "ev_first16": r[4] / max(1, r[3]), "ev_kept": r[5] / max(1, r[3])})
        print(f"{os.path.basename(os.path.dirname(st))}/{os.path.basename(st):18s} {b:6s} {c:28s} n={r[0]:3d} | luna-correct {r[1] / r[0]:.2f} | "
              f"searches/q {r[2] / r[0]:.2f} | evidence in first 16 {r[4] / max(1, r[3]):.2f} kept {r[5] / max(1, r[3]):.2f}")
    FULL = json.load(open(f"{D}/catfull.json"))         # reweight the stratified sample to each benchmark's real category mix (LoCoMo: categories 1-4, as the papers)
    for b in {k[0] for k in R}:
        W = {c: n for c, n in FULL[b].items() if c != "adversarial" and (b, c) in R}
        if W:
            acc = sum(n * R[(b, c)][1] / R[(b, c)][0] for c, n in W.items()) / sum(W.values())
            out.append({"bench": b, "cat": "WEIGHTED", "luna": acc})
            print(f"{os.path.basename(os.path.dirname(st))}/{os.path.basename(st):18s} {b:6s} {'WEIGHTED (paper mix' + (', no cat 5)' if b == 'locomo' else ')'):28s} | luna-correct {acc:.3f}")
    json.dump(out, open(st + ".judged.json", "w"), indent=1)
