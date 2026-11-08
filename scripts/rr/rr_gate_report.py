#!/usr/bin/env python3
"""No-answer scoreboard of the rerank choppers (2026-11-06), plain and gated. Gate = if the decision head says SEARCH (p_answer < .5) keep nothing. For each chopper: holdout F1/P/R on passage bodies (no tags),
kept text and rows with any text on real no-answer rows and on the made and deleted test sets, SEARCH rate on each, wrong SEARCH on answer rows, benchmark answer kept.
usage: rr_gate_report.py <rr dir> <who> [<who> ...]"""
import json, os, re, sys
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
D = sys.argv[1]; P = f"{D}/pack"; _t = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"); SEP = f" {_t.sep_token} " if _t.sep_token else "\n\n"
load = lambda f: list(map(json.loads, open(f)))
CN = {(c["dataset"], c["row"]): c for c in load(f"{P}/ctx_rrval_notag.jsonl")}; QN = {(q["dataset"], q["row"]): q for q in load(f"{P}/qw_rrval_notag.jsonl")}
CM = {n: {(c["dataset"], c["row"]): c for c in load(f"{P}/ctx_rrval_{n}.jsonl")} for n in ("made", "del")}


def holdout(who, gate):
    tp = pr = gl = 0; f = f"{D}/evalrr_{who}_rrval_notag/preds_val.jsonl"; noans = []; wrong = [0, 0]; real = [0, 0]
    for x in load(f):
        k = (x["dataset"], x["row"]); c = CN[k]; q = QN[k]; s = f"Question: {c['question']}"; st = []
        for p in c["passages"]:
            s += SEP; st.append(len(s)); s += p["text"].strip()
        g = np.zeros(len(s) + 1, bool); d = np.zeros(len(s) + 1, bool); search = x["p_answer"] < .5
        lead = [len(p["text"]) - len(p["text"].lstrip()) for p in c["passages"]]
        for j, a, b in q["spans"]:
            g[st[j] + a - lead[j]:st[j] + b - lead[j]] = True
        if not (gate and search):
            for fr in x["fragments"]:
                d[fr[1]:fr[2]] = True
        tp += int((g & d).sum()); pr += int(d.sum()); gl += int(g.sum())
        if c["qtype"] == "no_answer":
            noans.append(d.sum() / len(s)); real[0] += 1; real[1] += search
        else:
            wrong[0] += 1; wrong[1] += search
    pp, rr = tp / max(pr, 1), tp / max(gl, 1)
    return 2 * pp * rr / max(pp + rr, 1e-9), pp, rr, float(np.mean(noans)), real[1] / real[0], wrong[1] / wrong[0]


def noans(who, n, gate):
    S, W = [], 0; f = f"{D}/evalrr_{who}_rrval_{n}/preds_val.jsonl"
    if not os.path.exists(f):
        return None
    R = load(f)
    for x in R:
        c = CM[n][(x["dataset"], x["row"])]; t = sum(len(p["text"]) for p in c["passages"]); k = 0 if gate and x["p_answer"] < .5 else sum(a[2] - a[1] for a in x["fragments"])
        S.append(k / t); W += k > 0
    return float(np.mean(S)), W, float(np.mean([x["p_answer"] < .5 for x in R]))


for who in sys.argv[2:]:
    b = {}
    for l in open(f"{D}/evalrr_{who}_val.out"):
        m = re.search(r"\| ([a-z]+) \(\d+ q\).*answer in kept ([\d.]+) \| decision: SEARCH recall ([\d.]+).*ANSWER recall ([\d.]+)", l)
        if m:
            b[m.group(1)[0].upper()] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    print(f"== {who}  benchmarks answer kept / search / answer recall: {b}", flush=True)
    for gate in (False, True):
        F, Pp, Rr, nk, real, wrong = holdout(who, gate)
        print(f" gate={gate!s:5} F1 {F:.3f} P {Pp:.3f} R {Rr:.3f} | real no-answer kept {nk:.3f} search {real:.3f} | answer rows wrong search {wrong:.3f}", flush=True)
        for n in ("made", "del"):
            v = noans(who, n, gate)
            if v:
                print(f"     {n}: kept {v[0]:.3f} rows with text {v[1]}/500 search {v[2]:.3f}", flush=True)
