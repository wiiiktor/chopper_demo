#!/usr/bin/env python3
"""Pilot verdict from the dumped fragments, scored fairly (2026-11-06). The trainer's F1 counts the tag as a target in tagged rows, which flatters them, so here
everything is on passage BODIES, in characters: P / R / F1 over the 8 languages, per query type and source kind; false keeps on no-answer rows; for tag-only, how often the
passages picked from tags alone are the true positives, against the chance rate. usage: rr_holdout_report.py <rr dir>"""
import json, os, re, sys, glob
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
D = sys.argv[1]; P = f"{D}/pack"; _t = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"); SEP = f" {_t.sep_token} " if _t.sep_token else "\n\n"
GOLD = {}
for c, q in zip(map(json.loads, open(f"{P}/ctx_rrval_tag.jsonl")), map(json.loads, open(f"{P}/qw_rrval_tag.jsonl"))):
    GOLD[(c["dataset"], c["row"])] = (c, q)


def body_offsets(c):
    return [(m.end() if (m := re.match(r"<src>.*?</src>\n", p["text"], re.S)) else 0) for p in c["passages"]]


def gold_body(c, q):
    off = body_offsets(c); G = defaultdict(set)
    for k, a, b in q["spans"]:
        if a >= off[k]:
            G[k] |= set(range(a - off[k], b - off[k]))
    return G


def layout(c):
    """the trainer's string: passage starts and, per passage, where its stripped text and its body begin"""
    s = f"Question: {c['question']}"; st, lead = [], []
    for p in c["passages"]:
        s += SEP; st.append(len(s)); s += p["text"].strip(); lead.append(len(p["text"]) - len(p["text"].lstrip()))
    return s, st, lead


def score(arm, who):
    ctx = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{P}/ctx_{arm}.jsonl"))}
    T = defaultdict(lambda: [0, 0, 0, 0]); F = [0, 0]
    for x in map(json.loads, open(glob.glob(f"{D}/evalrr_{who}_{arm}/preds*.jsonl")[0])):
        key = (x["dataset"], x["row"]); c0, q0 = GOLD[key]; c = ctx[key]; G = gold_body(c0, q0); off = body_offsets(c)
        s, st, lead = layout(c); body = np.zeros(len(s) + 1, bool); gold = np.zeros(len(s) + 1, bool); pred = np.zeros(len(s) + 1, bool)
        for k, p in enumerate(c["passages"]):
            b0 = st[k] + off[k] - lead[k]; body[b0:st[k] + len(p["text"].strip())] = True
            if G[k]:
                gold[[b0 + j for j in G[k]]] = True
        for fr in x.get("fragments") or []:
            pred[fr[1]:fr[2]] = True
        pred &= body; tp = int((pred & gold).sum()); pr = int(pred.sum()); gl = int(gold.sum())
        for tag in ("all", c["qtype"], "kind " + c["kind"], c["dataset"]):
            t = T[tag]; t[0] += tp; t[1] += pr; t[2] += gl; t[3] += 1
        if c["qtype"] == "no_answer":
            F[0] += 1; F[1] += pr > 0
    out = {}
    for k, (tp, pr, gl, n) in T.items():
        p, r = tp / max(1, pr), tp / max(1, gl); out[k] = (round(2 * p * r / max(1e-9, p + r), 3), round(p, 3), round(r, 3), n)
    return out, F


for who in ("old", "new"):
    for arm in ("rrval_tag", "rrval_notag", "rrval_swap"):
        o, F = score(arm, who); print(who, arm, "ALL F1/P/R", o["all"][:3], f"| no-answer rows with anything kept {F[1]}/{F[0]}", flush=True)
        if who == "new" and arm != "rrval_swap":
            for k in sorted(o):
                if k != "all":
                    print("   ", k, o[k], flush=True)
# tag-only: passages picked from tags alone vs true positives (a fragment belongs to the passage its midpoint falls in)
for who in ("old", "new"):
    ctx = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{P}/ctx_rrval_tagonly.jsonl"))}
    pick = hit = rows = pos = 0
    for x in map(json.loads, open(glob.glob(f"{D}/evalrr_{who}_rrval_tagonly/preds*.jsonl")[0])):
        c0, q0 = GOLD[(x["dataset"], x["row"])]; Tp = {k for k, *_ in q0["spans"]}; _, st, _ = layout(ctx[(x["dataset"], x["row"])])
        Pk = {int(np.searchsorted(st, (fr[1] + fr[2]) / 2, side="right")) - 1 for fr in x.get("fragments") or []}
        rows += 1; pos += len(Tp); pick += len(Pk); hit += len(Pk & Tp)
    print(who, f"tag-only: {pick / rows:.2f} passages picked per row, {hit / max(1, pick):.3f} of them true positives; chance {pos / rows / 16:.3f}", flush=True)

# 2026-11-06 (user: "add more charts"): everything the chart page needs, in one file
OUT = {"holdout": {}, "bench": {}, "sweep": {}, "tagonly": {}, "curve": [], "window": {}}
SW = re.compile(r"p>0\.(\d): P ([\d.]+) R ([\d.]+)")
for who in ("old", "new", "7k", "14k", "30down", "30made", "full", "fullc"):
    for arm in ("rrval_tag", "rrval_notag", "rrval_swap"):
        if glob.glob(f"{D}/evalrr_{who}_{arm}/preds*.jsonl"):
            o, F = score(arm, who); OUT["holdout"].setdefault(who, {})[arm] = {"sec": o, "noans": F}
            for l in open(f"{D}/evalrr_{who}_{arm}.out"):
                if "sweep" in l and arm == "rrval_notag":
                    OUT["sweep"][who] = [[float("0." + a), float(p), float(r)] for a, p, r in SW.findall(l)]
    for v in ("val", "wval"):
        f = f"{D}/evalrr_{who}_{v}.out"
        if os.path.exists(f):
            for l in open(f):
                m = re.search(r"\| ([a-z0-9]+) \(\d+ q\) \| KEEP P ([\d.]+) R ([\d.]+) F1 ([\d.]+) \| kept ([\d.]+).*answer in kept ([\d.]+)", l)
                if m:
                    OUT["bench"].setdefault(who, {})[m.group(1)] = {"f1": float(m.group(4)), "kept": float(m.group(5)), "ans": float(m.group(6))}
for who, rows in (("old", 0), ("7k", 7234), ("14k", 14468), ("new", 28937)):
    h = OUT["holdout"].get(who)
    if h and "rrval_notag" in h:
        OUT["curve"].append({"rows": rows, "tag": h["rrval_tag"]["sec"]["all"][0], "notag": h["rrval_notag"]["sec"]["all"][0], "bench": OUT["bench"].get(who)})
W = "/media/wii/bigdata/wtest"
for a in ("full", "w16k", "w8k", "w4k", "w8k_nopin", "tiers8k"):
    if os.path.exists(f"{W}/{a}.jsonl"):
        S = defaultdict(list)
        for x in map(json.loads, open(f"{W}/{a}.jsonl")):
            S[x["task"]].append(x["score"])
        OUT["window"][a] = {k: round(sum(v) / len(v), 3) for k, v in S.items()}
OUT["noans"] = {}
for who in ("old", "new", "full", "fullc"):
    for n in ("made", "del", "notag"):
        f = f"{D}/evalrr_{who}_rrval_{n}/preds_val.jsonl"
        if os.path.exists(f):
            R = [json.loads(l) for l in open(f)]; CC = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{P}/ctx_rrval_{n}.jsonl"))}
            if n == "notag":
                A1 = [x for x in R if x["decision_gold"] == 1]; R = [x for x in R if x["decision_gold"] == 0]
                OUT["noans"].setdefault(who, {})["wrong"] = sum(x["p_answer"] < .5 for x in A1) / len(A1)
            kept = [sum(a[2] - a[1] for a in x["fragments"]) / sum(len(p["text"]) for p in CC[(x["dataset"], x["row"])]["passages"]) for x in R]
            OUT["noans"].setdefault(who, {})[n] = {"kept": sum(kept) / len(kept), "search": sum(x["p_answer"] < .5 for x in R) / len(R)}
json.dump(OUT, open(f"{D}/pilot_report.json", "w"))
print("wrote pilot_report.json", flush=True)
