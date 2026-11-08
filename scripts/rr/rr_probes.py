#!/usr/bin/env python3
"""Pre-training shortcut audit of the packed rerank rows (taxonomy labels.tag_leak_audit). For rows with a positive and a negative passage, can a model
find the positives WITHOUT reading the passage for meaning? Logistic regressions on a train/test row split, scored by within-row AUC (.5 = no leak):
tag only (hashed tag tokens + field presence), tag + length (+ sentence count + slot), and query-passage word overlap, per query type and kind.
Also the positive rate per kind and the share of rows with no span.   usage: rr_probes.py <pack dir>"""
import json, re, sys, zlib
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
O = sys.argv[1]; C = list(map(json.loads, open(f"{O}/ctx_rrtrain.jsonl"))); Q = {(q["dataset"], q["row"]): q for q in map(json.loads, open(f"{O}/qw_rrtrain.jsonl"))}
H = 2 ** 14; TOK = re.compile(r"\w+", re.U); FIELDS = 8


def words(t):
    return set(TOK.findall(t.lower()))


def split_tag(t):
    m = re.match(r"<src>(.*?)</src>\n", t, re.S)
    return (m.group(1), t[m.end():]) if m else ("", t)


rows = []
for c in C:
    pos = {k for k, *_ in Q[(c["dataset"], c["row"])]["spans"]}
    if not pos or len(pos) == 16:
        continue
    qw = words(c["question"]); it = []
    for k, p in enumerate(c["passages"]):
        tg, body = split_tag(p["text"]); w = words(body)
        it.append({"y": k in pos, "tag": tg, "len": len(body), "slot": k, "ov": len(qw & w) / max(1, len(qw)), "cos": len(qw & w) / max(1, len(qw | w))})
    rows.append((c, it))
print(f"{len(C)} rows, {len(rows)} usable, no-span rows {sum(not Q[(c['dataset'], c['row'])]['spans'] for c in C)}", flush=True)
ids = np.random.RandomState(0).permutation(len(rows)); cut = int(.8 * len(rows)); TRN, TST = ids[:cut], ids[cut:]


def feats(p, mode):
    v = {}
    if p["tag"]:
        fs = p["tag"].split(" | "); v[H + min(len(fs), FIELDS - 1)] = 1.
        for t in TOK.findall(p["tag"].lower()):
            j = zlib.crc32(t.encode()) % H; v[j] = v.get(j, 0) + 1.
    if mode == "taglen":
        n = H + FIELDS; v[n] = np.log1p(p["len"]) / 8; v[n + 1 + p["slot"]] = 1.
    return v


def within(score, idx):
    A = []
    for i, s in zip(idx, score):
        y = np.array([p["y"] for p in rows[i][1]]); s = np.array(s)
        if y.any() and (~y).any():
            A.append(np.mean([(a > b) + .5 * (a == b) for a in s[y] for b in s[~y]]))
    return float(np.mean(A)), len(A)


def run(mode):
    from scipy.sparse import csr_matrix
    def mat(idx):
        d, r, c, y = [], [], [], []; n = 0
        for i in idx:
            for p in rows[i][1]:
                for j, v in feats(p, mode).items():
                    d.append(v); r.append(n); c.append(j)
                y.append(p["y"]); n += 1
        return csr_matrix((d, (r, c)), shape=(n, H + FIELDS + 20)), np.array(y)
    X, y = mat(TRN); m = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(X, y)
    Xt, _ = mat(TST); p = m.decision_function(Xt)
    return within([p[16 * k:16 * k + 16] for k in range(len(TST))], TST)


for mode in ("tag", "taglen"):
    print(f"probe {mode}: within-row AUC {run(mode)[0]:.3f} (.5 = no leak)", flush=True)
by = defaultdict(list)
for i, (c, it) in enumerate(rows):
    by["all"].append(i); by[c["qtype"]].append(i); by["kind " + c["kind"]].append(i)
for k, I in sorted(by.items()):
    a, n = within([[p["ov"] for p in rows[i][1]] for i in I], I); b, _ = within([[p["len"] for p in rows[i][1]] for i in I], I)
    rt = np.mean([sum(p["y"] for p in rows[i][1]) for i in I])
    print(f"{k:28s} rows {n:6d} | positives/row {rt:.2f} | word overlap AUC {a:.3f} | length AUC {b:.3f}", flush=True)
