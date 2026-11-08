#!/usr/bin/env python3
"""Which hop does a chopper lose? (2026-11-08) On val questions whose gold passages are all among the 16: share of rows where each hop's gold passage keeps any fragment, all hops kept,
and the answer string inside the kept text. usage: rr_hop_report.py <mm dir> <eval dir with preds_val.jsonl> [...]"""
import json, sys, collections
C = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{sys.argv[1]}/ctx_val.jsonl"))}
for d in sys.argv[2:]:
    P = {(x["dataset"], x["row"]): x for x in map(json.loads, open(f"{d}/preds_val.jsonl"))}
    out = []
    for ds in ("musique", "hotpot"):
        hop = collections.defaultdict(lambda: [0, 0]); n = allk = ans = 0
        for k, c in C.items():
            if k[0] != ds or c["allin"] != 1.0 or k not in P:
                continue
            n += 1; kept = {f[0] for f in P[k]["fragments"]}; ok = []
            for g in sorted({p["group"] for p in c["passages"] if p["group"] >= 0}):
                h = bool({i for i, p in enumerate(c["passages"]) if p["group"] == g} & kept); hop[g][0] += 1; hop[g][1] += h; ok.append(h)
            allk += all(ok); ans += c["answer"].lower() in " ".join(f[3] for f in P[k]["fragments"]).lower()
        out.append(f"{ds}: hops " + "/".join(f"{v[1] / v[0]:.2f}" for g, v in sorted(hop.items())) + f" all {allk / n:.2f} answer {ans / n:.2f}")
    print(d.split("/")[-1], " | ".join(out), flush=True)
