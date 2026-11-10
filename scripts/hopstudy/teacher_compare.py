#!/usr/bin/env python3
"""Compare teachers' labels on the same contexts: answer kept, each hop's gold passage quoted, text kept, and the character
agreement between the first two label sets. Rows are bucketed by hop count (hs rows: dataset hopK; val: MuSiQue / HotpotQA = 2
gold groups, TriviaQA = 1).
usage: teacher_compare.py <ctx.jsonl> name=labels.jsonl ... [--rows rows.json]"""
import json, sys
CTX = sys.argv[1]; L = [a.split("=", 1) for a in sys.argv[2:] if "=" in a and not a.startswith("--")]
ROWS = json.load(open(sys.argv[sys.argv.index("--rows") + 1])) if "--rows" in sys.argv else None
AL = {r["row"]: r.get("aliases", []) for r in ROWS} if ROWS else {}
C = [json.loads(l) for l in open(CTX)]
S = {nm: {(x["dataset"], x["row"]): x for x in map(json.loads, open(f))} for nm, f in L}
nm_ = lambda t: " ".join(t.lower().split())
chars = lambda c, sp: {(k, i) for k, a, b in sp for i in range(a, b) if not c["passages"][k]["text"][i].isspace()}
R = {}
for c in C:
    key = (c["dataset"], c["row"]); G = [g for g in c["groups"] if g]
    hk = c["dataset"] if not c["dataset"].startswith("hop") else c["dataset"]
    hk = f"{len(G)}-hop ({c['dataset']})" if not c["dataset"].startswith("hop") else c["dataset"].replace("hop", "") + "-hop"
    gi = {}                                                          # passage index -> hop position
    for k, p in enumerate(c["passages"]):
        if p["group"] >= 0:
            gi[k] = p["group"]
    sets = []
    for nm, D in S.items():
        sp = D.get(key, {}).get("spans", []); sets.append(chars(c, sp))
        kept = nm_(" | ".join(c["passages"][k]["text"][a:b] for k, a, b in sp))
        r = R.setdefault(hk, {}).setdefault(nm, {"n": 0, "ans": 0, "kept": 0, "reach": [0] * len(G), "miss": 0})
        r["n"] += 1; r["ans"] += nm_(c["answer"]) in kept or any(nm_(a) in kept for a in AL.get(c["row"], []))
        r["kept"] += sum(b - a for k, a, b in sp) / sum(len(p["text"]) for p in c["passages"])
        r["miss"] += len(D.get(key, {}).get("miss", []))
        for j in range(len(G)):
            r["reach"][j] += any(gi.get(k) == j for k, a, b in sp)
    if len(sets) >= 2:
        a, b = sets[0], sets[1]; f1 = 2 * len(a & b) / max(1, len(a) + len(b))
        R[hk].setdefault("_agree", []).append(f1)
out = {}
for hk in sorted(R):
    out[hk] = {}
    for nm, r in R[hk].items():
        if nm == "_agree":
            out[hk]["agreement_f1"] = sum(r) / len(r); continue
        out[hk][nm] = {"n": r["n"], "answer_quoted": r["ans"] / r["n"], "hop_quoted": [x / r["n"] for x in r["reach"]], "text_kept": r["kept"] / r["n"],
                       "quotes_not_found": r["miss"]}
        print(f"{hk:22s} {nm:8s} n={r['n']:3d} | answer quoted {r['ans'] / r['n']:.2f} | hop passage quoted {'/'.join(f'{x / r[chr(110)]:.2f}' for x in r['reach'])} "
              f"| text kept {r['kept'] / r['n']:.3f} | quotes not found {r['miss']}")
    if "agreement_f1" in out[hk]:
        print(f"{hk:22s} agreement (char F1, {L[0][0]} vs {L[1][0]}) {out[hk]['agreement_f1']:.2f}")
json.dump(out, open(sys.argv[2].split("=", 1)[1] + ".compare.json", "w"), indent=1)
