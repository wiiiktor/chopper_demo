#!/usr/bin/env python3
"""Hop study report: per hop count (1-4), for the teacher (Qwen3-14B quotes), the perfect notes and every chopper:
answer string kept, each hop's gold paragraph reached (any kept text inside it), each hop's own answer kept, text kept;
and luna-correct answers per notes source (memb_judge.py <state>.judged.json). Writes <hs dir>/report.json.
usage: hs_report.py <hs dir> name=labels_or_preds ... -- arm=state.jsonl ..."""
import json, re, sys
HD = sys.argv[1]; i = sys.argv.index("--"); SRC = [a.split("=", 1) for a in sys.argv[2:i]]; ARMS = [a.split("=", 1) for a in sys.argv[i + 1:]]
R = {r["row"]: r for r in json.load(open(f"{HD}/rows.json"))}
C = {c["row"]: c for c in map(json.loads, open(f"{HD}/ctx_hs.jsonl"))}
Q = {q["row"]: q for q in map(json.loads, open(f"{HD}/qw_hs.jsonl"))}
nm = lambda t: " ".join(t.lower().split())
out = {"chop": {}, "answers": {}}
for name, f in [("perfect", None)] + SRC:
    L = {x["row"]: x for x in map(json.loads, open(f))} if f else {}
    st = {}
    for n, r in R.items():
        c = C[n]
        if name == "perfect":
            sp = Q[n]["spans"]
        elif "fragments" in next(iter(L.values()), {}):
            # mm_train fragment offsets are in its concatenated input, and a fragment can run across passages: place its pieces by text
            fr = L.get(n, {}).get("fragments", []); sp = []
            for x in fr:
                for piece in re.split(r"<eos>|\n", x[3]):
                    piece = piece.strip()
                    if len(piece) < 12:
                        continue
                    for k, pp in enumerate(c["passages"]):
                        i = pp["text"].find(piece)
                        if i >= 0:
                            sp.append([k, i, i + len(piece)]); break
        else:
            sp = L.get(n, {}).get("spans", [])
        kept = nm(" | ".join(c["passages"][k]["text"][a:b] for k, a, b in sp))
        tot = sum(len(p["text"]) for p in c["passages"]); kc = sum(b - a for k, a, b in sp)
        s = st.setdefault(r["hops"], {"n": 0, "ans": 0, "kept": 0.0, "reach": [0] * r["hops"], "hopans": [0] * r["hops"]})
        s["n"] += 1; s["ans"] += nm(r["a"]) in kept or any(nm(x) in kept for x in r["aliases"]); s["kept"] += kc / tot
        for j, (pi, ha) in enumerate(r["gold"]):
            s["reach"][j] += any(k == pi for k, a, b in sp); s["hopans"][j] += nm(ha) in kept
    out["chop"][name] = {k: {"n": v["n"], "answer_kept": v["ans"] / v["n"], "text_kept": v["kept"] / v["n"],
                             "hop_reached": [x / v["n"] for x in v["reach"]], "hop_answer_kept": [x / v["n"] for x in v["hopans"]]} for k, v in sorted(st.items())}
for arm, f in ARMS:
    J = json.load(open(f + ".judged.json"))
    W = [json.loads(l)["notes_words"] for l in open(f)]
    out["answers"][arm] = {"words": sum(W) / len(W), **{x["cat"]: x["luna"] for x in J if x["cat"].startswith("hop")}}
json.dump(out, open(f"{HD}/report.json", "w"), indent=1)
for name, v in out["chop"].items():
    print(name, " | ".join(f"hop{k}: ans {x['answer_kept']:.2f} reach {'/'.join(f'{y:.2f}' for y in x['hop_reached'])} kept {x['text_kept']:.3f}" for k, x in v.items()))
for arm, v in out["answers"].items():
    print("answers", arm, {k: round(x, 3) for k, x in v.items()})
