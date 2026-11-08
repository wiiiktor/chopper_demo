#!/usr/bin/env python3
"""Who is right when a cheaper judge (env CAND, default Gemini 2.5 Flash-Lite) and luna disagree? (2026-11-05) Re-asks it on the rr_judge_market.py pairs, saving its
answers, then two stronger models judge every disagreement plus 60 agreed pairs with the same prompts, blind to who said what.
usage: rr_adjudicate.py <rr dir> <strong model> <strong model>   -> <rr dir>/adjudicate.jsonl"""
import json, os, re, sys, random, importlib.util as iu
import concurrent.futures as cf
sys.argv, ARGS = [sys.argv[0], sys.argv[1], "__none__"], sys.argv
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_judge_market.py")).read().split("for model in MODELS:")[0]
exec(src)
if os.environ.get("ONLY"):
    A = [a for a in A if a[0] == os.environ["ONLY"]]
D, S1, S2 = ARGS[1:4]; CAND = os.environ.get("CAND", "google/gemini-2.5-flash-lite")
with cf.ThreadPoolExecutor(24) as ex:
    G = [r[0] for r in ex.map(lambda a: ask(CAND, a[1]), A)]
dis = [i for i, g in enumerate(G) if g is not None and g != A[i][2]]
agr = random.Random(0).sample([i for i, g in enumerate(G) if g is not None and g == A[i][2]], 60)
print(f"{len(dis)} disagreements, {len(agr)} agreed controls", flush=True)
T = dis + agr; cost = 0.; out = {i: {} for i in T}
for s in (S1, S2):
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda i: ask(s, A[i][1]), T))
    for i, r in zip(T, res):
        out[i][s] = r[0]; cost += r[1]
    print(s, "fail", sum(r[0] is None for r in res), f"cost so far ${cost:.2f}", flush=True)
with open(f"{D}/adjudicate_{CAND.split('/')[1]}.jsonl", "w") as f:
    for i in T:
        f.write(json.dumps({"set": A[i][0], "disagree": i in dis, "luna": A[i][2], "cand": G[i], S1: out[i][S1], S2: out[i][S2], "prompt": A[i][1]}, ensure_ascii=False) + "\n")
for tag, I in (("natural", [i for i in dis if A[i][0] == "natural"]), ("source", [i for i in dis if A[i][0] == "source"])):
    both = [i for i in I if out[i][S1] is not None and out[i][S1] == out[i][S2]]
    l = sum(out[i][S1] == A[i][2] for i in both); print(f"{tag}: {len(I)} disagreements | strong models agree on {len(both)} | side with luna {l}, with candidate {len(both) - l}", flush=True)
print("controls: strong agree with both", sum(out[i][S1] == A[i][2] and out[i][S2] == A[i][2] for i in agr), "/", len(agr), flush=True)
