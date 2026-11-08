#!/usr/bin/env python3
"""The full retrain's mix (2026-11-07): all pilot answer rows + 20% of the old rows (without them the benchmarks collapse) + 30% no-answer: every real no-answer row + rows with the answer sentences deleted, pool intact
(the filler kind teaches a shortcut, see rr_del_pack.py). Writes ctx_/qw_rrfull.   usage: rr_full_pack.py <pack dir>"""
import json, random, sys, os
sys.argv, ARGS = [sys.argv[0], sys.argv[1]], sys.argv
P = ARGS[1]; R = random.Random(13)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_del_pack.py")).read().split("TR = load(")[0]
exec(src); R = random.Random(13)
TR = load("rrtrain"); ANS = [x for x in TR if x[0]["qtype"] != "no_answer" and x[1]["spans"]]; NO = [x for x in TR if x[0]["qtype"] == "no_answer"]; OLD = load("old20")
A = ANS + OLD; N = round(len(A) * 3 / 7)
rows = A + NO + [delete(c, q, 30_000_000 + i) for i, (c, q) in enumerate(R.choices(ANS, k=N - len(NO)))]
write("rrfull", rows)
