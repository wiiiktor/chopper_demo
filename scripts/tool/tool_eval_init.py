#!/usr/bin/env python3
"""Start states for the native tool-call loop: the 300 val questions, notes = the continued chopper's fragments of the first 16; seen = passages already returned by a tool turn (empty: a first-pool passage the chopper skipped under the original question may be exactly the hop the search asks for).
Also tool_eval_final: write preds for judge_answerer_luna.py.   usage: tool_eval_init.py <mm dir> <state.jsonl> | tool_eval_init.py --final <state.jsonl> <run dir>"""
import json, os, sys
if sys.argv[1] == "--final":
    ST, RUN = sys.argv[2:4]; os.makedirs(RUN, exist_ok=True)
    with open(f"{RUN}/preds_0.jsonl", "w") as f:
        for s in map(json.loads, open(ST)):
            f.write(json.dumps({"dataset": s["key"][0], "row": s["key"][1], "notes": "chopper", "pred": s.get("pred", ""), "queries": s.get("queries", [])}, ensure_ascii=False) + "\n")
    raise SystemExit
MM, ST = sys.argv[1:3]
B = json.loads(open("/media/wii/bigdata/rr/tool_train.jsonl").readline())
P = {(x["dataset"], x["row"]): x for x in map(json.loads, open("/media/wii/bigdata/rr/evalrr_fullc_val/preds_val.jsonl"))}
with open(ST, "w") as f:
    for c in map(json.loads, open(f"{MM}/ctx_val.jsonl")):
        k = (c["dataset"], c["row"]); fr = [" ".join(x[3].split()) for x in P[k]["fragments"]] if k in P else []
        notes = "\n".join(f"- {t}" for t in fr) if fr else "(none)"
        f.write(json.dumps({"key": list(k), "system": B["system"], "tools": B["tools"], "seen": [],
                            "messages": [{"role": "user", "content": f"Notes:\n{notes}\n\nQuestion: {c['question']}"}]}, ensure_ascii=False) + "\n")
