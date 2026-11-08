#!/usr/bin/env python
"""Tool side of the native tool-call loop (2026-11-08). For every pending search(query): dense (Qwen3-Embedding) top-100 + BM25 top-50 over the question's own
pool, bge-reranker on the QUERY alone, the 8 best passages not yet seen; the continued mmBERT chopper then cuts them with the query as its question, and its
fragments come back as the tool turn.   usage: tool_eval_fetch.py <mm dir> <state.jsonl> <work dir> <chopper.pt>"""
import json, os, subprocess, sys
MM, ST, WD, CH = sys.argv[1:5]; os.makedirs(WD, exist_ok=True)
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mm_search_loop.py")).read().split("R = {}; t0 = time.time()")[0]
sys.argv = ["x", MM, "/media/wii/bigdata/rr/evalrr_fullc_val/preds_val.jsonl"]; exec(SRC)          # pools, chunks, embedder, reranker, search()
S = [json.loads(l) for l in open(ST)]; todo = [s for s in S if s.get("pending")]
rows = []
for n, s in enumerate(todo):
    key = (s["key"][0], s["key"][1]); p = POOL[key]; seen = set(s["seen"])
    new = [i for i in search(p, s["pending"]) if i not in seen][:8]; s["seen"] += new
    grp = {i: gi for gi, g in enumerate(p["groups"]) for i in g}
    rows.append({"dataset": key[0], "split": "val", "row": n, "question": s["pending"], "answer": "", "groups": [], "allin": 1.0,
                 "passages": [{"pos": i, "cid": p["chunks"][i], "text": T[p["chunks"][i]], "group": grp.get(i, -1)} for i in new]})
with open(f"{WD}/ctx_fetch.jsonl", "w") as f, open(f"{WD}/qw_fetch.jsonl", "w") as g:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); g.write(json.dumps({"dataset": r["dataset"], "split": "val", "row": r["row"], "spans": [], "steps": [], "miss": []}) + "\n")
    for nm in ("rrval_tagonly",):
        src = f"/media/wii/bigdata/rr/pack/ctx_{nm}.jsonl"
        for pre in ("ctx", "qw"):
            d = f"{WD}/{pre}_{nm}.jsonl"
            if not os.path.exists(d):
                os.symlink(src.replace("ctx_", pre + "_"), d)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mm_train.py"), WD, f"{WD}/chop", "--base", "jhu-clsp/mmBERT-base",
                "--train-splits", "rrval_tagonly", "--eval-only", CH, "--val-split", "fetch", "--pred-thr", "0.5", "--merge-gap", "5", "--min-run", "4"],
               check=True, stdout=open(f"{WD}/chop.out", "w"), stderr=subprocess.STDOUT)
P = {(x["dataset"], x["row"]): x for x in map(json.loads, open(f"{WD}/chop/preds_val.jsonl"))}
for n, s in enumerate(todo):
    fr = [" ".join(f[3].split()) for f in P[(rows[n]["dataset"], n)]["fragments"]]
    s["messages"].append({"role": "tool", "content": "\n".join(f"- {t}" for t in fr) if fr else "(no relevant passages found)"}); s["pending"] = None
with open(ST, "w") as f:
    f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in S)
print(f"fetched {len(todo)}", flush=True); sys.stdout.flush(); os._exit(0)
