#!/usr/bin/env python3
"""Complete labels on generic rows of the rerank pilot (2026-11-05, user approved): on rows whose query names nothing and that
rr_query_class.py kept (SOURCE), luna reads every passage the first pass skipped, so an unread passage is never a silent "no answer".
Resumable. Writes <dir>/luna_v2_extra_<lang>.jsonl {rid, i, how: "all", v}.   usage: rr_luna_extra.py <dir> [lang ...]  (env THREADS=32)"""
import json, os, sys, threading, time
import concurrent.futures as cf
sys.argv, ARGS = [sys.argv[0], sys.argv[1], "__none__"], sys.argv
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_luna_label.py")).read().split("for lang in LANGS:")[0]
exec(src)                                              # same prompt, client and retry logic as the first pass
D = ARGS[1]; LANGS = ARGS[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]      # after exec, which sets its own
for lang in LANGS:
    out = f"{D}/luna_v2_extra_{lang}.jsonl"
    done = {(x["rid"], x["i"]) for f in (f"{D}/luna_v2_{lang}.jsonl", out) if os.path.exists(f) for x in map(json.loads, open(f))}
    C = {x["rid"]: x["c"] for x in map(json.loads, open(f"{D}/qclass_v2_{lang}.jsonl"))}
    todo = []
    for r in map(json.loads, open(f"{D}/rows_v2_{lang}.jsonl")):
        if r["specific"] or C.get(r["rid"]) != "SOURCE":
            continue
        todo += [{"rid": r["rid"], "i": i, "q": r["query"], "text": p["text"]} for i, p in enumerate(r["passages"]) if (r["rid"], i) not in done]
    t0 = time.time(); n = y = 0
    with cf.ThreadPoolExecutor(TH) as ex, open(out, "a") as f:
        for x, v in zip(todo, ex.map(ask, todo)):
            if v is not None:
                f.write(json.dumps({"rid": x["rid"], "i": x["i"], "how": "all", "v": v}) + "\n"); n += 1; y += v
    print(f"{lang}: {n}/{len(todo)} read in {time.time() - t0:.0f}s | YES {y} | running ${(tok[0] * 0.20 + tok[1] * 1.20) / 1e6:.2f}", flush=True)
