#!/usr/bin/env python3
"""GPT-6 Luna re-judges every passage luna 5.6 accepted in the rerank pilot (2026-11-05, user: "apply the 6 usd fix"). In blind tests
Opus 5.5 + GPT-6.1 Sol sided with GPT-6 Luna 57 to 17 over luna 5.6, which errs towards YES. Rows naming something use the source-line
prompt, generic rows the text prompt. Resumable. A failed call keeps the earlier verdict. Writes <dir>/luna_v2_g6_<lang>.jsonl {rid, i, how, v}.
usage: rr_g6_rejudge.py <dir> [lang ...]  (env THREADS=24)"""
import json, os, re, sys, time, urllib.request
import concurrent.futures as cf
D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]; TH = int(os.environ.get("THREADS", "24"))
KEY = open("/home/wii/Projects/certain/.openrouter_key").read().strip()
here = os.path.dirname(os.path.abspath(__file__))
P1 = open(f"{here}/rr_luna_gold.py").read().split('P = """')[1].split('"""')[0]
P2 = open(f"{here}/rr_luna_recheck.py").read().split('P = """')[1].split('"""')[0]
cost = [0.]


def ask(x):
    body = json.dumps({"model": "openai/gpt-6-luna", "max_tokens": 800, "messages": [{"role": "user", "content": x["p"]}], "usage": {"include": True}}).encode()
    for k in range(7):
        try:
            rq = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            j = json.load(urllib.request.urlopen(rq, timeout=120)); cost[0] += float(j.get("usage", {}).get("cost") or 0)
            m = re.search(r"\b(YES|NO)\b", (j["choices"][0]["message"]["content"] or "").upper())
            if m:
                return m.group(1) == "YES"
        except Exception as e:
            print("err", type(e).__name__, flush=True); time.sleep(3 * (k + 1))
    return None


for lang in LANGS:
    out = f"{D}/luna_v2_g6_{lang}.jsonl"
    done = {(x["rid"], x["i"]) for x in map(json.loads, open(out))} if os.path.exists(out) else set()
    L = {}
    for fn in (f"luna_v2_{lang}", f"luna_v2_extra_{lang}", f"luna_v2_recheck_{lang}"):
        L |= {(x["rid"], x["i"]): x for x in map(json.loads, open(f"{D}/{fn}.jsonl"))}
    C = {x["rid"]: x["c"] for x in map(json.loads, open(f"{D}/qclass_v2_{lang}.jsonl"))}
    todo = []
    for r in map(json.loads, open(f"{D}/rows_v2_{lang}.jsonl")):
        if C.get(r["rid"]) != "SOURCE":
            continue
        for i, p in enumerate(r["passages"]):
            x = L.get((r["rid"], i))
            if x and x["v"] and (r["rid"], i) not in done:
                m = p["meta"]; src = " | ".join(str(m[k]) for k in ("domain", "title", "section") if m.get(k)) or "unknown"
                pr = P2.format(q=r["query"], src=src, text=p["text"]) if r["specific"] else P1.format(q=r["query"], text=p["text"])
                todo.append({"rid": r["rid"], "i": i, "how": x["how"], "p": pr})
    t0 = time.time(); n = y = 0
    with cf.ThreadPoolExecutor(TH) as ex, open(out, "a") as f:
        for x, v in zip(todo, ex.map(ask, todo)):
            if v is not None:
                f.write(json.dumps({"rid": x["rid"], "i": x["i"], "how": x["how"], "v": v}) + "\n"); n += 1; y += v
    print(f"{lang}: {n}/{len(todo)} rejudged in {time.time() - t0:.0f}s | still YES {y} | running ${cost[0]:.2f}", flush=True)
