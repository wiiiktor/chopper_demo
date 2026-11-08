#!/usr/bin/env python3
"""Cheaper judges against luna (2026-11-05, OpenRouter key added). Same prompts, same pairs: natural = luna_cost_1k.jsonl (text-only
prompt), source = up to 300 English rechecks (source shown). Prints agreement with luna, precision and recall of YES, tokens and $ per 1k.
usage: rr_judge_market.py <rr dir> <model> [<model> ...]"""
import json, os, re, sys, time
import concurrent.futures as cf
import urllib.request
D = sys.argv[1]; MODELS = sys.argv[2:]
KEY = open("/home/wii/Projects/certain/.openrouter_key").read().strip()
here = os.path.dirname(os.path.abspath(__file__))
P1 = open(f"{here}/rr_luna_gold.py").read().split('P = """')[1].split('"""')[0]
P2 = open(f"{here}/rr_luna_recheck.py").read().split('P = """')[1].split('"""')[0]
ROWS = {}
for l in ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]:
    for r in map(json.loads, open(f"{D}/rows_{l}.jsonl")):
        ROWS[r["rid"]] = r
A = []
for x in map(json.loads, open(f"{D}/luna_cost_1k.jsonl")):
    if x["luna"] is not None:
        r = ROWS[x["rid"]]; A.append(("natural", P1.format(q=r["query"], text=r["passages"][x["i"]]["text"]), x["luna"]))
V2 = {r["rid"]: r for r in map(json.loads, open(f"{D}/rows_v2_en.jsonl"))}
for n, x in enumerate(map(json.loads, open(f"{D}/luna_v2_recheck_en.jsonl"))):
    if n < 300:
        r = V2[x["rid"]]; p = r["passages"][x["i"]]; m = p["meta"]
        src = " | ".join(str(m[k]) for k in ("domain", "title", "section") if m.get(k)) or "unknown"
        A.append(("source", P2.format(q=r["query"], src=src, text=p["text"]), x["v"]))


def ask(model, prompt):
    body = json.dumps({"model": model, "max_tokens": int(os.environ.get("MAXTOK", "600")), "messages": [{"role": "user", "content": prompt}], "usage": {"include": True}}).encode()
    for k in range(7):
        try:
            rq = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            j = json.load(urllib.request.urlopen(rq, timeout=120)); c = j["choices"][0]["message"]["content"] or ""
            m = re.search(r"\b(YES|NO)\b", c.upper()); u = j.get("usage", {})
            return (m.group(1) == "YES" if m else None), float(u.get("cost") or 0), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        except Exception as e:
            err = str(e)[:100]; time.sleep(3 * (k + 1))   # OpenRouter 429s under load showed up as silent failures
    return None, 0, 0, 0


for model in MODELS:
    with cf.ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda a: ask(model, a[1]), A))
    line = f"{model:38s}"
    for tag in ("natural", "source"):
        Z = [(r[0], a[2]) for a, r in zip(A, res) if a[0] == tag and r[0] is not None]
        tp = sum(p and g for p, g in Z); fp = sum(p and not g for p, g in Z); fn = sum((not p) and g for p, g in Z)
        line += f" | {tag} n{len(Z)} agree {sum(p == g for p, g in Z) / max(1, len(Z)):.3f} prec {tp / max(1, tp + fp):.2f} rec {tp / max(1, tp + fn):.2f}"
    usd = sum(r[1] for r in res); print(line + f" | ${usd / len(A) * 1000:.3f}/1k out {sum(r[3] for r in res) / len(A):.0f} fail {sum(r[0] is None for r in res)}", flush=True)
