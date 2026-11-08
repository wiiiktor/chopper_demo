#!/usr/bin/env python3
"""Step 4b of the rerank dataset pilot (user 2026-11-05: "ok use luna"): gpt-5.6-luna decides, per (query, passage),
whether the passage itself states the answer. Every Qwen labeller said yes to ~90% of reads (precision 0.44-0.47 against
luna, rr-dataset-pilot memory); luna costs $0.14 per 1k pairs. Read: the seed, every passage with Qwen3-Reranker-0.6B
p > TAU, and a 5% random audit of the rest (its YES rate = what the prefilter misses). Unread passages count as NO.
Resumable: verdicts are appended to <dir>/luna_<V>_<lang>.jsonl {rid, i, how, v} and skipped on a re-run.
usage: rr_luna_label.py <dir> [lang ...]   (env V=v2 TAU=0.3 THREADS=16)"""
import json, os, random, re, sys, threading, time
import concurrent.futures as cf
from openai import OpenAI

D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]
V = os.environ.get("V", "v2"); TAU = float(os.environ.get("TAU", "0.3")); TH = int(os.environ.get("THREADS", "16"))
P = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_luna_gold.py")).read().split('P = """')[1].split('"""')[0]
cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
lock = threading.Lock(); tok = [0, 0]


def ask(x):
    for k in range(4):
        try:
            r = cl.chat.completions.create(model="gpt-5.6-luna", max_completion_tokens=400,
                                           messages=[{"role": "user", "content": P.format(q=x["q"], text=x["text"])}])
            with lock:
                tok[0] += r.usage.prompt_tokens; tok[1] += r.usage.completion_tokens
            m = re.search(r"\b(YES|NO)\b", (r.choices[0].message.content or "").upper())
            if m:
                return m.group(1) == "YES"
        except Exception as e:
            print("err", type(e).__name__, str(e)[:120], flush=True); time.sleep(5 * (k + 1))
    return None


for lang in LANGS:
    rng = random.Random(lang)
    out = f"{D}/luna_{V}_{lang}.jsonl"
    done = {(x["rid"], x["i"]) for x in map(json.loads, open(out))} if os.path.exists(out) else set()
    RR = json.load(open(f"{D}/rr_{V}_{lang}.json"))
    todo = []
    for r in map(json.loads, open(f"{D}/rows_{V}_{lang}.jsonl")):
        for i, p in enumerate(r["passages"]):
            s = RR[r["rid"]][i]
            how = "seed" if i == r["seed"] else "rr" if s > TAU else "audit" if rng.random() < 0.05 else None
            if how and (r["rid"], i) not in done:
                todo.append({"rid": r["rid"], "i": i, "how": how, "q": r["query"], "text": p["text"]})
    t0 = time.time(); n = 0
    with cf.ThreadPoolExecutor(TH) as ex, open(out, "a") as f:
        for x, v in zip(todo, ex.map(ask, todo)):
            if v is not None:                         # unparsed / failed -> retried on the next run
                f.write(json.dumps({"rid": x["rid"], "i": x["i"], "how": x["how"], "v": v}) + "\n"); n += 1
            if n % 2000 == 0:
                f.flush()
    print(f"{lang}: {n}/{len(todo)} new verdicts in {time.time() - t0:.0f}s | running tokens in {tok[0]} out {tok[1]} "
          f"= ${(tok[0] * 0.20 + tok[1] * 1.20) / 1e6:.2f}", flush=True)
