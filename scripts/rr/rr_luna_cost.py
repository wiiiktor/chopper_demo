#!/usr/bin/env python3
"""1,000-pair gpt-5.6-luna cost check for the rerank-pilot labels (user 2026-11-05: "run the 1,000-pair luna cost
check"). Pairs = (query, passage) with Qwen3-Reranker-0.6B p > 0.3 from the v1 rows, all 8 languages, natural mix
(v2 rows have no reranker scores yet; per-pair cost depends on passage length, which v2 does not change).
Same YES/NO prompt as rr_luna_gold.py. Prints tokens and $ per pair at $0.20 / $1.20 per 1M in / out.
usage: rr_luna_cost.py <rr dir> <out jsonl> [N]"""
import json, os, random, re, sys
import concurrent.futures as cf
from openai import OpenAI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D, OUT = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
P = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_luna_gold.py")).read().split('P = """')[1].split('"""')[0]
cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
rng = random.Random(0); pairs = []
for lang in ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]:
    RR = json.load(open(f"{D}/rr_{lang}.json"))
    for r in map(json.loads, open(f"{D}/rows_{lang}.jsonl")):
        for i, p in enumerate(r["passages"]):
            if RR[r["rid"]][i] > 0.3:
                pairs.append({"rid": r["rid"], "i": i, "lang": lang, "qtype": r["qtype"], "seed": i == r["seed"],
                              "q": r["query"], "text": p["text"], "rr": RR[r["rid"]][i]})
S = rng.sample(pairs, N)
print(f"{len(pairs)} candidate pairs with p > 0.3 in v1; sampling {N}", flush=True)


def ask(x):
    for _ in range(3):
        try:
            r = cl.chat.completions.create(model="gpt-5.6-luna", max_completion_tokens=400,
                                           messages=[{"role": "user", "content": P.format(q=x["q"], text=x["text"])}])
            m = re.search(r"\b(YES|NO)\b", (r.choices[0].message.content or "").upper())
            return (m.group(1) == "YES" if m else None), r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:
            print("err", e, flush=True)
    return None, 0, 0


with cf.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(ask, S))
tin = sum(r[1] for r in res); tout = sum(r[2] for r in res)
with open(OUT, "w") as f:
    for x, (v, a, b) in zip(S, res):
        f.write(json.dumps({k: x[k] for k in ("rid", "i", "lang", "qtype", "seed", "rr")} | {"luna": v, "in": a, "out": b}) + "\n")
usd = (tin * 0.20 + tout * 1.20) / 1e6
print(f"{N} pairs | tokens in {tin} out {tout} | ${usd:.3f} = ${usd / N * 1000:.3f} per 1k pairs | unparsed {sum(r[0] is None for r in res)} "
      f"| YES {sum(r[0] is True for r in res) / N:.3f} | full-run estimate for {len(pairs)} pairs: ${usd / N * len(pairs):.0f}", flush=True)
