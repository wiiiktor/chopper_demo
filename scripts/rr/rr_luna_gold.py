#!/usr/bin/env python3
"""gpt-5.6-luna gold for the rerank-pilot labeller (2026-11-05): every Qwen labeller prompt (v1-v3, 14B and 4B) graded
~80-95% of near-neighbour passages as answers. luna judges a stratified sample of read (query, passage) pairs from the
smoke rows; the labellers and the reranker are scored against it.
usage: rr_luna_gold.py <rr dir> <lab smoke file> <out jsonl> [N]"""
import json, os, random, re, sys
import concurrent.futures as cf
from openai import OpenAI
D, LAB, OUT = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 160
RR = json.load(open(f"{D}/rr_pl.json"))
cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
P = """Query: {q}

Passage:
{text}

Does this passage itself state the answer to the query (for multi-part queries: the answer to at least one part)? If the
query names or clearly implies a specific organisation, product, place or case, the passage must be about that one.
Being on the same topic, sharing words, or answering a related but different question does NOT count.
Reply with one word: YES or NO."""
rng = random.Random(0); pairs = []
for r in map(json.loads, open(LAB)):
    for i, p in enumerate(r["passages"]):
        if p["how"] in ("rr", "audit"):
            pairs.append({"rid": r["rid"], "i": i, "qtype": r["qtype"], "q": r["query"], "text": p["text"], "rr": RR[r["rid"]][i]})
B = {}
for x in pairs:
    B.setdefault("<0.3" if x["rr"] < .3 else "0.3-0.7" if x["rr"] < .7 else "0.7-0.95" if x["rr"] < .95 else ">=0.95", []).append(x)
S = [x for b in B.values() for x in rng.sample(b, min(len(b), N // 4))]


def ask(x):
    for _ in range(3):
        try:
            t = cl.chat.completions.create(model="gpt-5.6-luna", max_completion_tokens=400,
                                           messages=[{"role": "user", "content": P.format(q=x["q"], text=x["text"])}]).choices[0].message.content
            m = re.search(r"\b(YES|NO)\b", t.upper())
            if m:
                return m.group(1) == "YES"
        except Exception as e:
            print("err", e, flush=True)
    return None


with cf.ThreadPoolExecutor(16) as ex:
    for x, v in zip(S, ex.map(ask, S)):
        x["luna"] = v
with open(OUT, "w") as f:
    f.writelines(json.dumps({k: x[k] for k in ("rid", "i", "qtype", "rr", "luna")}) + "\n" for x in S)
print(len(S), "judged;", sum(x["luna"] is None for x in S), "unparsed")
