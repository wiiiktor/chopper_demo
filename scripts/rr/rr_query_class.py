#!/usr/bin/env python3
"""Query filter for the rerank pilot (2026-11-05, user: questions answerable from common knowledge should go). gpt-5.6-luna sorts every
v2 query from its text alone: COMMON (answerable with no documents), VAGUE (points at something unnamed, "the company", so it cannot
be understood alone), SOURCE (needs a document). Resumable. Writes <dir>/qclass_v2_<lang>.jsonl {rid, c}.
usage: rr_query_class.py <dir> [lang ...]   (env N=rows per language, THREADS=16)"""
import json, os, re, sys, time
import concurrent.futures as cf
from collections import Counter
from openai import OpenAI
D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]
N = int(os.environ.get("N", "0")); TH = int(os.environ.get("THREADS", "16"))
cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
P = """Search query: {q}

Sort this query into one class, judging only its text:
COMMON = a well-informed person could answer it correctly with no documents: definitions, widely known facts, general advice.
VAGUE = it points at something it does not name ("the company", "the customer", "this product", "the program"), so nobody could tell what is meant without seeing the page it came from.
SOURCE = it asks for specific information that has to be looked up in a document: a named organisation's or product's details, figures, rules, deadlines, procedures, or a specific topic where documents differ.
Reply with one word: COMMON, VAGUE or SOURCE."""


def ask(x):
    for k in range(4):
        try:
            t = cl.chat.completions.create(model="gpt-5.6-luna", max_completion_tokens=300, messages=[{"role": "user", "content": P.format(q=x["query"])}]).choices[0].message.content
            m = re.search(r"\b(COMMON|VAGUE|SOURCE)\b", (t or "").upper())
            if m:
                return m.group(1)
        except Exception as e:
            print("err", type(e).__name__, flush=True); time.sleep(5 * (k + 1))
    return None


for lang in LANGS:
    out = f"{D}/qclass_v2_{lang}.jsonl"
    done = {x["rid"] for x in map(json.loads, open(out))} if os.path.exists(out) else set()
    R = [{"rid": r["rid"], "query": r["query"]} for r in map(json.loads, open(f"{D}/rows_v2_{lang}.jsonl")) if r["rid"] not in done][:N or None]
    with cf.ThreadPoolExecutor(TH) as ex, open(out, "a") as f:
        res = list(ex.map(ask, R))
        for x, c in zip(R, res):
            if c:
                f.write(json.dumps({"rid": x["rid"], "c": c}) + "\n")
    print(lang, len(R), Counter(res), flush=True)
