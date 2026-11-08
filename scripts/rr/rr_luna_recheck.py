#!/usr/bin/env python3
"""Second look at luna's non-seed YES on specific rows (2026-11-05). rr_luna_label.py showed luna the passage text only, so a query
naming "greenfingers.com" or "wine.com" could not be told apart from the same text on another site: of 14 hand-read low-score YES
on English rows most were another organisation's page. Here luna also sees the passage's source (domain/title, or synthetic
title/section). Resumable. Writes <dir>/luna_v2_recheck_<lang>.jsonl {rid, i, v}; a False overrides the first verdict.
usage: rr_luna_recheck.py <dir> [lang ...]   (env THREADS=32)"""
import json, os, re, sys, time
import concurrent.futures as cf
from openai import OpenAI
D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]; TH = int(os.environ.get("THREADS", "32"))
cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
P = """Query: {q}

Passage source: {src}
Passage:
{text}

The query names or clearly implies a specific organisation, website, product, place or case. Does this passage state the answer
FOR THAT ONE? Use the source line and the text. If the source or text shows a different organisation, website or product, or
gives no sign it is the one asked about, reply NO. For multi-part queries, the answer to at least one part is enough.
Reply with one word: YES or NO."""


def ask(x):
    for k in range(4):
        try:
            t = cl.chat.completions.create(model="gpt-5.6-luna", max_completion_tokens=400, messages=[{"role": "user", "content": x["p"]}]).choices[0].message.content
            m = re.search(r"\b(YES|NO)\b", (t or "").upper())
            if m:
                return m.group(1) == "YES"
        except Exception as e:
            print("err", type(e).__name__, flush=True); time.sleep(5 * (k + 1))
    return None


for lang in LANGS:
    out = f"{D}/luna_v2_recheck_{lang}.jsonl"
    done = {(x["rid"], x["i"]) for x in map(json.loads, open(out))} if os.path.exists(out) else set()
    Y = {(x["rid"], x["i"]): x["how"] for x in map(json.loads, open(f"{D}/luna_v2_{lang}.jsonl")) if x["v"] and x["how"] != "seed"}
    todo = []
    for r in map(json.loads, open(f"{D}/rows_v2_{lang}.jsonl")):
        if not r["specific"]:
            continue
        for i, p in enumerate(r["passages"]):
            if (r["rid"], i) in Y and (r["rid"], i) not in done:
                m = p["meta"]; src = " | ".join(str(m[k]) for k in ("domain", "title", "section") if m.get(k)) or "unknown"
                todo.append({"rid": r["rid"], "i": i, "how": Y[(r["rid"], i)], "p": P.format(q=r["query"], src=src, text=p["text"])})
    with cf.ThreadPoolExecutor(TH) as ex, open(out, "a") as f:
        res = list(ex.map(ask, todo))
        for x, v in zip(todo, res):
            if v is not None:
                f.write(json.dumps({"rid": x["rid"], "i": x["i"], "how": x["how"], "v": v}) + "\n")
    n = len(todo); print(f"{lang}: {n} rechecked | still YES rr {sum(v is True and x['how'] == 'rr' for x, v in zip(todo, res))}/{sum(x['how'] == 'rr' for x in todo)} "
                         f"audit {sum(v is True and x['how'] == 'audit' for x, v in zip(todo, res))}/{sum(x['how'] == 'audit' for x in todo)}", flush=True)
