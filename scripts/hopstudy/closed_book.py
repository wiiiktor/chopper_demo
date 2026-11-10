#!/usr/bin/env python3
"""Closed-book check (user 2026-11-10: "opus may simply know a lot more answers ... questions that opus replies based on model
knowledge should not be in comparison"): ask a model every hop-study question with NO passages; gpt-5.6-luna judges the reply
against the gold answer (+ aliases). Writes <hs dir>/closed_<tag>.json {row: true/false}; known rows can be dropped from comparisons.
usage: closed_book.py <hs dir> [--model anthropic/claude-opus-5.5] [--tag opus]"""
import concurrent.futures as cf, json, os, sys, time
sys.path.insert(0, "/home/wii/Projects/certain/scripts")
from judge_mh_luna import QA, ask as judge
from openai import OpenAI
HD = sys.argv[1]; MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "anthropic/claude-opus-5.5"
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "opus"
R = json.load(open(f"{HD}/rows.json"))
orc = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open("/home/wii/Projects/certain/.openrouter_key").read().strip())
oai = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
st = {"in": 0, "out": 0, "unparsed": 0, "errors": 0}


def answer(r):
    for k in range(5):
        try:
            x = orc.chat.completions.create(model=MODEL, max_tokens=200, temperature=0, messages=[{"role": "user", "content":
                f"Answer from your own knowledge, as briefly as possible (a name, date or number). If you do not know, guess.\n\nQuestion: {r['q']}"}])
            return x.choices[0].message.content or ""
        except Exception as e:
            print("err", str(e)[:120], flush=True); time.sleep(10 * (k + 1))
    return ""


with cf.ThreadPoolExecutor(4) as ex:
    A = list(ex.map(answer, R))
with cf.ThreadPoolExecutor(16) as ex:
    V = list(ex.map(lambda ra: judge(oai, QA.format(q=ra[0]["q"], gold=ra[0]["a"], alias=(" (also: " + "; ".join(ra[0]["aliases"]) + ")") if ra[0]["aliases"] else "",
                                                     pred=ra[1]), st), zip(R, A)))
out = {r["row"]: bool(v) for r, v in zip(R, V)}
json.dump({"model": MODEL, "known": out, "answers": {r["row"]: a for r, a in zip(R, A)}}, open(f"{HD}/closed_{TAG}.json", "w"), indent=0)
for k in (1, 2, 3, 4):
    rr = [r for r in R if r["hops"] == k]
    print(f"{k}-hop: answered correctly with NO passages {sum(out[r['row']] for r in rr)}/{len(rr)}")
