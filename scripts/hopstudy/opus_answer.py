#!/usr/bin/env python3
"""Hop study, answer stage with a strong answerer (user 2026-11-10: "answers should be done by opus 5.5 for both models - no need to
lower results for chopper by using small model as answerer"). For each notes source (st_<arm>.jsonl from hs_states.py) Opus 5.5 answers
from the notes only; gpt-5.6-luna judges against the gold answer + aliases. Cached per (arm, row) in <hs>/opus_answers.jsonl.
usage: opus_answer.py <hs dir> arm [arm ...]   (env THREADS=4)"""
import concurrent.futures as cf, json, os, sys, time
sys.path.insert(0, "/home/wii/Projects/certain/scripts")
from judge_mh_luna import QA, ask as judge
from openai import OpenAI
HD, ARMS = sys.argv[1], sys.argv[2:]; TH = int(os.environ.get("THREADS", "4"))
R = {r["row"]: r for r in json.load(open(f"{HD}/rows.json"))}
orc = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open("/home/wii/Projects/certain/.openrouter_key").read().strip())
oai = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
CF = f"{HD}/opus_answers.jsonl"; C = {}
if os.path.exists(CF):
    C = {(x["arm"], x["row"]): x for x in map(json.loads, open(CF))}
INS = ("Answer the question using ONLY the notes. Reply with just the answer (a name, date, number or short phrase). "
       "If the notes do not contain enough to answer, reply exactly: not in notes.\n\n")
st = {"in": 0, "out": 0, "unparsed": 0, "errors": 0}


def one(job):
    arm, s = job; r = R[s["q"]["qid"]]
    for k in range(5):
        try:
            x = orc.chat.completions.create(model="anthropic/claude-opus-5.5", max_tokens=2000, temperature=0,   # Opus 5.5 reasons first: 100 tokens left 53% of replies EMPTY
                                           
                                            messages=[{"role": "user", "content": INS + s["messages"][0]["content"]}])
            a = (x.choices[0].message.content or "").strip(); break
        except Exception as e:
            print("err", str(e)[:100], flush=True); time.sleep(10 * (k + 1)); a = ""
    al = (" (also: " + "; ".join(r["aliases"]) + ")") if r["aliases"] else ""
    v = judge(oai, QA.format(q=r["q"], gold=r["a"], alias=al, pred=a or "(no answer)"), st)
    return {"arm": arm, "row": r["row"], "hops": r["hops"], "pred": a, "ok": bool(v)}


jobs = [(arm, s) for arm in ARMS for s in map(json.loads, open(f"{HD}/st_{arm}.jsonl")) if (arm, s["q"]["qid"]) not in C]
print(f"{len(jobs)} answers to get", flush=True)
with cf.ThreadPoolExecutor(TH) as ex, open(CF, "a") as f:
    for x in ex.map(one, jobs):
        C[(x["arm"], x["row"])] = x; f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
for arm in ARMS:
    print(arm, " | ".join(f"{k}-hop {sum(C[(arm, n)]['ok'] for n, r in R.items() if r['hops'] == k and (arm, n) in C)}/100" for k in (1, 2, 3, 4)))
