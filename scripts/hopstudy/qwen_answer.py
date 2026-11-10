#!/usr/bin/env python3
"""Hop study, answer stage with Qwen3-14B-AWQ (user 2026-11-10: "test qwen 14b with chopper vs qwen 14b without chopper; does qwen need
fine tuning?"). Zero-shot, same instruction as opus_answer.py: answer from the notes only. Arms = st_<arm>.jsonl (raw = all 20
paragraphs, i.e. no chopper). Records prompt tokens per arm (the chopper's saving). gpt-5.6-luna judges against gold + aliases.
Writes <hs>/qwen_answers_<mode>.jsonl.   usage: qwen_answer.py <hs dir> arm ... (env THINK=0|1, GPU_UTIL=0.45)"""
import concurrent.futures as cf, json, os, sys, time
sys.path.insert(0, "/home/wii/Projects/certain/scripts")
HD, ARMS = sys.argv[1], sys.argv[2:]; THINK = os.environ.get("THINK", "0") == "1"
R = {r["row"]: r for r in json.load(open(f"{HD}/rows.json"))}
INS = ("Answer the question using ONLY the notes. Reply with just the answer (a name, date, number or short phrase). "
       "If the notes do not contain enough to answer, reply exactly: not in notes.\n\n")
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-14B-AWQ", max_model_len=8192, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.45")), seed=0)
sp = SamplingParams(temperature=0, max_tokens=2048 if THINK else 64)
jobs = [(arm, s) for arm in ARMS for s in map(json.loads, open(f"{HD}/st_{arm}.jsonl"))]
t0 = time.time()
outs = llm.chat([[{"role": "user", "content": INS + s["messages"][0]["content"]}] for _, s in jobs], sp, use_tqdm=False,
                chat_template_kwargs={"enable_thinking": THINK})
dt = time.time() - t0
res = []
for (arm, s), o in zip(jobs, outs):
    a = o.outputs[0].text.split("</think>")[-1].strip()
    res.append({"arm": arm, "row": s["q"]["qid"], "hops": R[s["q"]["qid"]]["hops"], "pred": a, "ptok": len(o.prompt_token_ids)})
from judge_mh_luna import QA, ask as judge
from openai import OpenAI
oai = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip()); st = {"in": 0, "out": 0, "unparsed": 0, "errors": 0}


def j(x):
    r = R[x["row"]]; al = (" (also: " + "; ".join(r["aliases"]) + ")") if r["aliases"] else ""
    x["ok"] = bool(judge(oai, QA.format(q=r["q"], gold=r["a"], alias=al, pred=x["pred"] or "(no answer)"), st)); return x


with cf.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(j, res))
mode = "think" if THINK else "nothink"
open(f"{HD}/qwen_answers_{mode}.jsonl", "w").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in res)
print(f"generation {dt:.0f}s for {len(jobs)} answers", flush=True)
for arm in ARMS:
    xs = [x for x in res if x["arm"] == arm]
    print(f"{arm:10s} " + " | ".join(f"{k}-hop {sum(x['ok'] for x in xs if x['hops'] == k)}/100" for k in (1, 2, 3, 4))
          + f" | prompt tokens {sum(x['ptok'] for x in xs) / len(xs):.0f}", flush=True)
sys.stdout.flush(); os._exit(0)
