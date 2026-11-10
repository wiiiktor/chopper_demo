#!/usr/bin/env python3
"""The chopper teacher (mm_qwen_labels.py: same prompt, same quote locating, same output format) run on a strong API model through
OpenRouter, to measure how much better labels a near-perfect teacher would give (user 2026-11-09: "compare qwen teacher quality to
some openrouter strong model"). Resumable (cached per row in <out>.raw.jsonl).
usage: teacher_api.py <ctx.jsonl> <out.jsonl> [--model anthropic/claude-opus-5.5] [--threads 8]"""
import concurrent.futures as cf, json, os, re, sys, time
CTX, OUT = sys.argv[1], sys.argv[2]
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "anthropic/claude-opus-5.5"
TH = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 8
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mm_qwen_labels.py")).read().split("from vllm import")[0]
sys.argv = ["x", CTX, OUT]; exec(SRC)                             # D, prompt(), norm_map(), locate()
from openai import OpenAI
cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open("/home/wii/Projects/certain/.openrouter_key").read().strip())
RAW = OUT + ".raw.jsonl"; done = {}
if os.path.exists(RAW):
    done = {(x.get("dataset", "hs"), x["row"]): x for x in map(json.loads, open(RAW))}
tok = [0, 0]


BLIND = "--blind" in sys.argv                                   # answer-blind teacher: no gold answer in the prompt, it must work the answer out


def P(d):
    t = prompt(d)
    if BLIND:
        t = t.replace(f"Correct answer: {d['answer']}\n", "").replace("to the correct answer", "to its answer (work the answer out from the passages)")
    return t


def ask(d):
    for k in range(5):
        try:
            r = cl.chat.completions.create(model=MODEL, max_tokens=2000, temperature=0,
                                           messages=[{"role": "user", "content": P(d)}])
            tok[0] += r.usage.prompt_tokens; tok[1] += r.usage.completion_tokens
            return r.choices[0].message.content or ""
        except Exception as e:
            print("err", type(e).__name__, str(e)[:150], flush=True); time.sleep(10 * (k + 1))
    return None


K = lambda d: (d["dataset"] if not d["dataset"].startswith("hop") else "hs", d["row"])   # val rows repeat row numbers across datasets
todo = [d for d in D if K(d) not in done]
t0 = time.time()
with cf.ThreadPoolExecutor(TH) as ex, open(RAW, "a") as fr:
    for d, txt in zip(todo, ex.map(ask, todo)):
        if txt is not None:
            done[K(d)] = {"dataset": K(d)[0], "row": d["row"], "txt": txt}; fr.write(json.dumps(done[K(d)], ensure_ascii=False) + "\n"); fr.flush()
bad = nq = nmiss = 0
with open(OUT, "w") as fo:
    for d in D:
        txt = done.get(K(d), {}).get("txt", "")
        try:
            steps = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)).get("steps") or []
        except Exception:
            bad += 1; steps = []
        NMs = [norm_map(p["text"]) for p in d["passages"]]; spans, miss = set(), []
        for s in steps:
            for qt in (s.get("quotes") or []) if isinstance(s, dict) else []:
                nq += 1; found = False
                for k, p in enumerate(d["passages"]):
                    for c0, c1 in locate(p["text"], NMs[k], str(qt)):
                        spans.add((k, c0, c1)); found = True
                if not found:
                    miss.append(qt); nmiss += 1
        fo.write(json.dumps({"dataset": d["dataset"], "split": d["split"], "row": d["row"], "spans": sorted(spans), "steps": steps, "miss": miss},
                            ensure_ascii=False) + "\n")
print(f"[{MODEL}] {len(D)} rows ({len(todo)} new) in {(time.time() - t0) / 60:.1f} min | unparsable {bad} | quotes {nq}, not found {nmiss} | "
      f"tokens in {tok[0]} out {tok[1]}", flush=True)
