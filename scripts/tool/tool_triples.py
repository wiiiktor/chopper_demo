#!/usr/bin/env python3
"""MuSiQue's hop-1 sub-questions are often Wikidata triples ("David Gelernter >> educated at"); Qwen3-14B-AWQ rewrites each into a natural, self-contained
question ("Where was David Gelernter educated?") for the native tool-call rows. Writes {triple: question}.   usage: tool_triples.py <out json>"""
import json, sys, os
from vllm import LLM, SamplingParams
T = set()
for f in ("train", "dev"):
    for l in open(f"/media/wii/bigdata/musique_raw/musique_ans_v1.0_{f}.jsonl"):
        for h in json.loads(l)["question_decomposition"]:
            if ">>" in h["question"]:
                T.add(h["question"])
T = sorted(T); print(len(T), "triples", flush=True)
llm = LLM(model="Qwen/Qwen3-14B-AWQ", max_model_len=1024, gpu_memory_utilization=0.9, seed=0)
P = 'Rewrite this knowledge-base lookup as one natural, self-contained English question (subject >> relation). Reply with the question only.\n\n{t}'
outs = llm.chat([[{"role": "user", "content": P.format(t=t)}] for t in T], SamplingParams(temperature=0, max_tokens=40), use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
M = {t: o.outputs[0].text.strip().split("\n")[0] for t, o in zip(T, outs)}
json.dump(M, open(sys.argv[1], "w"), ensure_ascii=False, indent=0); print(list(M.items())[:5], flush=True)
sys.stdout.flush(); os._exit(0)
