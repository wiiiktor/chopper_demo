#!/usr/bin/env python3
"""Natural English for every hop question of the tool-call rows (2026-11-08, user: "do the natural-question rewrite"). The first model learned the
knowledge-base wording of some of them ("What is the founded by of Gagosian Gallery?"); Qwen3-14B-AWQ rewrites each into one natural, self-contained
question that keeps every name, number and date. Already natural questions come back unchanged.   usage: tool_natq.py <hop questions json> <out json>"""
import json, sys, os, re
from vllm import LLM, SamplingParams
Q = json.load(open(sys.argv[1])); print(len(Q), "hop questions", flush=True)
llm = LLM(model="Qwen/Qwen3-14B-AWQ", max_model_len=1024, gpu_memory_utilization=0.9, seed=0)
P = ("Rewrite the question below as one natural, fluent English question, the way a person would ask it. Many inputs are machine-made "
     "(\"What is the X of Y?\" with a database relation as X); turn those into normal English. Keep every name, title, number and date exactly. "
     "Reply with the question only.\n\nExamples:\nWhat is the educated at of David Gelernter? -> Where was David Gelernter educated?\n"
     "What is the founded by of Gagosian Gallery? -> Who founded the Gagosian Gallery?\nWhat is the award of Shaun Tan? -> What award did Shaun Tan receive?\n"
     "What is the named after of Nina Sky? -> What is Nina Sky named after?\nWhat is the place of birth of Thayer Scudder? -> Where was Thayer Scudder born?\n"
     "When was Houston Baptist University founded? -> When was Houston Baptist University founded?\n\n{q} ->")
outs = llm.chat([[{"role": "user", "content": P.format(q=q)}] for q in Q], SamplingParams(temperature=0, max_tokens=60), use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
M = {}
for q, o in zip(Q, outs):
    w = o.outputs[0].text.strip().split("\n")[0].strip().strip('"')
    names = [t for t in re.findall(r"[\w'.-]+", q)[1:] if t[:1].isupper() or t[:1].isdigit()]   # capitalised words and numbers, not the opening question word
    M[q] = w if w and all(t in w for t in names) else q          # a rewrite that drops a name or number keeps the original
json.dump(M, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print("changed", sum(M[q] != q for q in Q), "of", len(Q), list(M.items())[:6], flush=True)
sys.stdout.flush(); os._exit(0)
