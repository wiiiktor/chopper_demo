#!/usr/bin/env python
"""One answer round of the native tool-call loop (2026-11-08). For every open conversation in <state>: MiniCPM3-4B (+ LoRA) continues it greedily; a native
search(query=...) call is recorded and left open for tool_eval_fetch.py, anything else is the final answer. On the last round a call is not allowed
(<|tool_call_start|> banned), so every question ends with an answer.   usage: tool_eval_answer.py <state.jsonl> <lora dir|none> [--last]"""
import json, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ST, LORA = sys.argv[1], sys.argv[2]; LAST = "--last" in sys.argv; M_ = "openbmb/MiniCPM3-4B"
tk = AutoTokenizer.from_pretrained(M_, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(M_, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
if LORA != "none":
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, LORA).eval()
BAN = [[tk.convert_tokens_to_ids("<|tool_call_start|>")]]
CALL = re.compile(r'search\(\s*query\s*=\s*(["\'])(.*?)\1\s*\)', re.S)
S = [json.loads(l) for l in open(ST)]; n_call = n_ans = 0
for s in S:
    if s.get("done") or s.get("pending"):
        continue
    p = tk.apply_chat_template([{"role": "system", "content": s["system"]}] + s["messages"], tools=s["tools"], tokenize=False, add_generation_prompt=True)
    ids = tk(p, return_tensors="pt", add_special_tokens=False).input_ids[:, -3000:].cuda()
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=150, do_sample=False, bad_words_ids=BAN if LAST else None, eos_token_id=tk.convert_tokens_to_ids("<|im_end|>"))
    t = tk.decode(o[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "")
    th = re.search(r"<\|thought_start\|>\s*(.*?)\s*<\|thought_end\|>", t, re.S); th = th.group(1) if th else ""
    c = CALL.search(t)
    if c and not LAST:
        q = c.group(2).strip()
        s["messages"].append({"role": "assistant", "thought": th, "content": "", "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": q}}}]})
        s["pending"] = q; s.setdefault("queries", []).append(q); n_call += 1
    else:
        a = re.sub(r"<\|thought_start\|>.*?<\|thought_end\|>", "", t, flags=re.S)
        a = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", a, flags=re.S).strip()
        s["messages"].append({"role": "assistant", "thought": th, "content": a}); s["done"] = True; s["pred"] = a; n_ans += 1
with open(ST, "w") as f:
    f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in S)
print(f"round: {n_call} calls, {n_ans} answers, open {sum(not s.get('done') for s in S)}", flush=True)
