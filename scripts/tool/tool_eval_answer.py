#!/usr/bin/env python
"""One answer round of the native tool-call loop (2026-11-08). For every open conversation in <state>: MiniCPM3-4B (+ LoRA) continues it greedily; a native
search(query=...) call is recorded and left open for tool_eval_fetch.py, anything else is the final answer. On the last round a call is not allowed
(<|tool_call_start|> banned), so every question ends with an answer.   usage: tool_eval_answer.py <state.jsonl> <lora dir|none> [--last] [--batch N]   (2026-11-08: batched, left-padded; --batch 1 = the old one-by-one path)"""
import json, os, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ST, LORA = sys.argv[1], sys.argv[2]; LAST = "--last" in sys.argv; M_ = "openbmb/MiniCPM3-4B"
tk = AutoTokenizer.from_pretrained(M_, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(M_, trust_remote_code=True, torch_dtype=torch.bfloat16).to(os.environ.get("DEV", "cuda")).eval()
if LORA != "none":
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, LORA).eval()
BAN = [[tk.convert_tokens_to_ids("<|tool_call_start|>")]]
CALL = re.compile(r'search\(\s*query\s*=\s*(["\'])(.*?)\1\s*\)', re.S)
BS = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 16
tk.padding_side = "left"
if tk.pad_token is None:
    tk.pad_token = tk.unk_token or tk.eos_token
END = tk.convert_tokens_to_ids("<|im_end|>")
S = [json.loads(l) for l in open(ST)]; n_call = n_ans = 0
OPEN = [s for s in S if not (s.get("done") or s.get("pending"))]
OPEN.sort(key=lambda s: len(json.dumps(s["messages"])))           # similar lengths together, less padding
GEN = {}
for i in range(0, len(OPEN), BS):
    B = OPEN[i:i + BS]
    P = [tk.apply_chat_template([{"role": "system", "content": s["system"]}] + s["messages"], tools=s["tools"], tokenize=False, add_generation_prompt=True) for s in B]
    enc = tk(P, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=int(os.environ.get("MAXLEN", "8192"))).to(os.environ.get("DEV", "cuda"))
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=150, do_sample=False, bad_words_ids=BAN if LAST else None, eos_token_id=END, pad_token_id=tk.pad_token_id)
    for s, row in zip(B, o[:, enc.input_ids.shape[1]:]):
        GEN[id(s)] = tk.decode(row, skip_special_tokens=False).split("<|im_end|>")[0].replace(tk.pad_token, "")
for s in OPEN:
    t = GEN[id(s)]
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
