#!/usr/bin/env python
"""MiniCPM3-4B LoRA on native tool-call chats (tool_build.py rows). Loss only on the assistant turns (thought, native call or answer, <|im_end|>);
system, tool definition, notes and tool results are context. Recipe of mm_qwen_answer.py: LoRA r32 on MLA + MLP, lr 1e-4 with warm-up and cosine,
accum 8, one epoch, fresh LoRA on the base model so its native tool-call skill is the starting point.
usage: .venv-cpm/bin/python tool_train.py <rows.jsonl> <out dir> [--smoke N]"""
import argparse, json, math, os, random, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
ap = argparse.ArgumentParser(); ap.add_argument("rows"); ap.add_argument("out"); ap.add_argument("--model", default="openbmb/MiniCPM3-4B")
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--r", type=int, default=32)
ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--max-len", type=int, default=2048)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); dev = "cuda"; random.seed(0); torch.manual_seed(0)
LOG = open(os.path.join(A.out, "train.log"), "a")
log = lambda s: (print(time.strftime("%H:%M:%S ") + s, flush=True), LOG.write(time.strftime("%H:%M:%S ") + s + "\n"), LOG.flush())
tk = AutoTokenizer.from_pretrained(A.model, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(A.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev)
base.gradient_checkpointing_enable(); base.enable_input_require_grads()
M = get_peft_model(base, LoraConfig(r=A.r, lora_alpha=A.r, lora_dropout=0.0,
                                    target_modules=["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
render = lambda r, msgs, gen=False: tk.apply_chat_template([{"role": "system", "content": r["system"]}] + msgs, tools=r["tools"], tokenize=False, add_generation_prompt=gen)


def encode(r):
    full = render(r, r["messages"]); spans = []
    for i, m in enumerate(r["messages"]):
        if m["role"] == "assistant":
            a = len(render(r, r["messages"][:i], gen=True)); b = len(render(r, r["messages"][:i + 1]))
            assert full[:b] == render(r, r["messages"][:i + 1]), "template prefix mismatch"
            spans.append((a, b))
    e = tk(full, return_offsets_mapping=True, add_special_tokens=False)
    ids = e["input_ids"][-A.max_len:]; off = e["offset_mapping"][-A.max_len:]
    lab = [t if any(a <= s < b for a, b in spans) else -100 for t, (s, _) in zip(ids, off)]
    return ids, lab


TR = [json.loads(l) for l in open(A.rows)]; random.shuffle(TR)
if A.smoke:
    TR = TR[:A.smoke]
ids, lab = encode(TR[0]); log("first row targets: " + repr(tk.decode([t for t in lab if t != -100])[:400]))
params = [p for p in M.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.0)
log(f"tool-call LoRA: {len(TR)} rows | {sum(p.numel() for p in params) / 1e6:.1f}M trainable | lr {A.lr} accum {A.accum}")
M.train(); t0 = time.time(); win = []; upd = math.ceil(len(TR) / A.accum)
for step, r in enumerate(TR, 1):
    u = (step - 1) // A.accum
    for g in opt.param_groups:
        g["lr"] = A.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u / upd)))
    ids, lab = encode(r); x = torch.tensor([ids], device=dev); y = torch.tensor([lab], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = M(input_ids=x).logits[0, :-1].float()
    yt = y[0, 1:]; ce = torch.nn.functional.cross_entropy(lg, yt, ignore_index=-100)
    (ce / A.accum).backward(); m = yt != -100; win.append((float(ce), float((lg.argmax(-1)[m] == yt[m]).float().mean())))
    if step % A.accum == 0 or step == len(TR):
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step % 200 == 0 or A.smoke:
        w = np.array(win); win.clear()
        log(f"step {step}/{len(TR)} | CE {w[:, 0].mean():.3f} tok {w[:, 1].mean():.3f} | {(time.time() - t0) / step:.2f}s/step | peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
M.save_pretrained(os.path.join(A.out, "lora")); log("saved")
