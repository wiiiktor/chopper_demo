#!/usr/bin/env python
"""QWEN answers from FRAGMENTS, or writes "[search] <sub-question>" (2026-10-29, user: "migrating to some qwen model for
everything apart from initial retrieval … test training with a smaller qwen"; the RWKV version is mm_rwkv_answer.py).
Qwen3 (1.7B by default) + LoRA (peft, r 32 on attention + MLP) reads ONLY the kept fragments and writes, ending with EOS:
    Notes:\n- <fragment>\n- …\n\nQuestion: <q>\nAnswer: <answer | [search] <sub-question>>
Training: the train contexts (TRAIN_SPLITS) with QWEN3-14B's fragments; target = the answer when every hop's gold
passage was among the 16 (ctx allin), else "[search] " + the sub-question Qwen3-14B wrote from those fragments WITHOUT
the answer (mm_subq.py --train). One epoch.
Eval (val), three inputs: Qwen's fragments, the mmBERT chopper's fragments (preds), no notes. Greedy generation →
decision recall per gold class, exact / span on answer-class questions, teacher-forced CE / tok; every prediction
(incl. the written sub-questions) is saved to preds_<step>.jsonl for the search step.
usage: [TRAIN_SPLITS=train,train2] .venv-gen/bin/python mm_qwen_answer.py <mm dir> <chopper preds_val.jsonl> <out dir>
       --subq <subq_train.jsonl>,<subq_train2.jsonl> [--model Qwen/Qwen3-1.7B] [--lr 1e-4] [--accum 8]"""
import argparse, json, math, os, random, re, time
import numpy as np, torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("mm"); ap.add_argument("preds"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--subq", default="", help="comma list of mm_subq.py --train outputs: the [search] targets get the sub-question")
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--r", type=int, default=32); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--max-notes", type=int, default=1500, help="max note tokens")
ap.add_argument("--init", default="", help="LoRA dir (save_pretrained) to CONTINUE from; fresh optimiser and schedule")
ap.add_argument("--eval-every", type=int, default=0, help="extra val evals every N rows (0 = only at the end)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); dev = "cuda"; random.seed(0); torch.manual_seed(0)
LOG = open(os.path.join(A.out, "train.log"), "a")


def log(s):
    s = time.strftime("%H:%M:%S ") + s; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
tk = AutoTokenizer.from_pretrained(A.model)
base = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16).to(dev)
base.gradient_checkpointing_enable(); base.enable_input_require_grads()
M = get_peft_model(base, LoraConfig(r=A.r, lora_alpha=A.r, lora_dropout=0.0,
                                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
if A.init:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(M, load_file(os.path.join(A.init, "adapter_model.safetensors")))
STOP = tk.eos_token_id; SEARCH = "[search]"
SUBQ = {}
for f_ in filter(None, A.subq.split(",")):
    for x in map(json.loads, open(f_)):
        SUBQ[(x["dataset"], x["row"])] = x.get("fragments", "")


def frags_of(c, spans):
    return [c["passages"][k]["text"][a:b].strip() for k, a, b in spans]


def prompt(q, notes):
    s = "Notes:\n" + ("\n".join(f"- {' '.join(n.split())}" for n in notes) if notes else "(none)") + f"\n\nQuestion: {q}\nAnswer:"
    ids = tk.encode(s)
    if len(ids) > A.max_notes + 64:                    # keep the question; cut the notes from the front
        ids = ids[-(A.max_notes + 64):]
    return ids


def target(c):
    if c["allin"] == 1.0:
        return tk.encode(" " + c["answer"]) + [STOP]
    sq = SUBQ.get((c["dataset"], c["row"]), "")
    return tk.encode(" " + SEARCH + (" " + sq if sq else "")) + [STOP]


def load(split):
    C = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{A.mm}/ctx_{split}.jsonl"))}
    L = {(q["dataset"], q["row"]): q for q in map(json.loads, open(f"{A.mm}/qw_{split}.jsonl"))}
    return C, L


CT, LT = {}, {}
for sp_ in os.environ.get("TRAIN_SPLITS", "train,train2").split(","):
    c_, l_ = load(sp_); CT.update(c_); LT.update(l_)
CV, LV = load("val")
PRD = {(q["dataset"], q["row"]): q for q in map(json.loads, open(A.preds))} if os.path.exists(A.preds) else {}
TR = [(CT[k], frags_of(CT[k], LT[k]["spans"])) for k in CT if k in LT]; random.shuffle(TR)
if A.smoke:
    TR = TR[:A.smoke]
params = [p for p in M.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.0)
log(f"answerer ({A.model}): {len(TR)} train rows | [search] share {np.mean([c['allin'] != 1.0 for c, _ in TR]):.2f} | "
    f"LoRA r {A.r}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable | lr {A.lr} accum {A.accum}")


def tf_loss(ids, y):
    x = torch.tensor([ids + y[:-1]], device=dev)
    lg = M(input_ids=x, logits_to_keep=len(y)).logits[0].float(); yt = torch.tensor(y, device=dev)   # only the target logits
    return F.cross_entropy(lg, yt), (lg.argmax(-1) == yt).float()


@torch.no_grad()
def generate(ids, maxlen=40):
    o = M(input_ids=torch.tensor([ids], device=dev), use_cache=True); c = o.past_key_values; out = []
    t_ = int(o.logits[0, -1].argmax())
    for _ in range(maxlen):
        if t_ == STOP:
            break
        out.append(t_); o = M(input_ids=torch.tensor([[t_]], device=dev), past_key_values=c, use_cache=True)
        c = o.past_key_values; t_ = int(o.logits[0, -1].argmax())
    return tk.decode(out).strip()


def norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower()); s = re.sub(r"[^\w\s]", " ", s); return " ".join(s.split())


@torch.no_grad()
def evaluate(tag):
    M.eval(); res = {}
    PREDS = []
    for src in ("qwen", "chopper", "none"):
        R = {}
        for k, c in (list(CV.items())[::30] if A.smoke else CV.items()):
            if src == "qwen":
                notes = frags_of(c, LV[k]["spans"]) if k in LV else []
            elif src == "chopper":
                if k not in PRD:
                    continue
                notes = [f[3] for f in PRD[k]["fragments"]]
            elif src == "none":
                notes = []
            else:
                notes = [p["text"] for p in c["passages"]]
            ids = prompt(c["question"], notes); pred = generate(ids); gold_search = c["allin"] != 1.0
            said_search = pred.startswith(SEARCH)
            PREDS.append({"dataset": c["dataset"], "row": c["row"], "notes": src, "pred": pred})
            r = R.setdefault(c["dataset"], {"n": 0, "ss": 0, "s": 0, "aa": 0, "a": 0, "ex": 0, "sp": 0, "ce": [], "tok": []})
            r["n"] += 1
            if gold_search:
                r["s"] += 1; r["ss"] += said_search
            else:
                r["a"] += 1; r["aa"] += not said_search
                r["ex"] += norm(pred) == norm(c["answer"]); r["sp"] += bool(norm(c["answer"])) and norm(c["answer"]) in norm(pred)
                ce, hit = tf_loss(ids, tk.encode(" " + c["answer"]) + [STOP]); r["ce"].append(float(ce)); r["tok"].append(float(hit.mean()))
        for ds, r in R.items():
            log(f"EVAL {tag} | notes={src:6s} | {ds:8s} ({r['n']} q) | decision: SEARCH recall {r['ss'] / max(1, r['s']):.2f} ({r['s']}) "
                f"ANSWER recall {r['aa'] / max(1, r['a']):.2f} ({r['a']}) | answer-class: exact {r['ex'] / max(1, r['a']):.3f} "
                f"span {r['sp'] / max(1, r['a']):.3f} tok {np.mean(r['tok']) if r['tok'] else float('nan'):.3f} "
                f"CE {np.mean(r['ce']) if r['ce'] else float('nan'):.3f}")
            res[f"{src}/{ds}"] = r
    with open(os.path.join(A.out, f"preds_{tag.split()[1]}.jsonl"), "w") as f_:
        for x in PREDS:
            f_.write(json.dumps(x, ensure_ascii=False) + "\n")
    M.train(); return res


if not A.smoke:
    evaluate("step 0 (continued)" if A.init else "step 0 (base model)")
M.train(); t0 = time.time(); win = []; upd = math.ceil(len(TR) / A.accum)
for step, (c, notes) in enumerate(TR, 1):
    u = (step - 1) // A.accum
    for g in opt.param_groups:
        g["lr"] = A.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u / upd)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ce, hit = tf_loss(prompt(c["question"], notes), target(c))
    (ce / A.accum).backward(); win.append((float(ce), float(hit.mean()), c["allin"] != 1.0))
    if step % A.accum == 0 or step == len(TR):
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step % 200 == 0 or A.smoke:
        w = np.array(win, dtype=float); win.clear()
        log(f"step {step}/{len(TR)} | CE {w[:, 0].mean():.3f} tok {w[:, 1].mean():.3f} | {(time.time() - t0) / step:.2f}s/step | "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
    if A.eval_every and step % A.eval_every == 0 and step < len(TR):
        evaluate(f"step {step} (training)")
M.save_pretrained(os.path.join(A.out, "lora"))
evaluate(f"step {len(TR)} (trained)")
log("QWEN ANSWER DONE")
