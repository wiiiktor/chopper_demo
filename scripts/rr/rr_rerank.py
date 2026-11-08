#!/usr/bin/env python3
"""Step 4a of the rerank dataset pilot: Qwen3-Reranker-0.6B scores all 16 (query, passage) pairs of every row, so the
Qwen3-14B labeller (rr_label.py) reads the seed + the passages the reranker finds plausible + a 5% random audit of the
rest instead of all 640k pairs (~16 h). The audit measures how many real answers the prefilter misses.
Writes <dir>/rr_<lang>.json {rid: [p_yes x 16]}.   usage: rr_rerank.py <dir> [lang ...] [SMOKE=N env]"""
import json, math, os, sys, time
from vllm import LLM, SamplingParams

D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]
SMOKE = int(os.environ.get("SMOKE", "0")); V = os.environ.get("V", ""); VP = V + "_" if V else ""   # V=v2: rows_v2_<lang>
llm = LLM(model="Qwen/Qwen3-Reranker-0.6B", max_model_len=2048, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True)
TOK = llm.get_tokenizer()
YES, NO = TOK.convert_tokens_to_ids("yes"), TOK.convert_tokens_to_ids("no")
sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[YES, NO])
PRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
       "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INS = "Given a search query, does the passage contain information that answers it?"
for lang in LANGS:
    dst = f"{D}/rr_{VP}{lang}{'_smoke' if SMOKE else ''}.json"
    R = [json.loads(l) for l in open(f"{D}/rows_{VP}{lang}.jsonl")][:SMOKE or None]
    ids = []
    for r in R:
        for p in r["passages"]:
            x = TOK(PRE + f"<Instruct>: {INS}\n<Query>: {r['query']}\n<Document>: ", add_special_tokens=False)["input_ids"]
            d = TOK(p["text"], add_special_tokens=False)["input_ids"][:1700]
            ids.append({"prompt_token_ids": x + d + TOK(SUF, add_special_tokens=False)["input_ids"]})
    t0 = time.time(); outs = llm.generate(ids, sp, use_tqdm=False)
    P = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        y = math.exp(lp[YES].logprob) if YES in lp else 0.0; n = math.exp(lp[NO].logprob) if NO in lp else 0.0
        P.append(round(y / (y + n), 4) if y + n else 0.0)
    out = {r["rid"]: P[i * 16:(i + 1) * 16] for i, r in enumerate(R)}
    json.dump(out, open(dst + ".tmp", "w")); os.rename(dst + ".tmp", dst)
    seedp = sum(v[r["seed"]] for r, v in zip(R, out.values()) if r["qtype"] != "no_answer") / max(1, sum(r["qtype"] != "no_answer" for r in R))
    print(f"{lang}: {len(R)} rows x16 in {time.time() - t0:.0f}s ({len(ids) / (time.time() - t0):.0f} pairs/s) | mean p(seed) answer-rows {seedp:.3f} "
          f"| pairs >0.3 {sum(p > 0.3 for p in P) / len(P):.3f}", flush=True)
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
