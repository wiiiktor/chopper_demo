#!/usr/bin/env python3
"""Step 1b of the rerank chopper dataset: Qwen3-Embedding-0.6B embeds the start of every pool passage (200 words /
400 CJK characters) and each area description of taxonomy.json (as an instructed query). Qwen3-4B on all ~1.4M
passages ran 17/s (23 h); the embedding picks the candidates and rr_verify.py checks only those. The same vectors
later retrieve the 16-passage pools (step 3).
usage: rr_embed.py <rr dir> [lang ...] -> <rr>/emb_<lang>.npy (float16, L2-normalised, row order = chunks_<lang>.jsonl)
                                          <rr>/emb_areas.npy"""
import json, os, sys, time
import numpy as np
from vllm import LLM

TX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.json")))
D = sys.argv[1]; LANGS = sys.argv[2:] or list(TX["languages"])
SMOKE = int(os.environ.get("SMOKE", "0"))
llm = LLM(model="Qwen/Qwen3-Embedding-0.6B", runner="pooling", max_model_len=1024, gpu_memory_utilization=0.85, seed=0)


TOK = llm.get_tokenizer()


def emb(texts):
    ids = [{"prompt_token_ids": x[:1000]} for x in TOK(texts, add_special_tokens=True)["input_ids"]]   # never over max_model_len
    v = np.array([o.outputs.embedding for o in llm.embed(ids, use_tqdm=False)], dtype=np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


TASK = "Instruct: Given a description of a business document category, retrieve web passages that belong to it\nQuery: "
np.save(f"{D}/emb_areas.npy", emb([TASK + TX["areas"][k] for k in TX["areas"] if k != "0"]))
for lang in LANGS:
    dst = f"{D}/emb_{lang}.npy"
    if os.path.exists(dst) and not SMOKE:
        continue
    T = [json.loads(l)["text"] for l in open(f"{D}/chunks_{lang}.jsonl")]
    if SMOKE:
        T = T[:SMOKE]
    T = [t[:400] if lang in ("zh", "ja") else " ".join(t.split()[:200])[:1500] for t in T]   # 1500 chars: a 200-"word" URL list broke 1024 tokens
    t0 = time.time(); E = np.concatenate([emb(T[i:i + 20000]) for i in range(0, len(T), 20000)])
    print(f"{lang}: {len(T)} passages in {time.time() - t0:.0f}s ({len(T) / (time.time() - t0):.0f}/s)", flush=True)
    if not SMOKE:
        np.save(dst + ".tmp.npy", E); os.rename(dst + ".tmp.npy", dst)
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
