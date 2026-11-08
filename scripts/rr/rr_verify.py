#!/usr/bin/env python3
"""Step 1c of the rerank chopper dataset: per language x area, the CAND passages closest to the area description
(rr_embed.py vectors) go to Qwen3-4B (no thinking, greedy), which names the area (or 0) and grades the passage 0-2.
A passage is in the area only when Qwen3-4B agrees with the embedding's pick and grades it 2.
usage: rr_verify.py <rr dir> [lang ...] -> <rr>/area_<lang>.jsonl {id, row, cand_area, area, q, cos}"""
import json, os, re, sys, time
from collections import Counter
import numpy as np
from vllm import LLM, SamplingParams

TX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.json")))
D = sys.argv[1]; LANGS = sys.argv[2:] or list(TX["languages"])
CAND = int(os.environ.get("CAND", "3000"))
AREAS = "\n".join(f"{k}: {v}" for k, v in TX["areas"].items())
PROMPT = """Classify this web passage for a business search dataset.

Areas:
{areas}

Quality:
0: boilerplate, navigation, spam, a list of links or keywords, or unreadable
1: readable but thin (little concrete information)
2: a self-contained passage with concrete information someone could search for

Passage ({lang}):
<<<
{text}
>>>

Reply with exactly two numbers, the area then the quality, e.g. "6 2"."""
llm = LLM(model="Qwen/Qwen3-4B", max_model_len=2048, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True,
          max_num_batched_tokens=32768, max_num_seqs=256)
sp = SamplingParams(temperature=0.0, max_tokens=6)
A = np.load(f"{D}/emb_areas.npy").astype(np.float32)            # areas 1..8
for lang in LANGS:
    dst = f"{D}/area_{lang}.jsonl"
    if os.path.exists(dst):
        continue
    while not os.path.exists(f"{D}/emb_{lang}.npy"):
        time.sleep(60)
    E = np.load(f"{D}/emb_{lang}.npy"); C = [json.loads(l) for l in open(f"{D}/chunks_{lang}.jsonl")]
    S = np.concatenate([E[i:i + 50000].astype(np.float32) @ A.T for i in range(0, len(E), 50000)])
    pick = {}
    for a in range(A.shape[0]):                                  # top CAND per area; a passage goes to its best-scoring area
        for r in np.argsort(-S[:, a])[:CAND]:
            if r not in pick or S[r, a] > S[r, pick[r]]:
                pick[r] = a
    rows = sorted(pick); t0 = time.time()
    hd = lambda t: t[:400] if lang in ("zh", "ja") else " ".join(t.split()[:200])[:1500]
    outs = llm.chat([[{"role": "user", "content": PROMPT.format(areas=AREAS, lang=lang, text=hd(C[r]["text"]))}] for r in rows], sp,
                    use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    R = []
    for r, o in zip(rows, outs):
        m = re.match(r"\s*(\d)\D+(\d)", o.outputs[0].text)
        a, q = (int(m.group(1)), int(m.group(2))) if m and m.group(1) in TX["areas"] and m.group(2) in "012" else (-1, -1)
        R.append({"id": C[r]["id"], "row": int(r), "cand_area": pick[r] + 1, "area": a, "q": q, "cos": round(float(S[r, pick[r]]), 4)})
    with open(dst + ".tmp", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in R)
    os.rename(dst + ".tmp", dst)
    ok = Counter(x["area"] for x in R if x["area"] == x["cand_area"] and x["q"] == 2)
    print(f"{lang}: {len(R)} candidates in {time.time() - t0:.0f}s ({len(R) / (time.time() - t0):.1f}/s) | agreed + quality 2 per area "
          + " ".join(f"{k}:{ok[k]}" for k in range(1, 9)) + f" | unparsed {sum(x['area'] < 0 for x in R)}", flush=True)
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
