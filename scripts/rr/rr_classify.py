#!/usr/bin/env python3
"""Step 1b of the rerank chopper dataset: Qwen3-4B (no thinking, greedy) sorts every pool passage into one of the 8
business areas of taxonomy.json (or 0 = none) and grades it 0-2 (0 boilerplate / spam / navigation, 1 thin, 2 a
self-contained informative passage). Reads only the passage start (200 words / 400 CJK characters): enough to tell
the area. Waits for each chunks_<lang>.jsonl to appear (rr_pool.py writes it atomically).
usage: rr_classify.py <rr dir> [lang ...]  -> <rr>/area_<lang>.jsonl {id, area, q}  (area/q = -1: unparsed)"""
import json, os, re, sys, time
from collections import Counter
from vllm import LLM, SamplingParams

TX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.json")))
D = sys.argv[1]; LANGS = sys.argv[2:] or list(TX["languages"])
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
SMOKE = int(os.environ.get("SMOKE", "0"))


def head(t, lang):
    return t[:400] if lang in ("zh", "ja") else " ".join(t.split()[:200])


llm = LLM(model="Qwen/Qwen3-4B", max_model_len=2048, gpu_memory_utilization=float(os.environ.get("GPU_FRAC", "0.85")), seed=0,
          enable_prefix_caching=True)
sp = SamplingParams(temperature=0.0, max_tokens=6)
for lang in LANGS:
    src, dst = f"{D}/chunks_{lang}.jsonl", f"{D}/area_{lang}.jsonl"
    if os.path.exists(dst) and not SMOKE:
        continue
    while not os.path.exists(src):
        time.sleep(30)
    C = [json.loads(l) for l in open(src)]
    if SMOKE:
        C = C[:SMOKE]
    t0 = time.time()
    outs = llm.chat([[{"role": "user", "content": PROMPT.format(areas=AREAS, lang=lang, text=head(c["text"], lang))}] for c in C], sp,
                    use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    R = []
    for c, o in zip(C, outs):
        m = re.match(r"\s*(\d)\D+(\d)", o.outputs[0].text)
        a, q = (int(m.group(1)), int(m.group(2))) if m and m.group(1) in TX["areas"] and m.group(2) in "012" else (-1, -1)
        R.append({"id": c["id"], "area": a, "q": q})
    if not SMOKE:
        with open(dst + ".tmp", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in R)
        os.rename(dst + ".tmp", dst)
    ca = Counter(r["area"] for r in R); good = Counter(r["area"] for r in R if r["q"] == 2)
    print(f"{lang}: {len(R)} passages in {time.time() - t0:.0f}s ({len(R) / (time.time() - t0):.1f}/s) | area share "
          + " ".join(f"{k}:{ca[int(k)] / len(R):.3f}" for k in TX["areas"]) + f" | unparsed {ca[-1]} | quality-2 per area "
          + " ".join(f"{k}:{good[int(k)]}" for k in TX["areas"]), flush=True)
    if SMOKE:
        for c, r in list(zip(C, R))[:12]:
            print(r["area"], r["q"], "|", c["text"][:110].replace("\n", " "))
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
