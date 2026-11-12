#!/usr/bin/env python3
"""Head-to-head on the hop study (user 2026-11-12: "implement 2"): other context compressors cut the same 20 snippets per question,
Qwen3-14B answers from their output with the same prompt as every other arm (qwen_answer.py). Writes st_<arm>.jsonl (hs_states.py format).
  llml2   LLMLingua-2 (microsoft/llmlingua-2-xlm-roberta-large-meetingbank): question-agnostic token dropping, rate RATE (default 0.1,
          about the chopper's size); every snippet compressed on its own, kept as one line each
  prov    Provence (naver/provence-reranker-debertav3-v1): question-aware sentence pruning, its default threshold 0.1; the pruned
          snippets, empty ones dropped
usage: cmp_compress.py <hs dir> <llml2|prov>   (venv /media/wii/bigdata/venv_cmp)"""
import json, os, sys
HD, ARM = sys.argv[1], sys.argv[2]
B = json.loads(open("/media/wii/bigdata/rr/tool_train.jsonl").readline())
R = {r["row"]: r for r in json.load(open(f"{HD}/rows.json"))}
C = {c["row"]: c for c in map(json.loads, open(f"{HD}/ctx_hs.jsonl"))}
notes = {}
if ARM == "llml2":
    from llmlingua import PromptCompressor
    pc = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)
    rate = float(os.environ.get("RATE", "0.1"))
    for n, c in C.items():
        out = [pc.compress_prompt(p["text"], rate=rate, force_tokens=["\n", "?", "."])["compressed_prompt"] for p in c["passages"]]
        notes[n] = "\n".join(f"- {' '.join(o.split())}" for o in out if o.strip())
elif ARM in ("prov", "prov5"):                                  # prov5 = threshold 0.5, closer to the chopper's size
    from transformers import AutoModel
    m = AutoModel.from_pretrained("naver/provence-reranker-debertav3-v1", trust_remote_code=True).to("cuda").eval()
    for n, c in C.items():
        q = R[n]["q"]
        o = m.process([q], [[p["text"] for p in c["passages"]]], threshold=0.5 if ARM == "prov5" else 0.1, always_select_title=False,
                      enable_warnings=False)
        pr = o["pruned_context"][0]
        notes[n] = "\n".join(f"- {' '.join(t.split())}" for t in pr if t.strip())
with open(f"{HD}/st_{ARM}.jsonl", "w", encoding="utf-8") as f:
    for n, r in R.items():
        t = notes.get(n) or "(none)"
        f.write(json.dumps({"key": [f"hop{r['hops']}", n], "q": {"bench": "hs", "cat": f"hop{r['hops']}", "question": r["q"], "gold": r["a"], "date": "",
                                                               "ev": [], "qid": n}, "system": B["system"], "tools": B["tools"], "seen": [],
                            "notes_words": len(t.split()), "messages": [{"role": "user", "content": f"Notes:\n{t}\n\nQuestion: {r['q']}"}]},
                           ensure_ascii=False) + "\n")
print(ARM, len(R), "states, mean words", sum(len(v.split()) for v in notes.values()) / max(1, len(notes)), flush=True)
