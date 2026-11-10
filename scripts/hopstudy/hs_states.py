#!/usr/bin/env python3
"""Hop study: answerer input states (tool_eval_answer.py format, judged by memb_judge.py) for one notes source.
  raw      all 20 MuSiQue paragraphs (plain RAG over a perfect pool)
  perfect  hs_build.py's perfect notes (gold hop sentences: the near-perfect-teacher proxy)
  teacher  Qwen3-14B quotes (mm_qwen_labels.py output; the teacher saw the answer)
  chop     a chopper's fragments (preds_val.jsonl from mm_train.py / mla_keeper_train.py --eval-only)
usage: hs_states.py <hs dir> <arm> <out state.jsonl> [labels or preds file]"""
import json, sys
HD, ARM, OUT = sys.argv[1:4]; SRC = sys.argv[4] if len(sys.argv) > 4 else ""
B = json.loads(open("/media/wii/bigdata/rr/tool_train.jsonl").readline())
R = {r["row"]: r for r in json.load(open(f"{HD}/rows.json"))}
C = {c["row"]: c for c in map(json.loads, open(f"{HD}/ctx_hs.jsonl"))}
L = {x["row"]: x for x in map(json.loads, open(SRC))} if SRC else {}
with open(OUT, "w") as f:
    for n, r in R.items():
        c = C[n]
        if ARM == "raw":
            notes = "\n\n".join(p["text"] for p in c["passages"])
        elif ARM == "perfect":
            notes = "\n".join(f"- {t}" for t in r["perfect"])
        elif ARM == "teacher":
            sp = L.get(n, {}).get("spans", [])
            notes = "\n".join(f"- {' '.join(c['passages'][k]['text'][a:b].split())}" for k, a, b in sp) or "(none)"
        else:
            fr = L.get(n, {}).get("fragments", [])
            notes = "\n".join(f"- {' '.join(x[3].replace('<eos>', ' ').split())}" for x in fr) or "(none)"
        f.write(json.dumps({"key": [f"hop{r['hops']}", n], "q": {"bench": "hs", "cat": f"hop{r['hops']}", "question": r["q"], "gold": r["a"], "date": "",
                                                               "ev": [], "qid": n}, "system": B["system"], "tools": B["tools"], "seen": [],
                            "notes_words": len(notes.split()), "messages": [{"role": "user", "content": f"Notes:\n{notes}\n\nQuestion: {r['q']}"}]},
                           ensure_ascii=False) + "\n")
print(ARM, len(R), "states")
