#!/usr/bin/env python
"""Accept / reject the generated Wikipedia questions (wk_gen.py), 2026-10-30. Rules first, then Qwen3-4B-Instruct-2507
answers the question from the gold passages (the qa3 local pipeline's judge model, which tied gpt-5.6-luna there):
  rules  — answer non-empty, ≤ 8 words, found in its passage (B for 2-hop) and, for 2-hop, NOT in Passage A nor in the
           question; the question does not name B (full title or its part before a parenthesis) and has no
           "passage" / "according to" / "mentioned"; question 20-300 chars
  solve  — 4B with the gold passage(s) gets the answer (normalised containment either way, or token F1 ≥ 0.6)
  closed — 4B with NO passage (recorded only: how often the fact is common knowledge)
Writes every row with "ok", "why" and the gold paragraph index b_pi (2-hop: which of B's paragraphs holds the answer).
usage: .venv-vllm/bin/python wk_judge.py <gen.jsonl> <out.jsonl>"""
import json, os, re, string, sys, time
from collections import Counter
INP, OUT = sys.argv[1:3]
JUDGE = os.environ.get("WK_JUDGE", "Qwen/Qwen3-4B-Instruct-2507"); FRAC = float(os.environ.get("WK_FRAC", "0.85"))
R = [json.loads(l) for l in open(INP)]


def norm(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation + "“”‘’"))
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def match(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    if g in p or (p in g and len(p) >= 3):
        return True
    pc, gc = Counter(p.split()), Counter(g.split()); c = sum((pc & gc).values())
    return c > 0 and 2 * c / (len(p.split()) + len(g.split())) >= 0.6


def rules(r):
    q, a = r.get("question", ""), r.get("answer", "")
    if not q or not a:
        return "empty"
    if len(a.split()) > 8:
        return "long answer"
    if not 20 <= len(q) <= 300:
        return "question length"
    if re.search(r"\bpassage|according to|mentioned\b", q, re.I):
        return "refers to the passage"
    if norm(a) and norm(a) in norm(q):
        return "answer in question"
    if r["type"] == "2hop":
        b = r["b"]; core = b.split(" (")[0]
        if b.lower() in q.lower() or (len(core) > 3 and core.lower() in q.lower()):
            return "names B"
        if norm(a) in norm(r["pa"]):
            return "answer in A"
        ps = [p for p in r["pb"].split("\n\n")]
        hit = [k for k, p in enumerate(ps) if norm(a) in norm(p)]
        if not hit:
            return "answer not in B"
        r["b_pi"] = hit[0]
    elif norm(a) not in norm(r["pa"]):
        return "answer not in passage"
    return ""


for r in R:
    r["why"] = rules(r)
live = [r for r in R if not r["why"]]
print(f"{len(R)} rows, {len(live)} pass the rules | " + ", ".join(f"{k} {v}" for k, v in Counter(r['why'] for r in R if r['why']).most_common()), flush=True)

ASK = "Answer with the shortest possible phrase (a few words), nothing else."
def solve_msg(r):
    ctx = (f'Passage — "{r["a"]}":\n{r["pa"]}\n\n' + (f'Passage — "{r["b"]}":\n{r["pb"]}\n\n' if r["type"] == "2hop" else ""))
    return f"{ctx}Question: {r['question']}\n{ASK}"


from vllm import LLM, SamplingParams
llm = LLM(model=JUDGE, max_model_len=4096, gpu_memory_utilization=FRAC, seed=0)
sp = SamplingParams(temperature=0.0, max_tokens=32); t0 = time.time()
S = [o.outputs[0].text.strip() for o in llm.chat([[{"role": "user", "content": solve_msg(r)}] for r in live], sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})]
Cb = [o.outputs[0].text.strip() for o in llm.chat([[{"role": "user", "content": f"Question: {r['question']}\n{ASK}"}] for r in live], sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})]
for r, s, c in zip(live, S, Cb):
    r["solve"], r["closed"] = s, c
    r["closed_ok"] = match(c, r["answer"])
    if not match(s, r["answer"]):
        r["why"] = "judge disagrees"
for r in R:
    r["ok"] = not r["why"]
with open(OUT, "w") as fo:
    for r in R:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
for t in ("2hop", "1hop"):
    x = [r for r in R if r["type"] == t]; ok = [r for r in x if r["ok"]]
    print(f"{t}: {len(ok)}/{len(x)} accepted ({len(ok) / max(1, len(x)):.1%}) | closed-book right on "
          f"{sum(r['closed_ok'] for r in ok) / max(1, len(ok)):.1%} of the accepted", flush=True)
print(f"WK JUDGE DONE in {(time.time() - t0) / 60:.1f} min | rejected: " +
      ", ".join(f"{k} {v}" for k, v in Counter(r['why'] for r in R if r['why']).most_common()), flush=True)
