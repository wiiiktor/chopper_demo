#!/usr/bin/env python
"""Questions for the Wikipedia candidates (wk_chains.py), Qwen3-14B-AWQ (2026-10-30). Benchmark-free replacement for
MuSiQue / HotpotQA (2-hop: A mentions B, the answer is in B, B is described only through A) and TriviaQA (1-hop: one
paragraph, a question that stands on its own). The model sees the passages, never a benchmark question.
Takes the next --n2 2-hop and --n1 1-hop candidates after --skip2 / --skip1 (round-based, see wiki_chain.sh).
Writes per candidate the candidate fields + question, answer, gold paragraphs (a_pi, b_text / text) and the raw reply.
usage: .venv-vllm/bin/python wk_gen.py <wiki dir> <out.jsonl> --n2 20000 --n1 10000 [--skip2 0 --skip1 0]"""
import argparse, json, random, re, time
ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("out")
for k in ("n2", "n1", "skip2", "skip1"):
    ap.add_argument(f"--{k}", type=int, default=0)
ap.add_argument("--base", default="Qwen/Qwen3-14B-AWQ")
ap.add_argument("--gpu-frac", type=float, default=0.88)
A = ap.parse_args()
C = [json.loads(l) for l in open(f"{A.dir}/cand.jsonl")]
c2 = [c for c in C if c["type"] == "2hop"][A.skip2:A.skip2 + A.n2]
c1 = [c for c in C if c["type"] == "1hop"][A.skip1:A.skip1 + A.n1]
need = {c["a"] for c in c2 + c1} | {c["b"] for c in c2}
ART = {}
for l in open(f"{A.dir}/articles.jsonl"):
    d = json.loads(l)
    if d["t"] in need:
        ART[d["t"]] = d["p"]

P2 = """Passage A — "{a}":
{pa}

Passage B — "{b}":
{pb}

Passage A mentions "{b}", the subject of Passage B. Write ONE question that can only be answered by reading both passages:
- The answer is a short fact stated in Passage B (a name, place, date, number or title — at most 6 words), copied
  exactly from Passage B. It must not appear in Passage A.
- The question must NOT contain the name "{b}". Refer to it only through what Passage A says about it (for example
  "the label that released <album from A>", "the river that flows through <town from A>").
- The question must name "{a}" or something specific from Passage A, so that a reader knows where to start.
- It must make sense on its own, as a quiz question: never say "the passage", "according to" or "mentioned".
Reply with JSON only: {{"question": "...", "answer": "...", "bridge": "<the words of Passage A that identify {b}>"}}"""
P1 = """Passage — "{a}":
{pa}

Write ONE quiz question whose answer is a short fact stated in this passage (a name, place, date, number or title — at
most 6 words), copied exactly from the passage.
- The question must make sense on its own, without the passage: name the subject specifically, never say "the
  passage", "according to" or "mentioned", and do not ask about something only this passage could settle.
Reply with JSON only: {{"question": "...", "answer": "..."}}"""

rows, msgs = [], []
for c in c2:
    if c["a"] not in ART or c["b"] not in ART or not ART[c["b"]] or c["a_pi"] >= len(ART[c["a"]]):
        continue
    pb = "\n\n".join(ART[c["b"]][:2])
    rows.append({**c, "pa": ART[c["a"]][c["a_pi"]], "pb": pb})
    msgs.append(P2.format(a=c["a"], b=c["b"], pa=rows[-1]["pa"], pb=pb))
for c in c1:
    ps = ART.get(c["a"]) or []
    if not ps:
        continue
    pi = random.Random(c["a"]).randrange(min(4, len(ps)))
    rows.append({**c, "a_pi": pi, "pa": ps[pi]})
    msgs.append(P1.format(a=c["a"], pa=ps[pi]))
print(f"{len(rows)} candidates ({sum(r['type'] == '2hop' for r in rows)} 2-hop)", flush=True)

from vllm import LLM, SamplingParams
llm = LLM(model=A.base, max_model_len=4096, gpu_memory_utilization=A.gpu_frac, seed=0)
t0 = time.time()
TX = [o.outputs[0].text for o in llm.chat([[{"role": "user", "content": m}] for m in msgs],
                                          SamplingParams(temperature=0.0, max_tokens=200), use_tqdm=True,
                                          chat_template_kwargs={"enable_thinking": False})]
bad = 0
with open(A.out, "w") as fo:
    for r, t in zip(rows, TX):
        try:
            j = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
            r.update(question=str(j.get("question", "")).strip(), answer=str(j.get("answer", "")).strip(),
                     bridge=str(j.get("bridge", "")).strip())
        except Exception:
            bad += 1; r.update(question="", answer="")
        r["raw"] = t
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"WK GEN DONE {len(rows)} in {(time.time() - t0) / 60:.1f} min | unparsable {bad}", flush=True)
