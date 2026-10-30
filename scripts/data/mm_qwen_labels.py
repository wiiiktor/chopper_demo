#!/usr/bin/env python
"""Qwen fragment labels on the mmBERT contexts (mm_contexts.py), as CHARACTER spans per passage (2026-10-29). Same
teacher as qwen_keep_labels.py (Qwen3-14B-AWQ sees question + gold answer + all K passages at once, writes the reasoning
chain, quotes verbatim the shortest fragment(s) stating each fact, every place) plus one rule learned on the first set:
a fragment that names the needed entity only by a pronoun must also quote the place that names it.
Quotes are located per passage: exact, then on letters/digits only ('...' parts separately), then the longest common run
covering ≥60% of the quote. Writes per question: dataset, split, row, spans [[passage, char0, char1]], steps, miss.
usage: .venv-vllm/bin/python mm_qwen_labels.py <ctx.jsonl> <out.jsonl> [--base Qwen/Qwen3-14B-AWQ]"""
import argparse, difflib, json, re, time

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--base", default="Qwen/Qwen3-14B-AWQ")
ap.add_argument("--max-len", type=int, default=12288)
ap.add_argument("--gpu-frac", type=float, default=0.88)
A = ap.parse_args()
D = [json.loads(l) for l in open(A.inp)]


def prompt(d):
    ps = [f"Passage {k + 1}:\n{p['text'].strip()}" for k, p in enumerate(d["passages"])]
    return (f"Question: {d['question']}\nCorrect answer: {d['answer']}\n\n" + "\n\n".join(ps) +
            "\n\nTask: mark the text a reader needs to get from the question to the correct answer.\n"
            "1. Write the reasoning chain as short steps, one fact per step. Example for 'Who is the spouse of the "
            "director of X?': 'X was directed by Y', then 'Y is married to Z'. A one-step question has one step.\n"
            "2. For each step, QUOTE word for word the shortest fragment of the passages that states that fact. A fragment "
            "can be part of a sentence; it must be copied exactly, never paraphrased. If several places state the same "
            "fact, quote each place. A fragment that only mentions a name, without stating the fact, does not count. "
            "If a fragment refers to the needed person or thing only by a pronoun (he, she, it, they), also quote the "
            "nearby words that name it.\n"
            'Reply with JSON only: {"steps": [{"fact": "<the fact>", "quotes": ["<exact fragment>", ...]}]}. A step that '
            "no passage states gets an empty list.")


def norm_map(s):
    keep = [(c.lower(), i) for i, c in enumerate(s) if c.isalnum()]
    return "".join(c for c, _ in keep), [i for _, i in keep]


def locate(text, NM, quote):
    """character spans of the quote in one passage text (exact → letters/digits only → longest common run)"""
    q = quote.strip().strip('"').strip()
    if len(q) < 3:
        return []
    hits = [(m.start(), m.end()) for m in re.finditer(re.escape(q), text)]
    if not hits:
        nc, idx = NM
        for part in re.split(r"\.\.\.|…", q):
            np_ = norm_map(part)[0]
            if len(np_) < 8:
                continue
            k = nc.find(np_)
            while k >= 0:
                hits.append((idx[k], idx[k + len(np_) - 1] + 1)); k = nc.find(np_, k + 1)
        if not hits and nc:
            np_ = norm_map(q)[0]
            m = difflib.SequenceMatcher(None, nc, np_, autojunk=False).find_longest_match(0, len(nc), 0, len(np_))
            if m.size >= max(20, 0.6 * len(np_)):
                hits.append((idx[m.a], idx[m.a + m.size - 1] + 1))
    return hits


from vllm import LLM, SamplingParams
llm = LLM(model=A.base, max_model_len=A.max_len, gpu_memory_utilization=A.gpu_frac, seed=0)
sp = SamplingParams(temperature=0.0, max_tokens=1024)
t0 = time.time()
TX = [o.outputs[0].text for o in llm.chat([[{"role": "user", "content": prompt(d)}] for d in D], sp, use_tqdm=True,
                                          chat_template_kwargs={"enable_thinking": False})]
bad = nq = nmiss = 0
with open(A.out, "w") as fo:
    for d, txt in zip(D, TX):
        try:
            steps = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)).get("steps") or []
        except Exception:
            bad += 1; steps = []
        NMs = [norm_map(p["text"]) for p in d["passages"]]; spans, miss = set(), []
        for s in steps:
            for qt in (s.get("quotes") or []) if isinstance(s, dict) else []:
                nq += 1; found = False
                for k, p in enumerate(d["passages"]):
                    for c0, c1 in locate(p["text"], NMs[k], str(qt)):
                        spans.add((k, c0, c1)); found = True
                if not found:
                    miss.append(qt); nmiss += 1
        fo.write(json.dumps({"dataset": d["dataset"], "split": d["split"], "row": d["row"], "spans": sorted(spans),
                             "steps": steps, "miss": miss}, ensure_ascii=False) + "\n")
print(f"[mm-qwen] {len(D)} questions in {(time.time() - t0) / 60:.1f} min | unparsable {bad} | quotes {nq}, not found {nmiss}", flush=True)
print("MM QWEN DONE", flush=True)
