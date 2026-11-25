"""Leakage check for the chopper-0.31B MTEB reranking results (asked on embeddings-benchmark/results#870).
For each task: are its queries among our training questions, and do its documents appear in our training passages?
Overlap = word 8-grams (lowercased, letters/digits only). Training data = every row the released 0.31B saw:
stage 1 (benchmark_chains + wikipedia_chains = 54k rows) and stage 3 (final_mix/train, 102,654 rows), downloaded
from the public dataset huggingface.co/datasets/wiiiktor/chopper_demo (the files are byte-identical to the ones trained on).
usage: pip install mteb huggingface_hub; python leak_check.py OUT.json   (~15 min on one CPU core, ~3 GB RAM)"""
import json, re, sys
import mteb
from huggingface_hub import hf_hub_download

N = 8
TASKS = ["WikipediaRerankingMultilingual", "RuBQReranking", "StackOverflowDupQuestions"]
FILES = (["benchmark_chains/ctx_train.jsonl", "benchmark_chains/ctx_train2.jsonl", "benchmark_chains/ctx_train3.jsonl",
          "wikipedia_chains/ctx_wtrain.jsonl"] +
         [f"final_mix/train/ctx_{k}.jsonl" for k in "cm8k hpq mhret mhtrain2 mqu msc qasper rrfull20 rrfull20b w2".split()])
TRAIN = [hf_hub_download("wiiiktor/chopper_demo", f, repo_type="dataset") for f in FILES]
W = re.compile(r"\w+", re.U)


def words(s):
    return W.findall((s or "").lower())


def grams(s):
    w = words(s)
    if len(w) < N:
        return {hash(" ".join(w))} if len(w) >= 4 else set(), True
    return {hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)}, False


def text(r):
    t, x = r.get("title") or "", r.get("text") or ""
    return (t + " " + x).strip()


# 1. task side: per-doc gram sets, query strings
task = {}
need = set()
for name in TASKS:
    t = mteb.get_task(name); t.load_data()
    docs, qs, pos = {}, {}, set()
    for sub, d in t.dataset.items():
        for split, s in d.items():
            for r in s["corpus"]:
                docs[(sub, r["id"])] = text(r)
            for r in s["queries"]:
                qs[(sub, r["id"])] = r["text"] if isinstance(r["text"], str) else " ".join(r["text"])
            for q, rel in s["relevant_docs"].items():
                pos |= {(sub, k) for k, v in rel.items() if v > 0}
    G = {k: grams(v) for k, v in docs.items()}
    for g, _ in G.values():
        need |= g
    task[name] = dict(G=G, qs=qs, pos=pos)
    print(name, len(docs), "docs", len(qs), "queries", len(pos), "positives", flush=True)

# 2. stream training passages: which needed grams occur; also the training questions
seen, tq, short = set(), set(), {}
nrow = 0
for f in TRAIN:
    for line in open(f):
        d = json.loads(line); nrow += 1
        tq.add(" ".join(words(d.get("question", ""))))
        for p in d.get("passages", []):
            w = words(p["text"])
            for i in range(len(w) - N + 1):
                h = hash(" ".join(w[i:i + N]))
                if h in need:
                    seen.add(h)
            # short task docs (< N words) are matched as a contiguous run of any length 4..N-1
            for n in range(4, N):
                for i in range(len(w) - n + 1):
                    h = hash(" ".join(w[i:i + n]))
                    if h in need:
                        seen.add(h)
    print(f, nrow, flush=True)

# 3. report
out = {"train_files": TRAIN, "train_rows": nrow, "ngram": N}
for name, T in task.items():
    full = any_ = full_pos = any_pos = nshort = 0
    examples = []
    for k, (g, is_short) in T["G"].items():
        if not g:
            continue
        nshort += is_short
        hit = len(g & seen) / len(g)
        if hit > 0:
            any_ += 1; any_pos += k in T["pos"]
        if hit >= 0.5:
            full += 1; full_pos += k in T["pos"]
            if len(examples) < 5:
                examples.append({"doc": k[1], "subset": k[0], "share": round(hit, 2)})
    qhit = sum(" ".join(words(q)) in tq for q in T["qs"].values())
    out[name] = dict(docs=len(T["G"]), short_docs=nshort, docs_any_8gram=any_, docs_half_8grams=full,
                     positives=len(T["pos"]), pos_any_8gram=any_pos, pos_half_8grams=full_pos,
                     queries=len(T["qs"]), queries_exact_in_train_questions=qhit, examples=examples)
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("LEAK CHECK DONE")
