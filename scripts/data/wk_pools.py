#!/usr/bin/env python
"""mm_dump_pools.py-format pools from the accepted Wikipedia questions (wk_judge.py), 2026-10-30, so mm_contexts.py →
mm_qwen_labels.py → mm_subq.py run on them unchanged. Per question --N chunks ("Title\\nparagraph", like the benchmark
chunks): the gold paragraph(s), hard negatives (the other paragraphs of A and B, the first two paragraphs of up to 8
other articles A mentions), filled up with random padding leads; shuffled with rng 30_000_000 + row.
Datasets "wiki2hop" (groups [[A], [B]]) and "wiki1hop" (groups [[A]]). Takes accepted rows in file order up to the
quotas, writes pools_/chunks_<tag>.json(l) for train and val into <mm dir>.
usage: .venv-rwkv7/bin/python wk_pools.py <wiki dir> <mm dir> <judged.jsonl ...> [--q2 18000 --q1 9000 --v2 200 --v1 100
       --N 1000 --tag wtrain --vtag wval]"""
import json, os, random, sys
args = sys.argv[1:]; O = {"q2": 18000, "q1": 9000, "v2": 200, "v1": 100, "N": 1000, "tag": "wtrain", "vtag": "wval"}
for k in list(O):
    if f"--{k}" in args:
        i = args.index(f"--{k}"); O[k] = type(O[k])(args[i + 1]); del args[i:i + 2]
WD, MD, FILES = args[0], args[1], args[2:]
ok = [r for f in FILES for r in map(json.loads, open(f)) if r["ok"]]
seen, Q = set(), {("train", "2hop"): [], ("train", "1hop"): [], ("val", "2hop"): [], ("val", "1hop"): []}
quota = {("train", "2hop"): O["q2"], ("train", "1hop"): O["q1"], ("val", "2hop"): O["v2"], ("val", "1hop"): O["v1"]}
for r in ok:
    k = (r["split"], r["type"]); key = r["question"].lower()
    if key in seen or len(Q[k]) >= quota[k]:
        continue
    seen.add(key); Q[k].append(r)
print({f"{a}/{b}": len(v) for (a, b), v in Q.items()}, "of quotas", quota, flush=True)
need = {t for v in Q.values() for r in v for t in [r["a"], r.get("b"), *r.get("neg", [])] if t}
ART = {}
for l in open(f"{WD}/articles.jsonl"):
    d = json.loads(l)
    if d["t"] in need:
        ART[d["t"]] = d["p"]
PAD = [f'{d["t"]}\n{d["p"][0]}' for d in map(json.loads, open(f"{WD}/pad.jsonl")) if d["p"]]
print(f"{len(ART):,} articles, {len(PAD):,} padding leads", flush=True)

for split, tag in (("train", O["tag"]), ("val", O["vtag"])):
    table, tid = [], {}
    def cidx(t):
        if t not in tid:
            tid[t] = len(table); table.append(t)
        return tid[t]
    rows = Q[(split, "2hop")] + Q[(split, "1hop")]
    random.Random(7).shuffle(rows)
    with open(f"{MD}/pools_{tag}.jsonl", "w") as fo:
        for j, r in enumerate(rows):
            rng = random.Random(30_000_000 + j)
            A = ART[r["a"]]; gold = [f'{r["a"]}\n{A[r["a_pi"]]}']
            neg = [f'{r["a"]}\n{p}' for k, p in enumerate(A) if k != r["a_pi"]]
            if r["type"] == "2hop":
                B = ART[r["b"]]; gold.append(f'{r["b"]}\n{B[r["b_pi"]]}')
                neg += [f'{r["b"]}\n{p}' for k, p in enumerate(B) if k != r["b_pi"]]
                neg += [f"{t}\n{p}" for t in r.get("neg", []) for p in ART.get(t, [])[:2]]
            neg = list(dict.fromkeys(n for n in neg if n not in gold))
            fill = []
            while len(gold) + len(neg) + len(fill) < O["N"]:
                p = PAD[rng.randrange(len(PAD))]
                if p not in gold and p not in neg:
                    fill.append(p)
            items = [(t, g) for g, t in enumerate(gold)] + [(t, -1) for t in neg + fill]
            rng.shuffle(items)
            groups = [[i for i, (_, g) in enumerate(items) if g == k] for k in range(len(gold))]
            fo.write(json.dumps({"dataset": "wiki" + r["type"], "split": split, "N": O["N"], "row": j,
                                 "question": r["question"], "answer": r["answer"],
                                 "chunks": [cidx(t) for t, _ in items], "groups": groups,
                                 "doc": list(range(len(items))), "a": r["a"], "b": r.get("b")}, ensure_ascii=False) + "\n")
    json.dump(table, open(f"{MD}/chunks_{tag}.json", "w"), ensure_ascii=False)
    print(f"POOLS {tag}: {len(rows)} pools, {len(table):,} distinct chunks", flush=True)
print("WK POOLS DONE", flush=True)
