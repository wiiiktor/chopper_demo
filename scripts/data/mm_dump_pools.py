#!/usr/bin/env python
"""Pools as TEXT for the mmBERT chopper (2026-10-29, user: "switch to mmbert with conversation tail and top 16 snippets,
and decisions search / answer"). Like ab_dump_pools.py (FAIR pools, tokens only, tag lines stripped) but also the TRAIN
split and the gold answer. Every chunk is stored once (by its text) in a shared chunk table, pools reference it by index,
so 3,000 pools of 1,000 chunks stay small. Train rows: each dataset's rng-0 permutation from the start (mmBERT never saw
any of them; MuSiQue reuse is allowed), pool rng 20_000_000 + row; val: the eval pools (rng 123 in row order).
usage: FAIR_TOKS_ONLY=1 mm_dump_pools.py <out_dir> <trainer flags incl. --fair-pools …> [--n musique:1200,hotpot:800,triviaqa:1000]
       [--split train|val] [--N 1000]"""
import sys, json, re, os
OUTD = sys.argv[1]; sys.argv = [sys.argv[0]] + sys.argv[2:]
O = {"n": "musique:1200,hotpot:800,triviaqa:1000", "split": "train", "N": "1000", "start": "0", "tag": ""}
for k in list(O):
    if f"--{k}" in sys.argv:
        i = sys.argv.index(f"--{k}"); O[k] = sys.argv[i + 1]; del sys.argv[i:i + 2]
assert os.environ.get("FAIR_TOKS_ONLY"), "set FAIR_TOKS_ONLY=1"
src = open("/home/wii/Projects/certain/scripts/train_ab_multihop.py").read()
exec(src[:src.index("params_mh = [p_ for p_ in params4")])
TAGLINE = re.compile(r"^\[doc \d+ · part \d+ of \d+ · [^\]]*\]\n")
os.makedirs(OUTD, exist_ok=True)
N, SPL = int(O["N"]), O["split"]
FTAG = O["tag"] or SPL; ST = {}                         # --start musique:1200,… → skip rows already dumped; --tag names the files
for _sx in filter(None, O["start"].split(",")) if O["start"] != "0" else []:    # (names must not shadow the
    _sk, _sv = _sx.split(":"); ST[_sk] = int(_sv)                                  # exec'd trainer's globals)
table, tid = [], {}


def cidx(text):
    if text not in tid:
        tid[text] = len(table); table.append(text)
    return tid[text]


n_out = 0
with open(os.path.join(OUTD, f"pools_{FTAG}.jsonl"), "w") as fo:
    for spec in O["n"].split(","):
        name, n = spec.split(":"); n = int(n); X = DS[SPL][name]
        perm = np.random.default_rng(0).permutation(len(X)); rng_v = np.random.default_rng(123)
        for t in range(ST.get(name, 0), min(ST.get(name, 0) + n, len(X))):
            j = t if SPL == "val" else int(perm[t])
            q, a, flat, doc, groups, cid = mh_pool(name, X, j, N, rng_v if SPL == "val" else np.random.default_rng(20_000_000 + j))
            fo.write(json.dumps({"dataset": name, "split": SPL, "N": N, "row": j,
                                 "question": tk.decode(q).replace("Question: ", "").replace("\nAnswer:", "").strip(),
                                 "answer": tk.decode([x for x in a if x != 3330]).strip(),
                                 "chunks": [cidx(TAGLINE.sub("", tk.decode(c))) for c in flat], "groups": groups,
                                 "doc": [int(x) for x in doc]}, ensure_ascii=False) + "\n")
            n_out += 1
        log(f"DUMP {SPL} {name}: {min(n, len(X))} pools | table {len(table)} chunks")
json.dump(table, open(os.path.join(OUTD, f"chunks_{FTAG}.json"), "w"), ensure_ascii=False)
log(f"DUMP DONE {n_out} pools, {len(table)} distinct chunks -> {OUTD}")
