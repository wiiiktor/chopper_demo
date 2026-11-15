#!/usr/bin/env python3
"""Fit chopper training splits to one base model's tokenizer (user 2026-11-15, Polish bases for OPI).

Two problems appear when a base with a narrow vocabulary (pl-ModernBERT, polish-roberta) reads the universal mix:
  1. unreadable text: Japanese / Chinese become a few <unk> tokens, each covering a whole sentence
     -> drop a row when more than --max-unk (default 30%) of its input characters sit under <unk> tokens
     (measured: Japanese 0.83, Chinese 0.77 vs at most 0.05 for French/Portuguese, where only accented letters are <unk>)
  2. right-truncation at 8,192 tokens (mm_train tokenises with truncation=True): German/French/Spanish/Portuguese rows
     need ~10k tokens; the cut passages can hold the answer while the row still says ANSWER
     -> drop passages WITHOUT label spans from the end until the input fits; if the labelled passages alone do not
        fit, drop the row. Spans are re-indexed to the kept passages; the ANSWER/SEARCH label stays valid because
        every labelled passage is kept.
The input string is built exactly as mm_train.encode() builds it. Writes ctx_<split><suffix>.jsonl and
qw_<split><suffix>.jsonl next to the originals and prints per-split statistics.

usage: fit_splits.py <dir> <base model> <suffix> <split,split,...> [--max-len 8192] [--max-unk 0.30]
"""
import argparse, bisect, json
from collections import Counter

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("base"); ap.add_argument("suffix"); ap.add_argument("splits")
ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--max-unk", type=float, default=0.30)
A = ap.parse_args()
tk = AutoTokenizer.from_pretrained(A.base)
SEP = f" {tk.sep_token} "
BUDGET = A.max_len - 2                                   # [CLS] ... [SEP] added by the tokenizer
BATCH = 256


def build(c):
    s = f"Question: {c['question']}"; p0 = []
    for p in c["passages"]:
        s += SEP; p0.append(len(s)); s += p["text"].strip()
    return s, p0


def fit(c, lab, enc):
    """-> (ctx, lab, reason) with reason None when kept"""
    s, p0 = build(c)
    off = enc["offset_mapping"]; ids = enc["input_ids"]
    unk = sum(b - a for (a, b), i in zip(off, ids) if i == tk.unk_token_id)
    if unk > A.max_unk * max(len(s), 1):
        return None, None, "unk"
    if len(ids) <= BUDGET:
        return c, lab, None
    # tokens per passage (incl. its separator), from the offsets of ONE tokenisation of the full string
    starts = [p - len(SEP) for p in p0]                   # passage k owns [its separator + text]
    ntok = [0] * len(p0); head = 0
    for a, b in off:
        k = bisect.bisect_right(starts, a) - 1
        if k < 0:
            head += 1
        else:
            ntok[k] += 1
    labelled = {k for k, _, _ in lab["spans"]}
    keep = list(range(len(p0))); total = head + sum(ntok)
    for k in reversed(range(len(p0))):
        if total <= BUDGET - 16:                          # small margin: tokenisation at new borders may differ
            break
        if k not in labelled:
            keep.remove(k); total -= ntok[k]
    if total > BUDGET - 16:
        return None, None, "labelled_too_long"
    remap = {k: i for i, k in enumerate(keep)}
    c2 = dict(c); c2["passages"] = [c["passages"][k] for k in keep]
    if isinstance(c.get("groups"), list):
        c2["groups"] = [[remap[k] for k in g if k in remap] if isinstance(g, list) else g for g in c["groups"]]
    lab2 = dict(lab); lab2["spans"] = [[remap[k], a, b] for k, a, b in lab["spans"]]
    return c2, lab2, "trimmed"


for sp in A.splits.split(","):
    C = [json.loads(l) for l in open(f"{A.dir}/ctx_{sp}.jsonl")]
    L = {(d["dataset"], d["row"]): d for d in map(json.loads, open(f"{A.dir}/qw_{sp}.jsonl"))}
    C = [c for c in C if (c["dataset"], c["row"]) in L]
    st = Counter(); bylang = Counter(); oc, ol = [], []
    for i in range(0, len(C), BATCH):
        part = C[i:i + BATCH]
        encs = tk([build(c)[0] for c in part], return_offsets_mapping=True, add_special_tokens=False)
        for j, c in enumerate(part):
            e = {"input_ids": encs["input_ids"][j], "offset_mapping": encs["offset_mapping"][j]}
            c2, l2, why = fit(c, L[(c["dataset"], c["row"])], e)
            st[why or "kept"] += 1
            if why in ("unk", "labelled_too_long"):
                bylang[(c["dataset"], why)] += 1
                continue
            oc.append(c2); ol.append(l2)
    with open(f"{A.dir}/ctx_{sp}{A.suffix}.jsonl", "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in oc)
    with open(f"{A.dir}/qw_{sp}{A.suffix}.jsonl", "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in ol)
    n = len(C)
    print(f"{sp}{A.suffix}: {n} rows -> {len(oc)} | unchanged {st['kept']} | trimmed {st['trimmed']} | "
          f"dropped unk {st['unk']} | dropped labelled-too-long {st['labelled_too_long']} | "
          f"top drops {dict(Counter({f'{d}:{w}': v for (d, w), v in bylang.items()}).most_common(4))}", flush=True)
