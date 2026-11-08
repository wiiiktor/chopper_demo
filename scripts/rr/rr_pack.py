#!/usr/bin/env python3
"""Packs the rerank pilot's labelled rows into mm_train.py's ctx_<tag>/qw_<tag> format (user 2026-11-05). A passage = optional metadata tag + text;
a selected passage's KEEP spans = its whole tag + the chosen sentences (taxonomy labels). Tag on for half of the training rows; each field dropped
independently at FIELD_DROP for EVERY passage, so field presence says nothing about relevance. No-answer rows have no spans and decision SEARCH.
Holdout rows are written four ways for the leak audit on identical rows: rrval_tag, rrval_notag, rrval_swap (tags shuffled within the pool, spans
on the bodies), rrval_tagonly (bodies replaced by neutral filler, no spans). Also: old20 = 20% of the old 54k rows, symlinks to the benchmark val sets.
usage: rr_pack.py <rr dir> <mm dir> <out dir> [lang ...]  (env V=v2, HOLD=250 rows per language, FIELD_DROP=0.3)"""
import json, os, random, re, sys
from collections import Counter
D, MM, O = sys.argv[1:4]; LANGS = sys.argv[4:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]
V = os.environ.get("V", "v2"); HOLD = int(os.environ.get("HOLD", "250")); DROP = float(os.environ.get("FIELD_DROP", "0.3"))
os.makedirs(O, exist_ok=True); rng = random.Random(0)
SEP_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n+")
FILLER = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."


def sent_spans(t):
    """character spans of rr_sentences.sents(t), same pieces, same merging of stray short pieces"""
    out, pos = [], 0
    for m in list(SEP_RE.finditer(t)) + [None]:
        a, b = pos, (m.start() if m else len(t)); pos = m.end() if m else len(t)
        raw = t[a:b]; s = raw.strip()
        if not s:
            continue
        a += len(raw) - len(raw.lstrip()); b = a + len(s)
        if out and len(s) < 4:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def tag(p, lang):
    m = p["meta"]; f = [m.get("domain") or None, m.get("title"), m.get("section"), m.get("author"), m.get("date"), m.get("type"), lang]
    f = [str(x) for x in f if x and rng.random() >= DROP]
    return "<src>" + " | ".join(f) + "</src>\n" if f else ""


def row(r, n, lang, mode, stats):
    ps, spans = [], []
    tags = [tag(p, lang) if mode in ("tag", "swap", "tagonly") else "" for p in r["passages"]]
    if mode == "swap":
        rng.shuffle(tags)
    for k, p in enumerate(r["passages"]):
        tg = tags[k]; body = FILLER if mode == "tagonly" else p["text"]
        ps.append({"pos": k, "cid": -1, "text": tg + body, "group": -1})
        if mode == "tagonly" or not p["keep"]:
            continue
        ss = sent_spans(p["text"])
        if len(ss) != len(p["sents"]):
            stats["sentence_mismatch"] += 1; continue
        if tg and mode != "swap":
            spans.append([k, 0, len(tg)])
        spans += [[k, len(tg) + ss[j][0], len(tg) + ss[j][1]] for j in p["keep"]]
    sds = r["passages"][r["seed"]]
    ans = " ".join(sds["sents"][j] for j in sds["keep"])
    c = {"dataset": f"rr_{lang}", "split": "rr", "row": n, "question": r["query"], "answer": ans, "groups": [], "passages": ps,
         "allin": 0.0 if r["qtype"] == "no_answer" else 1.0, "qtype": r["qtype"], "kind": sds["kind"], "area": r["area"], "qlang": r["qlang"]}
    return c, {"dataset": c["dataset"], "split": "rr", "row": n, "spans": spans, "steps": [], "miss": []}


def write(name, pairs):
    with open(f"{O}/ctx_{name}.jsonl", "w") as f, open(f"{O}/qw_{name}.jsonl", "w") as g:
        for c, q in pairs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n"); g.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{name}: {len(pairs)} rows", flush=True)


st = Counter(); TR, VAL = [], {m: [] for m in ("tag", "notag", "swap", "tagonly")}
for lang in LANGS:
    R = [json.loads(l) for l in open(f"{D}/lab_{V}_{lang}{os.environ.get('SUF', '')}.jsonl")]
    rng.shuffle(R)
    hold, tr = R[:HOLD], R[HOLD:]
    for n, r in enumerate(hold):
        for m in VAL:
            VAL[m].append(row(r, n, lang, m, st))
    for n, r in enumerate(tr):
        TR.append(row(r, HOLD + n, lang, "tag" if rng.random() < .5 else "notag", st))
    print(lang, len(hold), "held out", len(tr), "train", Counter(r["qtype"] for r in R).most_common(), flush=True)
write("rrtrain", TR)
for m, x in VAL.items():
    write(f"rrval_{m}", x)
if os.environ.get("SKIP_OLD") != "1":
    old = []
    for sp in ("train", "train2", "train3", "wtrain"):
        C = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{MM}/ctx_{sp}.jsonl"))}
        for q in map(json.loads, open(f"{MM}/qw_{sp}.jsonl")):
            if (q["dataset"], q["row"]) in C:
                old.append((C[(q["dataset"], q["row"])], q))
    write("old20", rng.sample(old, len(old) // 5))
    for v in ("val", "wval"):
        for pre in ("ctx", "qw"):
            if not os.path.exists(f"{O}/{pre}_{v}.jsonl"):
                os.symlink(f"{MM}/{pre}_{v}.jsonl", f"{O}/{pre}_{v}.jsonl")
print("stats", dict(st), flush=True)
