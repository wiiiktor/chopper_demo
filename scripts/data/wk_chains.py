#!/usr/bin/env python
"""Benchmark-free multi-hop CANDIDATES from Wikipedia (2026-10-30, user: "start generating 27k training data samples
using wikipedia"). wikimedia/wikipedia 20231101.en has no hyperlinks, so a bridge is an exact, case-sensitive mention of
another article's TITLE in one of the first 3 paragraphs of article A (n-grams of 1-6 words starting with a capital;
single-word titles only when ≥ 5 letters and not the sentence's first word). Every B is used at most --cap times, so
"United States" cannot dominate.
Three passes over the 41 parquet shards:
  1. titles of "good" articles (lead ≥ 250 chars, no lists / disambiguation pages)
  2. a seeded sample of A articles → their mentions → one B per A (2-hop), and a separate sample for 1-hop questions
  3. the first --keep-paras paragraphs of every A, B, and of up to 8 other titles each A mentions (hard negatives),
     plus --pad random leads (easy padding)
Writes <out>/cand.jsonl (one candidate per line), <out>/articles.jsonl (title → paragraphs), <out>/pad.jsonl.
Split: val if crc32(A title) % 50 == 0; train rows never use a val row's A or B.
usage: .venv-rwkv7/bin/python wk_chains.py <parquet dir> <out dir> [--n2 90000 --n1 30000 --pad 200000 --cap 15]"""
import glob, json, os, random, re, sys, time, zlib
from collections import Counter
import pyarrow.parquet as pq
SRC, OUT = sys.argv[1:3]
O = {"n2": 90000, "n1": 30000, "pad": 200000, "cap": 15, "keep_paras": 6}
for k in O:
    if f"--{k.replace('_', '-')}" in sys.argv:
        O[k] = int(sys.argv[sys.argv.index(f"--{k.replace('_', '-')}") + 1])
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
SH = sorted(glob.glob(f"{SRC}/*.parquet"))
if os.environ.get("WK_SMOKE"):                                           # smoke test on the first shards
    SH = SH[:int(os.environ["WK_SMOKE"])]
else:
    assert len(SH) == 41, f"{len(SH)} shards — download not finished?"
BAD = re.compile(r"^(List of|Lists of|Index of|Outline of|Timeline of|Glossary of)|\(disambiguation\)|^\d+$")


def paras(text, n=None):
    """body paragraphs: '\n\n'-separated, headings and stubs (< 150 chars) dropped"""
    out = []
    for p in text.split("\n\n"):
        p = p.strip()
        if len(p) >= 150 and "\n" not in p[:80]:
            out.append(p[:1500])
            if n and len(out) >= n:
                break
    return out


def shards():
    for f in SH:
        t = pq.read_table(f, columns=["title", "text"])
        yield from zip(t.column("title").to_pylist(), t.column("text").to_pylist())


# ---- pass 1: good titles
t0 = time.time(); good = {}
for ti, tx in shards():
    if BAD.search(ti) or "may refer to" in tx[:300]:
        continue
    lead = paras(tx, 1)
    if lead and len(lead[0]) >= 250:
        good[ti] = len(good)
log(f"pass 1: {len(good):,} good articles | {time.time() - t0:.0f}s")
titles = list(good)
rng = random.Random(0)
order = rng.sample(range(len(titles)), O["n2"] * 2 + O["n1"])            # 2× for the A's that find no usable mention
A2 = set(order[:O["n2"] * 2]); A1 = set(order[O["n2"] * 2:])
isval = lambda t: zlib.crc32(t.encode()) % 50 == 0


DATEB = re.compile(r"^((January|February|March|April|May|June|July|August|September|October|November|December) )?\d{1,4}s?( |$)")
LOWER = {l.strip() for l in open("/usr/share/dict/words") if l[:1].islower()} | {
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}


def mentions(par, self_title):
    w = [(m.group(0), m.start()) for m in re.finditer(r"\S+", par)]
    out, i = [], 0
    while i < len(w):
        hit = None
        if w[i][0][:1].isupper():
            for L in range(min(6, len(w) - i), 0, -1):
                s0, s1 = w[i][1], w[i + L - 1][1] + len(w[i + L - 1][0])
                cand = par[s0:s1].rstrip(".,;:)\"'”’").lstrip("(\"'“‘")
                if cand != self_title and cand in good and not DATEB.match(cand):
                    if L == 1 and (len(cand) < 5 or i == 0 or w[i - 1][0].endswith(".") or cand.lower() in LOWER):
                        continue
                    # part of a longer name ("James Earl" of "James Earl Jones") → no bridge here
                    if i + L < len(w) and w[i + L][0][:1].isupper() and not w[i + L - 1][0][-1] in ".,;:)":
                        break
                    if i > 0 and w[i - 1][0][:1].isupper() and not w[i - 1][0][-1] in ".,;:":
                        break
                    hit = (cand, L); break
        if hit:
            out.append(hit[0]); i += hit[1]
        else:
            i += 1
    return out


# ---- pass 2: A samples → mentions
t0 = time.time(); m2, one = {}, []
for ti, tx in shards():
    gi = good.get(ti)
    if gi is None or (gi not in A2 and gi not in A1):
        continue
    ps = paras(tx, 3)
    if gi in A1:
        one.append(ti); continue
    ms = [(k, b) for k, p in enumerate(ps) for b in mentions(p, ti)]
    if ms:
        m2[ti] = ms
log(f"pass 2: {len(m2):,} A articles with a mention, {len(one):,} 1-hop articles | {time.time() - t0:.0f}s")
use, cand, need = Counter(), [], set()
for ti in rng.sample(sorted(m2), len(m2)):
    ms = [(k, b) for k, b in m2[ti] if use[b] < O["cap"]]
    if not ms:
        continue
    k, b = min(ms, key=lambda x: (use[x[1]], rng.random()))               # the least-used B first
    use[b] += 1
    others = list(dict.fromkeys(x for _, x in m2[ti] if x != b))[:8]
    cand.append({"type": "2hop", "a": ti, "a_pi": k, "b": b, "neg": others, "split": "val" if isval(ti) else "train"})
    need |= {ti, b, *others}
    if len(cand) >= O["n2"]:
        break
VAL = {c["a"] for c in cand if c["split"] == "val"} | {c["b"] for c in cand if c["split"] == "val"}
cand = [c for c in cand if c["split"] == "val" or (c["a"] not in VAL and c["b"] not in VAL)]
for ti in one:
    if ti not in VAL:
        cand.append({"type": "1hop", "a": ti, "split": "val" if isval(ti) else "train"}); need.add(ti)
rng.shuffle(cand)
pad = set(rng.sample(range(len(titles)), O["pad"] + 50000))
pad = [t for t in (titles[i] for i in pad) if t not in need][:O["pad"]]
need |= set(pad)
log(f"{sum(c['type'] == '2hop' for c in cand):,} 2-hop + {sum(c['type'] == '1hop' for c in cand):,} 1-hop candidates "
    f"({sum(c['split'] == 'val' for c in cand)} val) | {len(use):,} distinct B, most used {use.most_common(3)}")

# ---- pass 3: paragraphs
t0 = time.time(); padset = set(pad); n = 0
with open(f"{OUT}/articles.jsonl", "w") as fa, open(f"{OUT}/pad.jsonl", "w") as fp:
    for ti, tx in shards():
        if ti not in need:
            continue
        if ti in padset:
            fp.write(json.dumps({"t": ti, "p": paras(tx, 1)}, ensure_ascii=False) + "\n")
        else:                                                                # pad titles were drawn outside `need`
            fa.write(json.dumps({"t": ti, "p": paras(tx, O["keep_paras"])}, ensure_ascii=False) + "\n"); n += 1
with open(f"{OUT}/cand.jsonl", "w") as fo:
    for c in cand:
        fo.write(json.dumps(c, ensure_ascii=False) + "\n")
log(f"pass 3: {n:,} articles, {len(pad):,} padding leads | {time.time() - t0:.0f}s | CHAINS DONE")
