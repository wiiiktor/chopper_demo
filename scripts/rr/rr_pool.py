#!/usr/bin/env python3
"""Step 1a of the rerank chopper dataset: per language, read FineWeb(-2) row groups spread over one shard (text + url
only), chunk every document into passages of 100-300 words (200-600 characters for zh/ja) on paragraph boundaries,
keep at most `chunks_per_doc` per document. One chunker and normaliser for every passage (the live-vs-cached leak).
usage: rr_pool.py <out dir> [lang ...]   -> <out>/chunks_<lang>.jsonl {id, lang, url, text}"""
import json, os, re, sys, unicodedata
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

TX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.json")))
OUT = sys.argv[1]; LANGS = sys.argv[2:] or list(TX["languages"])
P = TX["pilot"]; CJK = {"zh", "ja"}


def norm(s):
    s = unicodedata.normalize("NFC", s).replace("\r", "")
    s = re.sub(r"[ \t ]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def size(s, lang):
    return len(re.sub(r"\s", "", s)) if lang in CJK else len(s.split())


def chunks(text, lang):
    lo, hi = TX["chunk"]["cjk_chars" if lang in CJK else "words"]
    out, cur = [], []
    for para in norm(text).split("\n"):
        para = para.strip()
        if not para:
            continue
        if size(para, lang) > hi:                       # one over-long paragraph: cut on sentence ends
            parts = re.split(r"(?<=[.!?。！？])\s*", para)
        else:
            parts = [para]
        for x in filter(None, parts):
            if cur and size(" ".join(cur + [x]), lang) > hi:
                if size(" ".join(cur), lang) >= lo:
                    out.append("\n".join(cur))
                cur = []
            cur.append(x)
    if cur and size("\n".join(cur), lang) >= lo:
        out.append("\n".join(cur))
    return [c for c in out if size(c, lang) <= hi * 1.2]


fs = HfFileSystem()
for lang in LANGS:
    src = TX["languages"][lang]["source"]
    path = ("datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet" if lang == "en" else
            f"datasets/HuggingFaceFW/fineweb-2/data/{src.split(':')[1]}/train/000_00000.parquet")
    pf = pq.ParquetFile(fs.open(path)); n = pf.metadata.num_row_groups
    need = P["pool_docs_per_language"] * (2 if lang == "en" else 1)
    per = pf.metadata.row_group(0).num_rows
    rgs = sorted({int(i * n / max(1, need // per)) for i in range(need // per)})
    o = f"{OUT}/chunks_{lang}.jsonl"; k = docs = 0
    with open(o + ".tmp", "w") as f:
        for j, rg in enumerate(rgs):
            t = pf.read_row_group(rg, columns=["text", "url"]).to_pylist()
            for d in t:
                docs += 1
                for i, c in enumerate(chunks(d["text"], lang)[:P["chunks_per_doc"]]):
                    f.write(json.dumps({"id": f"{lang}-{rg}-{docs}-{i}", "lang": lang, "url": d["url"], "text": c}, ensure_ascii=False) + "\n"); k += 1
            if j % 20 == 0:
                print(f"{lang}: row group {j + 1}/{len(rgs)} | {docs} docs -> {k} chunks", flush=True)
    os.rename(o + ".tmp", o)
    print(f"{lang}: DONE {docs} docs -> {k} chunks", flush=True)
