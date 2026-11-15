"""Reference for the JS port: per row the input string, HF ids/offsets, fp32 p(KEEP), p(ANSWER), relevance and chops."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "release/integrations/chopper-rag")
from chopper_rag.core import Chopper

variant, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ch = Chopper.from_pretrained(variant=variant, device="cpu")
rows = [json.loads(l) for l in open("/media/wii/bigdata/rr/pack/ctx_rrval_notag.jsonl")][7::67][:n]
ref = []
for r in rows:
    q = r["question"]; P = [p["text"] for p in r["passages"][:6]]
    res = ch.run(q, P)
    head = f"Question: {q}"; s = head
    for p in P:
        s += ch.sep + p.strip()
    e = ch.tokenizer(s, return_offsets_mapping=True)
    ref.append({"q": q, "P": P, "s": s, "ids": e["input_ids"], "off": e["offset_mapping"], "p_answer": res.p_answer,
                "rel": res.passage_scores, "chops": [[c.passage_index, c.start, c.end, c.text] for c in res.chops],
                "lang": r["dataset"]})
json.dump(ref, open(out, "w"), ensure_ascii=False)
print(len(ref), "rows")
