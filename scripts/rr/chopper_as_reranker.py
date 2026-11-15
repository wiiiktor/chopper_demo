#!/usr/bin/env python3
"""Chopper as a reranker (user 2026-11-15, after Marek Kozłowski / OPI suggested a chopper head inside a reranker).

On the rerank holdout (rrval_notag: 2,000 rows x 16 passages, 8 languages, gold = passages holding luna-accepted answer
sentences) rank the 16 passages by
  - chopper (released universal mmBERT-base, ONE forward pass over all 16): max p(KEEP) in the passage, mean of its
    top-16 token p, and kept-token count at the 0.5 threshold
  - cross-encoder rerankers, one pair at a time: BAAI/bge-reranker-v2-m3, Qwen3-Reranker-0.6B, Qwen3-Reranker-4B
and report MRR, nDCG@10, Hit@1, Hit@3, Recall@3 on rows with >=1 gold passage, plus GPU seconds per row.
CAVEAT: the gold labels were only judged on passages Qwen3-Reranker-0.6B scored > 0.3 (plus the seed, a 5% audit and
generic rows read in full), so Qwen3-Reranker-0.6B is favoured by construction.

usage: chopper_as_reranker.py <out.json> [stride]"""
import json, math, sys, time
import numpy as np
import torch

sys.path.insert(0, "release/integrations/chopper-rag")
from chopper_rag.core import Chopper  # noqa: E402

PACK = "/media/wii/bigdata/rr/pack"
OUT = sys.argv[1]; STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1     # every STRIDE-th row (all 8 languages)
C = [json.loads(l) for l in open(f"{PACK}/ctx_rrval_notag.jsonl")][::STRIDE]
Q = {(d["dataset"], d["row"]): d for d in map(json.loads, open(f"{PACK}/qw_rrval_notag.jsonl"))}
rows = []
for c in C:
    gold = {k for k, _, _ in Q[(c["dataset"], c["row"])]["spans"]}
    rows.append((c["question"], [p["text"] for p in c["passages"]], gold, c["dataset"]))
dev = "cuda"
scores, secs = {}, {}


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- chopper: per-passage scores from one pass
ch = Chopper.from_pretrained(variant="base", device=dev)


@torch.no_grad()
def chop_scores(q, P):
    head = f"Question: {q}"
    out = {k: np.zeros(len(P)) for k in ("chop_max", "chop_top16", "chop_kept")}
    for group in ch._groups(head, ch._segments(head, P)):
        s, spans = head, []
        for (i, off, text, _) in group:
            s += ch.sep; spans.append((len(s), len(s) + len(text), i)); s += text
        e = ch.tokenizer(s, return_offsets_mapping=True, truncation=True, max_length=ch.max_length, return_tensors="pt")
        off = e.pop("offset_mapping")[0].tolist()
        sync(); t = time.time()
        h = ch.encoder(**{k: v.to(dev) for k, v in e.items()}).last_hidden_state[0].float()
        p = torch.sigmoid(h @ ch.heads["keep.weight"].T + ch.heads["keep.bias"])[:, 0].cpu().numpy()
        FWD[0] += time.time() - t                     # model time only (the per-passage scoring below is plain Python)
        for sa, sb, i in spans:
            idx = [k for k, (a, b) in enumerate(off) if b > a and a >= sa and b <= sb]
            if not idx:
                continue
            v = np.sort(p[idx])[::-1]
            out["chop_max"][i] = max(out["chop_max"][i], v[0])
            out["chop_top16"][i] = max(out["chop_top16"][i], v[:16].mean())
            out["chop_kept"][i] += (v > ch.threshold).sum()
    return out


FWD = [0.0]
t0 = time.time(); acc = {k: [] for k in ("chop_max", "chop_top16", "chop_kept")}
for q, P, _, _ in rows:
    o = chop_scores(q, P)
    for k in acc:
        acc[k].append(o[k].tolist())
sync(); dt = time.time() - t0
scores.update(acc); secs["chopper forward"] = FWD[0] / len(rows); secs["chopper incl. python scoring"] = dt / len(rows)
print(f"chopper {dt:.0f}s", flush=True)
del ch; torch.cuda.empty_cache()

# ---------------------------------------------------------------- bge-reranker-v2-m3 (cross-encoder)
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

tk = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
m = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16).to(dev).eval()
t0 = time.time(); S = []
with torch.no_grad():
    for q, P, _, _ in rows:
        e = tk([q] * len(P), P, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(dev)
        S.append(m(**e).logits[:, 0].float().cpu().tolist())
sync(); dt = time.time() - t0
scores["bge-reranker-v2-m3"] = S; secs["bge-reranker-v2-m3"] = dt / len(rows)
print(f"bge {dt:.0f}s", flush=True)
del m; torch.cuda.empty_cache()

# ---------------------------------------------------------------- Qwen3-Reranker (same prompt as scripts/rr/rr_rerank.py)
PRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
       "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INS = "Given a search query, does the passage contain information that answers it?"
for name in ("Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"):
    tk = AutoTokenizer.from_pretrained(name, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(dev).eval()
    YES, NO = tk.convert_tokens_to_ids("yes"), tk.convert_tokens_to_ids("no")
    sufi = tk(SUF, add_special_tokens=False)["input_ids"]
    bs = 16 if "0.6B" in name else 4
    t0 = time.time(); S = []
    with torch.no_grad():
        for q, P, _, _ in rows:
            ids = []
            for p in P:
                x = tk(PRE + f"<Instruct>: {INS}\n<Query>: {q}\n<Document>: ", add_special_tokens=False)["input_ids"]
                ids.append(x + tk(p, add_special_tokens=False)["input_ids"][:1700] + sufi)
            r = []
            for i in range(0, len(ids), bs):
                b = tk.pad({"input_ids": ids[i:i + bs]}, return_tensors="pt").to(dev)
                lg = m(**b).logits[:, -1, :].float()
                r += torch.log_softmax(lg[:, [NO, YES]], -1)[:, 1].exp().cpu().tolist()
            S.append(r)
    sync(); dt = time.time() - t0
    k = name.split("/")[1]; scores[k] = S; secs[k] = dt / len(rows)
    print(f"{k} {dt:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()


# ---------------------------------------------------------------- metrics
def metrics(S):
    r = {"MRR": [], "nDCG@10": [], "Hit@1": [], "Hit@3": [], "Recall@3": []}
    for (q, P, gold, _), s in zip(rows, S):
        if not gold:
            continue
        order = np.argsort(-np.array(s), kind="stable")
        ranks = [int(np.where(order == g)[0][0]) + 1 for g in gold]
        r["MRR"].append(1 / min(ranks))
        dcg = sum(1 / math.log2(k + 1) for k in ranks if k <= 10)
        idcg = sum(1 / math.log2(k + 2) for k in range(min(len(gold), 10)))
        r["nDCG@10"].append(dcg / idcg)
        r["Hit@1"].append(float(min(ranks) == 1)); r["Hit@3"].append(float(min(ranks) <= 3))
        r["Recall@3"].append(sum(k <= 3 for k in ranks) / len(gold))
    return {k: round(float(np.mean(v)), 4) for k, v in r.items()} | {"n": len(r["MRR"])}


res = {"overall": {k: metrics(v) for k, v in scores.items()}, "sec_per_row": secs, "by_lang": {}}
for lang in sorted({d for *_, d in rows}):
    keep = [i for i, r in enumerate(rows) if r[3] == lang]
    sub = rows
    rows = [sub[i] for i in keep]
    res["by_lang"][lang] = {k: metrics([v[i] for i in keep])["nDCG@10"] for k, v in scores.items()}
    rows = sub
json.dump(res | {"scores": scores}, open(OUT, "w"))
print(json.dumps({k: res[k] for k in ("overall", "sec_per_row", "by_lang")}, indent=1))
