#!/usr/bin/env python
"""Clean top-16 contexts for the mmBERT chopper (2026-10-29). The text retriever of the RAG baseline (ab_rag_baseline.py)
on the mm_dump_pools.py pools: Qwen3-Embedding-0.6B dense over the question's 1,000 chunks → top --R → bge-reranker-v2-m3 →
top --K. Text only, every chunk through ONE path (no live-vs-cached fingerprint — see memory live-vs-cached-leak). Each
distinct chunk is embedded once (shared table), and the embeddings are saved for the later <search> step.
Writes per question: dataset, split, row, question, answer, passages [{cid, text, group (−1 = not gold)}] in rerank order,
dense100 (chunk ids, for re-ranking a second query later), allin (every gold group has a passage in the top K).
usage: .venv-gen/bin/python mm_contexts.py <mm dir> <split> [--K 16] [--R 100]"""
import sys, json, time, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

D_, SPL = sys.argv[1], sys.argv[2]
K = int(sys.argv[sys.argv.index("--K") + 1]) if "--K" in sys.argv else 16
R = int(sys.argv[sys.argv.index("--R") + 1]) if "--R" in sys.argv else 100
dev = "cuda"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
TASK = "Given a question, retrieve passages that contain facts needed to answer it"
table = json.load(open(f"{D_}/chunks_{SPL}.json"))
pools = [json.loads(l) for l in open(f"{D_}/pools_{SPL}.jsonl")]
log(f"{len(pools)} pools, {len(table)} distinct chunks")

etk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
emo = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(dev).eval()


@torch.no_grad()
def embed(texts, query=False, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        t = [f"Instruct: {TASK}\nQuery:{x}" for x in texts[i:i + bs]] if query else texts[i:i + bs]
        b = etk(t, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
        out.append(F.normalize(emo(**b).last_hidden_state[:, -1].float(), dim=-1).half())
    return torch.cat(out)


EP = f"{D_}/emb_{SPL}.pt"
if os.path.exists(EP):
    E = torch.load(EP).to(dev)
else:
    t0 = time.time(); order = np.argsort([len(t) for t in table])      # length-sorted batches: little padding
    E = torch.empty((len(table), emo.config.hidden_size), dtype=torch.half, device=dev)
    for i in range(0, len(order), 4096):
        idx = order[i:i + 4096]; E[torch.tensor(idx, device=dev)] = embed([table[k] for k in idx])
        log(f"embedded {min(i + 4096, len(order))}/{len(order)} | {time.time() - t0:.0f}s")
    torch.save(E.cpu(), EP)
Q = embed([p["question"] for p in pools], query=True)

rtk = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rmo = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.bfloat16).to(dev).eval()


@torch.no_grad()
def rerank(q, docs, bs=50):
    s = []
    for i in range(0, len(docs), bs):
        b = rtk([[q, d] for d in docs[i:i + bs]], padding=True, truncation=True, max_length=640, return_tensors="pt").to(dev)
        s.append(rmo(**b).logits.view(-1).float())
    return torch.cat(s).cpu()


t0 = time.time(); stat = {}
with open(f"{D_}/ctx_{SPL}.jsonl", "w") as fo:
    for pi, p in enumerate(pools):
        ch = p["chunks"]; gof = {i: g for g, grp in enumerate(p["groups"]) for i in grp}
        sc = (E[torch.tensor(ch, device=dev)].float() @ Q[pi].float())
        dense = sc.argsort(descending=True).tolist()
        top = dense[:R]; rs = rerank(p["question"], [table[ch[i]] for i in top])
        rer = [top[k] for k in rs.argsort(descending=True).tolist()][:K]
        allin = float(all(any(i in rer for i in grp) for grp in p["groups"] if grp)) if any(p["groups"]) else float("nan")
        fo.write(json.dumps({"dataset": p["dataset"], "split": SPL, "row": p["row"], "question": p["question"],
                             "answer": p["answer"], "groups": p["groups"],
                             "passages": [{"pos": i, "cid": ch[i], "text": table[ch[i]], "group": gof.get(i, -1)} for i in rer],
                             "dense100": [ch[i] for i in dense[:R]], "allin": allin}, ensure_ascii=False) + "\n")
        stat.setdefault(p["dataset"], []).append(allin)
        if (pi + 1) % 200 == 0:
            log(f"{pi + 1}/{len(pools)} | {(time.time() - t0) / (pi + 1):.2f}s/q | all-hops@{K} so far "
                + " ".join(f"{k} {np.nanmean(v):.2f}" for k, v in stat.items()))
log("CONTEXTS DONE | all-hops@%d " % K + " ".join(f"{k} {np.nanmean(v):.3f} (n={len(v)})" for k, v in stat.items()))
