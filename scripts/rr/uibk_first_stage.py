# user 2026-11-21: uibk-embedding-v1 (DataScience-UIBK, Apache 2.0, ColBERT on ModernBERT-base) as first stage for HotpotQA-HN.
# Exact brute-force MaxSim over the 225,621-doc corpus; writes MTEB first-stage predictions (top 1000) + recall@K.
import json, torch, numpy as np, time
from pylate import models
O = "/media/wii/bigdata/rr/mteb/uibk"
C = [json.loads(l) for l in open(f"{O}/corpus.jsonl")]; Q = [json.loads(l) for l in open(f"{O}/queries.jsonl")]
m = models.ColBERT("DataScience-UIBK/uibk-embedding-v1", trust_remote_code=True, device="cuda")
# no m.half(): fp16 model gives NaN embeddings (checked 17:30); model fp32, scoring fp16
t0 = time.time()
qe = m.encode([q["text"] for q in Q], is_query=True, prompt_name="query", batch_size=64, convert_to_tensor=False)
qe = [torch.as_tensor(x, dtype=torch.float16) for x in qe]
Lq = max(x.shape[0] for x in qe); QE = torch.zeros(len(qe), Lq, qe[0].shape[1], dtype=torch.float16)
for i, x in enumerate(qe): QE[i, :x.shape[0]] = x          # zero rows add 0 to MaxSim sum
QE = QE.cuda()
order = sorted(range(len(C)), key=lambda i: len(C[i]["text"]))   # length-sorted chunks = little padding
K = 1000; top_s = torch.full((len(Q), K), -1e9, device="cuda"); top_i = torch.zeros((len(Q), K), dtype=torch.long, device="cuda")
CH = 2048
for s in range(0, len(order), CH):
    idx = order[s:s + CH]
    de = m.encode([C[i]["text"] for i in idx], is_query=False, prompt_name="document", batch_size=128, convert_to_tensor=False)
    de = [torch.as_tensor(x, dtype=torch.float16) for x in de]
    Ld = max(x.shape[0] for x in de); D = torch.zeros(len(de), Ld, de[0].shape[1], dtype=torch.float16)
    mask = torch.zeros(len(de), Ld, dtype=torch.bool)
    for j, x in enumerate(de): D[j, :x.shape[0]] = x; mask[j, :x.shape[0]] = True
    D = D.cuda(); mask = mask.cuda()
    sc = torch.empty(len(Q), len(de), device="cuda")
    QB = max(1, int(2.5e9 / (len(de) * Lq * Ld * 2)))
    for qb in range(0, len(Q), QB):
        sim = torch.einsum("qad,nbd->qnab", QE[qb:qb + QB], D)            # q,n,Lq,Ld
        sim = sim.masked_fill(~mask[None, :, None, :], -1e4)
        sc[qb:qb + QB] = torch.nan_to_num(sim.max(-1).values.float(), nan=0.0).clamp(min=0).sum(-1)
    allS = torch.cat([top_s, sc], 1); allI = torch.cat([top_i, torch.as_tensor(idx, device="cuda")[None].expand(len(Q), -1)], 1)
    v, p = allS.topk(K, dim=1); top_s = v; top_i = allI.gather(1, p)
    if s // CH % 10 == 0: print(f"{s + len(idx)}/{len(C)} docs, {time.time() - t0:.0f}s", flush=True)
top_s = top_s.cpu().numpy(); top_i = top_i.cpu().numpy()
pred = {Q[a]["id"]: {C[top_i[a, b]]["id"]: float(top_s[a, b]) for b in range(K)} for a in range(len(Q))}
meta = json.load(open("/media/wii/bigdata/rr/mteb/first_stage/HotpotQAHardNegatives_predictions.json")).get("mteb_model_meta")
json.dump({"mteb_model_meta": meta, "default": {"test": pred}}, open("/media/wii/bigdata/rr/mteb/first_stage_uibk/HotpotQAHardNegatives_predictions.json", "w"))
rel = json.load(open(f"{O}/qrels.json"))
for k in (10, 50, 100, 200, 1000):
    rec = []; allin = 0
    for q, r in rel.items():
        top = set(list(dict(sorted(pred[q].items(), key=lambda z: -z[1])).keys())[:k]); g = {x for x, s in r.items() if s > 0}
        h = len(g & top); rec.append(h / len(g)); allin += h == len(g)
    print(f"UIBK recall@{k} {np.mean(rec):.4f} all-in {allin / len(rec):.4f}", flush=True)
print(f"done {time.time() - t0:.0f}s")
