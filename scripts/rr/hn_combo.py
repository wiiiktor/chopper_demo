# user 2026-11-21 21:50: combine chopper passes over uibk's ranking: blocks of 50 (naive merge vs tournament final), multi-K fusion
# (20/40/80/160), fusion with the uibk rank. Scores via the submission wrapper's _score (same windows, same joint packing).
import importlib.util, json, sys, time, numpy as np, pytrec_eval
spec = importlib.util.spec_from_file_location("cm", "/home/wii/Projects/certain/scripts/rr/mteb_submission/chopper_models.py")
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
M = cm.chopper_031b.load_model()
D = "/media/wii/bigdata/rr/mteb/uibk"; OUT = "/media/wii/bigdata/rr/mteb/hn_combo"
import os; os.makedirs(OUT, exist_ok=True)
pred = json.load(open("/media/wii/bigdata/rr/mteb/first_stage_uibk/HotpotQAHardNegatives_predictions.json"))["default"]["test"]
qrels = json.load(open(f"{D}/qrels.json"))
Q = {j["id"]: j["text"] for j in map(json.loads, open(f"{D}/queries.jsonl"))}
rank = {q: [d for d, _ in sorted(v.items(), key=lambda x: -x[1])][:200] for q, v in pred.items() if q in qrels}
need = {d for r in rank.values() for d in r}
C = {}
for l in open(f"{D}/corpus.jsonl"):
    j = json.loads(l)
    if j["id"] in need: C[j["id"]] = j["text"]
ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
def ndcg(run): r = ev.evaluate({q: {d: float(s) for d, s in v.items()} for q, v in run.items()}); return round(float(np.mean([x["ndcg_cut_10"] for x in r.values()])), 4)
cache_f = f"{OUT}/scores.json"
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
def S(tag, q, docs):
    k = f"{tag}|{q}"
    if k not in cache: cache[k] = [float(x) for x in M._score(Q[q], [C[d] for d in docs])]
    return dict(zip(docs, cache[k]))
def report(name, run):
    line = f"{name} {ndcg(run)}"; print(line, flush=True); open(f"{OUT}/results.txt", "a").write(line + "\n")
def phase(tag, fn):
    t = time.time(); out = {q: fn(q) for q in rank}; json.dump(cache, open(cache_f, "w")); print(f"[{tag} {time.time()-t:.0f}s]", flush=True); return out
qs = list(rank)
report("uibk_alone", {q: {d: -i for i, d in enumerate(rank[q][:100])} for q in qs})
# blocks of 50 over the top 200, each its own pass (block 1 = the top-50 run)
blk = [phase(f"b{b}", lambda q, b=b: S(f"b{b}", q, rank[q][50*b:50*b+50])) for b in range(4)]
report("top50 (repro .8558)", blk[0])
for nb in (2, 4):
    report(f"naive merge of {nb} blocks (top{50*nb})", {q: {**{k: v for b in blk[:nb] for k, v in b[q].items()}} for q in qs})
# tournament: best m of each block -> one joint final pass
for nb, m in ((2, 25), (4, 12), (2, 10), (4, 5)):
    def fin(q, nb=nb, m=m):
        w = [d for b in blk[:nb] for d in sorted(b[q], key=lambda d: -b[q][d])[:m]]
        w.sort(key=lambda d: rank[q].index(d))
        return S(f"fin{nb}x{m}", q, w)
    report(f"tournament {nb} blocks x top{m} -> final", phase(f"fin{nb}x{m}", fin))
# fusion with the uibk rank (RRF, reciprocal rank fusion, k=60)
for w in (0.25, 0.5, 1.0):
    for nb in (1, 2):
        run = {}
        for q in qs:
            sc = {k: v for b in blk[:nb] for k, v in b[q].items()}
            cr = {d: i for i, d in enumerate(sorted(sc, key=lambda d: -sc[d]))}
            run[q] = {d: 1/(60+cr[d]) + w/(60+rank[q].index(d)) for d in sc}
        report(f"rrf chopper(top{50*nb}) + {w}*uibk", run)
# multi-K: separate runs at K = 20, 40, 80, 160, fused by mean reciprocal rank
runs = {K: phase(f"k{K}", lambda q, K=K: S(f"k{K}", q, rank[q][:K])) for K in (20, 40, 80, 160)}
for name, Ks in (("20+40+80+160", (20, 40, 80, 160)), ("20+40+80", (20, 40, 80)), ("20+40", (20, 40)), ("20+50", (20,))):
    run = {}
    for q in qs:
        rr = {}
        sets = [runs[K][q] for K in Ks] + ([blk[0][q]] if name == "20+50" else [])
        for sc in sets:
            for i, d in enumerate(sorted(sc, key=lambda d: -sc[d])): rr[d] = rr.get(d, 0) + 1/(60+i)
        run[q] = rr
    report(f"multiK {name} (mean RR)", run)
for K in (20, 40, 80, 160): report(f"single K{K}", runs[K])
print("COMBO_DONE", flush=True)
