# round 2: tournament variants + extra rounds, reported on all queries and on two fixed halves (choose on A, check on B)
import json, sys, hashlib, numpy as np
exec(open("/media/wii/bigdata/ab_misc/hn_combo.py").read().split("qs = list(rank)")[0])
qs = list(rank)
half = {q: int(hashlib.md5(q.encode()).hexdigest(), 16) % 2 for q in qs}
def nd(run, h=None):
    r = ev.evaluate({q: {d: float(s) for d, s in v.items()} for q, v in run.items() if h is None or half[q] == h})
    return round(float(np.mean([x["ndcg_cut_10"] for x in r.values()])), 4)
def report(name, run):
    line = f"{name} all {nd(run)} A {nd(run,0)} B {nd(run,1)}"; print(line, flush=True); open(f"{OUT}/results2.txt", "a").write(line + "\n")
blk = [{q: S(f"b{b}", q, rank[q][50*b:50*b+50]) for q in qs} for b in range(4)]
json.dump(cache, open(cache_f, "w"))
def top(sc, m): return sorted(sc, key=lambda d: -sc[d])[:m]
def stack(layers):  # layers: list of score dicts, later = finer; rank by the deepest layer a doc reached
    out = {}
    for depth, sc in enumerate(layers):
        for d, s in sc.items(): out[d] = depth * 10 + s
    return out
def tour(q, nb, m, extra=()):
    base = {k: v for b in blk[:nb] for k, v in b[q].items()}
    w = [d for b in blk[:nb] for d in top(b[q], m)]; w.sort(key=lambda d: rank[q].index(d))
    layers = [base, S(f"fin{nb}x{m}", q, w)]
    for r in extra:
        w = top(layers[-1], r); w.sort(key=lambda d: rank[q].index(d))
        layers.append(S(f"fin{nb}x{m}_r{'_'.join(map(str, extra[:extra.index(r)+1]))}", q, w))
    return stack(layers)
for nb, m, extra in ((1, 10, ()), (1, 20, ()), (2, 25, ()), (2, 15, ()), (2, 10, ()), (2, 8, ()), (2, 5, ()), (4, 12, ()), (4, 10, ()), (4, 5, ()),
                     (2, 10, (10,)), (2, 25, (10,)), (4, 12, (20,)), (4, 12, (20, 10)), (2, 25, (20, 10))):
    run = {q: tour(q, nb, m, extra) for q in qs}; json.dump(cache, open(cache_f, "w"))
    report(f"tour {nb}x50 top{m}" + (f" then {'->'.join(map(str, extra))}" if extra else ""), run)
print("COMBO2_DONE", flush=True)
