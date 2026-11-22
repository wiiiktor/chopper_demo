# user 2026-11-21 22:40: pass on uibk 1-20, pass on 21-100, winners of both -> final joint pass. Halves A/B as in combo2.
import json, hashlib, numpy as np
exec(open("/media/wii/bigdata/ab_misc/hn_combo2.py").read().split("blk = [")[0])
A = {q: S("r1_20", q, rank[q][:20]) for q in qs}; json.dump(cache, open(cache_f, "w")); print("A done", flush=True)
B = {q: S("r21_100", q, rank[q][20:100]) for q in qs}; json.dump(cache, open(cache_f, "w")); print("B done", flush=True)
def top(sc, m): return sorted(sc, key=lambda d: -sc[d])[:m]
report("1-20 single pass", A)
report("21-100 merged naive with 1-20", {q: {**A[q], **B[q]} for q in qs})
for ma, mb, extra in ((10, 10, ()), (20, 10, ()), (10, 5, ()), (15, 5, ()), (20, 20, ()), (10, 10, (10,)), (20, 10, (10,))):
    def f(q):
        w = top(A[q], ma) + top(B[q], mb); w.sort(key=lambda d: rank[q].index(d))
        layers = [{**A[q], **B[q]}, S(f"r20_80_fin{ma}_{mb}", q, w)]
        for r in extra:
            w2 = top(layers[-1], r); w2.sort(key=lambda d: rank[q].index(d)); layers.append(S(f"r20_80_fin{ma}_{mb}_r{r}", q, w2))
        out = {}
        for depth, sc in enumerate(layers):
            for d, s in sc.items(): out[d] = depth * 10 + s
        return out
    run = {q: f(q) for q in qs}; json.dump(cache, open(cache_f, "w"))
    report(f"split 1-20 top{ma} + 21-100 top{mb} -> final {ma+mb}" + (f" then {extra[0]}" if extra else ""), run)
print("COMBO3_DONE", flush=True)
