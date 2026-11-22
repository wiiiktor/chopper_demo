# user 2026-11-21 23:40: blocks 1-32 and 33-96, winners -> final -> (re-read top 10). Halves A/B as in combo2.
import json, hashlib, numpy as np
exec(open("/media/wii/bigdata/ab_misc/hn_combo2.py").read().split("blk = [")[0])
tok = M.model.tokenizer
fit = [sum(len(tok(C[d][:900], add_special_tokens=False)["input_ids"]) + 2 for d in rank[q][32:96]) for q in qs]
print("33-96 tokens median", int(np.median(fit)), "share >8150", round(float(np.mean(np.array(fit) > 8150)), 3), flush=True)
A = {q: S("r1_32", q, rank[q][:32]) for q in qs}; json.dump(cache, open(cache_f, "w")); print("A done", flush=True)
B = {q: S("r33_96", q, rank[q][32:96]) for q in qs}; json.dump(cache, open(cache_f, "w")); print("B done", flush=True)
def top(sc, m): return sorted(sc, key=lambda d: -sc[d])[:m]
report("[32/64] 1-32 single pass", A)
report("[32/64] naive merge", {q: {**A[q], **B[q]} for q in qs})
for ma, mb, extra in ((10, 10, ()), (16, 16, ()), (25, 25, ()), (16, 8, ()), (20, 10, ()),
                      (10, 10, (10,)), (16, 16, (10,)), (25, 25, (10,)), (16, 8, (10,)), (20, 10, (10,)), (25, 25, (20, 10))):
    def f(q):
        w = top(A[q], ma) + top(B[q], mb); w.sort(key=lambda d: rank[q].index(d))
        layers = [{**A[q], **B[q]}, S(f"r32_64_fin{ma}_{mb}", q, w)]
        for i, r in enumerate(extra):
            w2 = top(layers[-1], r); w2.sort(key=lambda d: rank[q].index(d))
            layers.append(S(f"r32_64_fin{ma}_{mb}_r{'_'.join(map(str, extra[:i+1]))}", q, w2))
        out = {}
        for depth, sc in enumerate(layers):
            for d, s in sc.items(): out[d] = depth * 10 + s
        return out
    run = {q: f(q) for q in qs}; json.dump(cache, open(cache_f, "w"))
    report(f"[32/64] top{ma} + top{mb} -> final {ma+mb}" + (f" then {'->'.join(map(str, extra))}" if extra else ""), run)
print("COMBO4_DONE", flush=True)
