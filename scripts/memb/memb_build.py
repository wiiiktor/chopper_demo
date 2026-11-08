#!/usr/bin/env python3
"""Long-conversation memory benchmarks for the chopper + native tool-call loop (user 2026-11-08: "Add LoCoMo and LongMemEval").
Every session is cut into passages of consecutive turns (~120 words, a long turn split on sentences), each with a <src> tag carrying the
session date, the format the rerank chopper was trained on ("<src>a | b</src>\\n" + body). A question searches only its own history
(LoCoMo: its conversation; LongMemEval: its ~50-session haystack). Unanswerable questions (LoCoMo category 5, LongMemEval *_abs) are kept,
their reference = "not mentioned". Evidence chunks (LoCoMo dia_ids, LongMemEval has_answer turns) give retrieval recall.
Writes <out>/chunks.json [text], <out>/hay.json {hay: [chunk ids]}, <out>/qs.jsonl.
usage: memb_build.py <memb dir> [--n 300]   (seed 0, stratified by category)"""
import json, random, re, sys
from collections import defaultdict
D = sys.argv[1]; N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
NOT = "Not mentioned in the conversation: the correct reply says this is unknown / was never mentioned."
LOCO_CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]


def chunk(turns, tag, target=120):
    """turns = [(speaker, text, evidence id or None)] -> [(text, {evidence ids})]; a passage never crosses a session"""
    units = []
    for sp, tx, ev in turns:
        w = tx.split()
        if len(w) <= 1.5 * target:
            units.append((f"{sp}: {tx.strip()}", len(w), ev)); continue
        cur, n, first = [], 0, True                      # long turn: sentence groups, each piece names the speaker
        for s in sents(tx):
            cur.append(s); n += len(s.split())
            if n >= target:
                units.append((f"{sp}{'' if first else ' (cont.)'}: {' '.join(cur)}", n, ev)); cur, n, first = [], 0, False
        if cur:
            units.append((f"{sp}{'' if first else ' (cont.)'}: {' '.join(cur)}", n, ev))
    out, cur, n, evs = [], [], 0, set()
    for u, k, ev in units:
        if cur and n + k > target:
            out.append((tag + "\n".join(cur), evs)); cur, n, evs = [], 0, set()
        cur.append(u); n += k
        if ev:
            evs.add(ev)
    if cur:
        out.append((tag + "\n".join(cur), evs))
    return out


def strat(rows, key, n, rng):
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, cats = [], sorted(by)
    while len(out) < n and any(by.values()):              # round-robin = equal per category while each lasts
        for c in cats:
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    return out


for bench in ("locomo", "lme"):
    rng = random.Random(0); C, H, Q = [], {}, []
    if bench == "locomo":
        for x in json.load(open(f"{D}/locomo10.json")):
            cv = x["conversation"]; ids = []; ev2c = defaultdict(list)
            ss = sorted((int(k.split("_")[1]) for k in cv if re.fullmatch(r"session_\d+", k)))
            for s in ss:
                turns = [(t["speaker"], t["text"] + (f" (shares a photo: {t['blip_caption']})" if t.get("blip_caption") else ""), t["dia_id"]) for t in cv[f"session_{s}"]]
                for tx, evs in chunk(turns, f"<src>{cv[f'session_{s}_date_time']} | session {s}</src>\n"):
                    for e in evs:
                        ev2c[e].append(len(C))
                    ids.append(len(C)); C.append(tx)
            H[x["sample_id"]] = ids
            for k, q in enumerate(x["qa"]):
                cat = LOCO_CAT[q["category"]]
                Q.append({"qid": f"{x['sample_id']}_{k}", "bench": bench, "cat": cat, "question": q["question"],
                          "gold": NOT if cat == "adversarial" else str(q["answer"]), "date": "", "hay": x["sample_id"],
                          "ev": sorted({c for e in q.get("evidence", []) for c in ev2c.get(e, [])}) if cat != "adversarial" else []})
        Q = strat(Q, "cat", N, rng)
    else:
        X = json.load(open(f"{D}/longmemeval_s_cleaned.json"))
        for x in X:
            x["cat"] = x["question_type"] + ("-abs" if x["question_id"].endswith("_abs") else "")
        X = strat(X, "cat", N, rng); seen = {}
        for x in X:
            ids, ev = [], []
            for sid, date, sess in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
                if sid not in seen:
                    turns = [("User" if t["role"] == "user" else "Assistant", t["content"], "A" if t.get("has_answer") else None) for t in sess]
                    seen[sid] = []
                    for tx, evs in chunk(turns, f"<src>{date} | chat {sid[-8:]}</src>\n"):
                        seen[sid].append((len(C), bool(evs))); C.append(tx)
                ids += [c for c, _ in seen[sid]]
                if sid in x["answer_session_ids"]:
                    ev += [c for c, a in seen[sid] if a]
            H[x["question_id"]] = ids
            Q.append({"qid": x["question_id"], "bench": bench, "cat": x["cat"], "question": x["question"],
                      "gold": NOT if x["cat"].endswith("-abs") else str(x["answer"]), "date": x["question_date"], "hay": x["question_id"], "ev": ev})
    import os; os.makedirs(f"{D}/{bench}", exist_ok=True)
    json.dump(C, open(f"{D}/{bench}/chunks.json", "w"), ensure_ascii=False); json.dump(H, open(f"{D}/{bench}/hay.json", "w"))
    with open(f"{D}/{bench}/qs.jsonl", "w") as f:
        f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in Q)
    cats = defaultdict(int)
    for q in Q:
        cats[q["cat"]] += 1
    print(f"{bench}: {len(C)} chunks, {len(H)} histories ({sum(map(len, H.values())) / len(H):.0f} chunks each), {len(Q)} questions {dict(cats)}, "
          f"words/chunk {sum(len(c.split()) for c in C) / len(C):.0f}, questions with evidence {sum(bool(q['ev']) for q in Q)}", flush=True)
