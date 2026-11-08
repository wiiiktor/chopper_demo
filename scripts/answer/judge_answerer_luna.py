#!/usr/bin/env python3
"""gpt-5.6-luna verdicts for the MiniCPM3-4B answerer's val predictions (2026-11-04, user: "run luna verification
instead of exact that is misleading"). For every preds_<step>.jsonl in each run dir, every answer-class question
(all gold passages among the 16) is judged against the reference; a "[search]" reply on one counts as wrong, as in
exact. Verdicts are cached on (question, gold, pred) in <mm>/luna_answer_cache.jsonl, so a re-run only pays for new
predictions. Writes <run>/judged.jsonl: {step, notes, dataset, n, judged, unparsed}.
usage: judge_answerer_luna.py <mm dir> <run dir> [<run dir> ...]   (env NOTES=chopper,qwen picks the note sources)"""
import json, os, re, sys
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_mh_luna import QA, ask
from openai import OpenAI

MM, RUNS = sys.argv[1], sys.argv[2:]
NOTES = os.environ.get("NOTES", "chopper,qwen").split(",")
ALLQ = os.environ.get("ALLQ") == "1"                     # 2026-11-08: judge every question, not only those with all gold passages among the first 16
CV = {(c["dataset"], c["row"]): c for c in map(json.loads, open(f"{MM}/ctx_val.jsonl"))}
CACHE_F = f"{MM}/luna_answer_cache.jsonl"
CACHE = {}
if os.path.exists(CACHE_F):
    for x in map(json.loads, open(CACHE_F)):
        CACHE[(x["q"], x["gold"], x["pred"])] = x["v"]
cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
stats = {"in": 0, "out": 0, "unparsed": 0, "errors": 0}

todo = {}
for run in RUNS:
    for f in os.listdir(run):
        if re.fullmatch(r"preds_\d+\.jsonl", f):
            for p in map(json.loads, open(os.path.join(run, f))):
                c = CV[(p["dataset"], p["row"])]
                if p["notes"] in NOTES and (ALLQ or c["allin"] == 1.0) and not p["pred"].startswith("[search]"):
                    k = (c["question"], c["answer"], p["pred"])
                    if k not in CACHE:
                        todo[k] = 1
print(f"{len(todo)} new verdicts to ask ({len(CACHE)} cached)", flush=True)
with cf.ThreadPoolExecutor(16) as ex, open(CACHE_F, "a") as cf_:
    for k, v in zip(todo, ex.map(lambda k: ask(cl, QA.format(q=k[0], gold=k[1], alias="", pred=k[2]), stats), todo)):
        CACHE[k] = v
        if v is not None:                       # an unparsed / errored verdict is retried next run, never cached
            cf_.write(json.dumps({"q": k[0], "gold": k[1], "pred": k[2], "v": v}, ensure_ascii=False) + "\n")
print(f"luna tokens in {stats['in']} out {stats['out']} | unparsed {stats['unparsed']} errors {stats['errors']}", flush=True)

for run in RUNS:
    rows = []
    for f in sorted(os.listdir(run), key=lambda f: int(re.sub(r"\D", "", f) or 0)):
        if not re.fullmatch(r"preds_\d+\.jsonl", f):
            continue
        R = {}
        for p in map(json.loads, open(os.path.join(run, f))):
            c = CV[(p["dataset"], p["row"])]
            if p["notes"] not in NOTES or (c["allin"] != 1.0 and not ALLQ):
                continue
            r = R.setdefault((p["notes"], p["dataset"]), {"n": 0, "ok": 0, "unparsed": 0})
            r["n"] += 1
            if not p["pred"].startswith("[search]"):
                v = CACHE.get((c["question"], c["answer"], p["pred"]))
                r["ok"] += bool(v); r["unparsed"] += v is None
        for (src, ds), r in sorted(R.items()):
            rows.append({"step": int(re.sub(r"\D", "", f)), "notes": src, "dataset": ds, "n": r["n"],
                         "judged": r["ok"] / r["n"], "unparsed": r["unparsed"]})
            print(f"{os.path.basename(run)} step {rows[-1]['step']:>5} | notes={src:7s} | {ds:8s} | luna-correct {rows[-1]['judged']:.3f} "
                  f"({r['ok']}/{r['n']}, unparsed {r['unparsed']})")
    with open(os.path.join(run, "judged_all.jsonl" if ALLQ else "judged.jsonl"), "w") as f_:
        f_.writelines(json.dumps(x) + "\n" for x in rows)
