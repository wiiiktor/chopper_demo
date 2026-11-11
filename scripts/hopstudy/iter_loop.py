#!/usr/bin/env python3
"""Iterative note-taking on the hop study (user 2026-11-11: "build this iterative test for chopper and opus"). Qwen3-14B (zero-shot) reads
the notes and either answers or writes ONE follow-up question; the note-taker then cuts the SAME 20 paragraphs again with that follow-up
question; the new notes are appended; up to 3 follow-ups, then Qwen must answer. Note-takers:
  chopper  the mmBERT chopper (mm_train.py --eval-only on a temporary split whose questions are the follow-ups)
  opus     Claude Opus 5.5 via OpenRouter, answer-blind (teacher_api.py --blind), round 0 = its existing answer-blind notes
gpt-5.6-luna judges the final answers. Writes <hs>/iter_<arm>.jsonl and prints accuracy per hop count, rounds used, notes tokens.
usage: iter_loop.py <hs dir> <chopper|opus> [chopper.pt]   (env GPU_UTIL=0.55 ROUNDS=3)"""
import concurrent.futures as cf, json, os, re, subprocess, sys
HD, ARM = sys.argv[1], sys.argv[2]; CK = sys.argv[3] if len(sys.argv) > 3 else ""
ROOT = "/home/wii/Projects/certain"; ROUNDS = int(os.environ.get("ROUNDS", "3")); WD = f"{HD}/iter_{ARM}"; os.makedirs(WD, exist_ok=True)
R = {r["row"]: r for r in json.load(open(f"{HD}/rows.json"))}
C = {json.loads(l)["row"]: json.loads(l) for l in open(f"{HD}/ctx_hs.jsonl")}
Q0 = {json.loads(l)["row"]: json.loads(l) for l in open(f"{HD}/qw_hs.jsonl")}


def pieces_from_fragments(x):
    return [" ".join(f[3].replace("<eos>", " ").split()) for f in x.get("fragments", []) if f[3].strip()]


def pieces_from_spans(c, x):
    return [" ".join(c["passages"][k]["text"][a:b].split()) for k, a, b in x.get("spans", [])]


def cut(round_no, qs):
    """qs = {row: question} -> {row: [note pieces]} from the note-taker, cutting the same 20 paragraphs with that question"""
    if not qs:
        return {}
    tag = f"it{round_no}"; d = f"{WD}/{tag}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/ctx_{tag}.jsonl", "w") as f, open(f"{d}/qw_{tag}.jsonl", "w") as g:
        for n, q in qs.items():
            f.write(json.dumps(dict(C[n], question=q, split=tag), ensure_ascii=False) + "\n")
            g.write(json.dumps(dict(Q0[n], split=tag)) + "\n")
    if ARM == "chopper":
        subprocess.run([f"{ROOT}/.venv-gen/bin/python", f"{ROOT}/scripts/mm_train.py", d, f"{d}/out", "--base", "jhu-clsp/mmBERT-base", "--train-splits", tag,
                        "--val-split", tag, "--eval-only", CK, "--pred-thr", "0.5", "--merge-gap", "5", "--min-run", "4"],
                       check=True, stdout=open(f"{d}/cut.out", "w"), stderr=subprocess.STDOUT, env=dict(os.environ, HF_HUB_OFFLINE="1"))
        P = {x["row"]: x for x in map(json.loads, open(f"{d}/out/preds_val.jsonl"))}
        return {n: pieces_from_fragments(P.get(n, {})) for n in qs}
    subprocess.run([f"{ROOT}/.venv-gen/bin/python", f"{ROOT}/scripts/hopstudy/teacher_api.py", f"{d}/ctx_{tag}.jsonl", f"{d}/opus.jsonl", "--blind", "--threads", "6"],
                   check=True, stdout=open(f"{d}/cut.out", "w"), stderr=subprocess.STDOUT)
    P = {x["row"]: x for x in map(json.loads, open(f"{d}/opus.jsonl"))}
    return {n: pieces_from_spans(C[n], P.get(n, {})) for n in qs}


# round 0 notes
if ARM == "chopper":
    NOTES = cut(0, {n: r["q"] for n, r in R.items()})
else:
    P = {x["row"]: x for x in map(json.loads, open(f"{HD}/teacher_opusblind.jsonl"))}
    NOTES = {n: pieces_from_spans(C[n], P.get(n, {})) for n in R}
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-14B-AWQ", max_model_len=8192, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.55")), seed=0)
sp = SamplingParams(temperature=0, max_tokens=80)
ASK = ("Use ONLY the notes. If they contain everything needed, reply exactly: ANSWER: <short answer>.\n"
       "Otherwise find the FIRST missing link in the chain of facts and ask about that ONE fact only, as a short simple question that names the "
       "entity the notes already give you. Do not repeat the original question or an earlier search. Reply exactly: SEARCH: <short question>.\n"
       "Example: question 'Who is the spouse of the performer of Green?', notes say 'Green is an album by Steve Hillage' -> "
       "SEARCH: Who is Steve Hillage's spouse?\n\nNotes:\n{notes}\n\nEarlier searches: {prev}\n\nQuestion: {q}")
FORCE = ("Answer the question using ONLY the notes. Reply with just the answer (a name, date, number or short phrase). If the notes do not "
         "contain enough, give your best guess.\n\nNotes:\n{notes}\n\nQuestion: {q}")
fmt = lambda n: "\n".join(f"- {p}" for p in dict.fromkeys(NOTES[n])) or "(none)"
FINAL, ROUND, FOLLOW, STUCK = {}, {n: 0 for n in R}, {n: [] for n in R}, []
open_ = list(R)
for rnd in range(1, ROUNDS + 1):
    outs = llm.chat([[{"role": "user", "content": ASK.format(notes=fmt(n), q=R[n]["q"], prev="; ".join(FOLLOW[n]) or "none")}] for n in open_], sp, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
    ask = {}
    for n, o in zip(open_, outs):
        t = o.outputs[0].text.strip(); m = re.search(r"SEARCH:\s*(.+)", t)
        if m and len(m.group(1).split()) >= 3 and m.group(1).strip().lower() not in [x.lower() for x in FOLLOW[n]] + [R[n]["q"].lower()]:
            ask[n] = m.group(1).strip(); FOLLOW[n].append(ask[n]); ROUND[n] = rnd
        elif m:                                                    # a repeated / too-short search: stop searching, force an answer later
            STUCK.append(n)
        else:
            FINAL[n] = re.sub(r"^ANSWER:\s*", "", t).strip()
    print(f"round {rnd}: {len(ask)} follow-up questions, {len(open_) - len(ask)} answered or stopped", flush=True)
    for n, ps in cut(rnd, ask).items():
        NOTES[n] += ps
    open_ = list(ask)
    if not open_:
        break
open_ = open_ + STUCK
if open_:
    outs = llm.chat([[{"role": "user", "content": FORCE.format(notes=fmt(n), q=R[n]["q"])}] for n in open_], sp, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
    for n, o in zip(open_, outs):
        FINAL[n] = o.outputs[0].text.strip()
tok = llm.get_tokenizer()
sys.path.insert(0, f"{ROOT}/scripts")
from judge_mh_luna import QA, ask as judge
from openai import OpenAI
oai = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip()); st = {"in": 0, "out": 0, "unparsed": 0, "errors": 0}


def j(n):
    r = R[n]; al = (" (also: " + "; ".join(r["aliases"]) + ")") if r["aliases"] else ""
    return bool(judge(oai, QA.format(q=r["q"], gold=r["a"], alias=al, pred=FINAL.get(n) or "(no answer)"), st))


with cf.ThreadPoolExecutor(16) as ex:
    OK = dict(zip(R, ex.map(j, R)))
with open(f"{HD}/iter_{ARM}.jsonl", "w") as f:
    for n, r in R.items():
        f.write(json.dumps({"row": n, "hops": r["hops"], "ok": OK[n], "pred": FINAL.get(n, ""), "follow": FOLLOW[n], "rounds": ROUND[n],
                            "notes_tok": len(tok(fmt(n))["input_ids"])}, ensure_ascii=False) + "\n")
for k in (1, 2, 3, 4):
    xs = [n for n, r in R.items() if r["hops"] == k]; m_ = max(1, len(xs))
    print(f"{ARM} {k}-hop: correct {sum(OK[n] for n in xs)}/{len(xs)} | follow-ups/q {sum(ROUND[n] for n in xs) / m_:.2f} | "
          f"notes tokens {sum(len(tok(fmt(n))['input_ids']) for n in xs) / m_:.0f}", flush=True)
sys.stdout.flush(); os._exit(0)
