#!/usr/bin/env python3
"""Conversation-memory rows, stage 1 (user 2026-11-09: "do we have now some new dataset to train mmbert?" -> conversation memory).
A history = one WildChat user's real conversations (grouped by hashed_ip, English, non-toxic, ordered by time), one conversation =
one dated session (real timestamps). Qwen writes first-person facts about the user as JSON; CODE places them into user turns of
chosen sessions and computes every answer that depends on dates, so the gold answer and the evidence turn are known by construction
(no judge needed). Kinds, after LongMemEval / LoCoMo:
  fact    one plain fact                              -> its value
  update  an old value, a later new value             -> the current value (and a past-value question)
  pref    a preference said in passing                 -> a question that needs it
  event   an event dated relative to its session       -> the ABSOLUTE date, computed here ("yesterday" in a 5 May session = 4 May)
  multi   2-3 items in different sessions               -> all items
  absent  a question about something never said        -> "not mentioned"
Writes <out>/hist_<shard>.jsonl: {hid, sessions: [{date, turns: [{role, text, plant?}]}], qs: [{kind, q, a, ev: [[s, t]], qdate}]}.
usage: cm_gen.py <wildchat parquet> <out dir> [--n HISTORIES] [--min-sess 5] [--max-sess 12]   (env MODEL=Qwen/Qwen3-14B-AWQ, SMOKE=1)"""
import datetime as dt, json, os, random, re, sys, time
from collections import defaultdict
import pyarrow.parquet as pq
PQ, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
N, MINS, MAXS = arg("--n", 100000), arg("--min-sess", 5), arg("--max-sess", 12)
PQS = sorted(PQ.split(","))                                    # one or more parquet shards, comma-separated
SHARD = "_".join(re.search(r"(\d{5})-of", p).group(1) for p in PQS) if len(PQS) < 3 else f"{len(PQS)}shards"; rng = random.Random(0)
WORDS = 150                                                    # assistant turns cut to ~150 words, user turns to 120: history stays readable

cut = lambda t, n: " ".join(t.split()[:n]) + (" …" if len(t.split()) > n else "")
U = defaultdict(list)
for p_ in PQS:
    for x in pq.read_table(p_, columns=["conversation", "language", "toxic", "redacted", "timestamp", "hashed_ip"]).to_pylist():
        if x["language"] != "English" or x["toxic"] or x["redacted"] or len(x["conversation"]) < 2 or x["timestamp"] is None:
            continue
        U[x["hashed_ip"]].append({"date": x["timestamp"], "turns": [{"role": t["role"], "text": cut(t["content"], WORDS if t["role"] == "assistant" else 120)}
                                                                     for t in x["conversation"][:8]]})
H = []
for ip, ss in U.items():
    ss.sort(key=lambda s: s["date"]); day = {}
    for x in ss:                                               # one session per calendar day: questions about dates need sessions on different days
        day.setdefault(x["date"].date(), x)
    ss = list(day.values())
    if len(ss) >= MINS:
        ss = ss[-MAXS:]
        H.append({"hid": f"{SHARD}_{len(H)}", "sessions": [{"date": s["date"].strftime("%Y/%m/%d (%a) %H:%M"), "dt": s["date"], "turns": s["turns"]} for s in ss]})
rng.shuffle(H); H = H[:N]
print(f"shard {SHARD}: {len(U)} users, {len(H)} histories with >= {MINS} sessions", flush=True)

PROMPT = """Here are short excerpts of what one person asked a chatbot over several weeks:
{topics}

Invent personal details this person could mention in passing, consistent with their interests. Write in the first person, casual,
one sentence each, the way someone adds a personal detail to a request. Use specific names, places, numbers and items.
Return ONLY a JSON object with exactly these keys:
{{"fact": [{{"say": "...", "q": "question asked later, in the first person (my/I)", "a": "short answer, copied exactly from the sentence"}}, ... 2 items],
 "update": [{{"say_old": "...", "say_new": "... a later change of the same detail ...", "q": "question about the CURRENT value", "a": "new value", "q_old": "question about the value BEFORE the change", "a_old": "old value"}}],
 "pref": [{{"say": "... a like or dislike ...", "q": "a request whose best answer depends on it, without naming the preference", "a": "what a good answer must respect, and why"}}],
 "event": [{{"say": "... something I DID in the past, where the placeholder {{WHEN}} stands for a phrase like yesterday or last Monday, e.g. I finally fixed my bike {{WHEN}}.", "q": "When did I ...?"}}, ... 2 items],
 "multi": [{{"says": ["...", "...", "..."], "q": "question whose answer lists all three", "a": "item1, item2, item3"}}],
 "absent": [{{"q": "a personal question about something NOT mentioned in any of the above"}}, ... 2 items]}}"""
WHEN = [("yesterday", 1), ("two days ago", 2), ("three days ago", 3), ("last weekend", None), ("last {wd}", None)]
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def when(d0):
    """relative phrase + its absolute date, from the session date d0"""
    k = rng.random()
    if k < 0.5:
        ph, n = WHEN[rng.randrange(3)]; return ph, d0 - dt.timedelta(days=n)
    w = rng.randrange(7); back = (d0.weekday() - w) % 7 or 7                 # "last Tuesday" = the most recent Tuesday before the session day
    return f"last {WD[w]}", d0 - dt.timedelta(days=back)


fmt = lambda d: f"{d.day} {d.strftime('%B %Y')}"


def plant(h, s, text):
    """put a sentence at the start of a random user turn of session s; return [s, turn index]"""
    us = [i for i, t in enumerate(h["sessions"][s]["turns"]) if t["role"] == "user"]
    free = [i for i in us if "plant" not in h["sessions"][s]["turns"][i]]       # one planted sentence per turn (2026-11-10: never stack)
    if not free:
        raise ValueError("no free user turn")                   # the fact is dropped (moving it would break its date / order)
    i = rng.choice(free); t = h["sessions"][s]["turns"][i]
    text = re.sub(r"^(by the way|btw|oh,? and|also|fyi)[,:]?\s+", "", text.strip(), flags=re.I)   # no fixed opener: it would be a shortcut
    text = text[:1].upper() + text[1:]
    ss_ = re.split(r"(?<=[.!?])\s+", t["text"]); k = rng.random()      # start, end or after the first sentence, not always first
    pos = 0 if k < 0.4 or len(ss_) < 2 else len(ss_) if k < 0.8 else 1
    t["text"] = " ".join(ss_[:pos] + [text] + ss_[pos:]); t.setdefault("plant", []).append(text)
    return [s, i]


def assemble(h, J):
    try:
        return _assemble(h, J)
    except ValueError:
        return []


def _assemble(h, J):
    S = len(h["sessions"]); qs = []; last = h["sessions"][-1]["dt"]
    qdate = (last + dt.timedelta(days=rng.randint(1, 10))).strftime("%Y/%m/%d (%a) %H:%M")
    ok = lambda x, *k: isinstance(x, dict) and all(isinstance(x.get(j), str) and x[j].strip() for j in k)
    nrm = lambda t: re.sub(r"[^a-z0-9 ]", " ", t.lower())
    has = lambda say, ans: all(w in nrm(say).split() for w in nrm(ans).split()) and nrm(ans).strip() != ""   # every answer word is in the planted sentence
    for f in J.get("fact", [])[:2]:
        if ok(f, "say", "q", "a") and has(f["say"], f["a"]) and len(f["a"].split()) <= 8:
            qs.append({"kind": "fact", "q": f["q"], "a": f["a"], "ev": [plant(h, rng.randrange(S), f["say"])]})
    for f in J.get("update", [])[:1]:
        if ok(f, "say_old", "say_new", "q", "a", "q_old", "a_old") and S >= 2 and has(f["say_new"], f["a"]) and has(f["say_old"], f["a_old"]):
            s0 = rng.randrange(S - 1); s1 = rng.randrange(s0 + 1, S)
            e0, e1 = plant(h, s0, f["say_old"]), plant(h, s1, f["say_new"])
            qs.append({"kind": "update", "q": f["q"], "a": f["a"], "ev": [e1, e0]})
            qs.append({"kind": "update-past", "q": f["q_old"], "a": f["a_old"], "ev": [e0, e1]})
    for f in J.get("pref", [])[:1]:
        if ok(f, "say", "q", "a"):
            qs.append({"kind": "pref", "q": f["q"], "a": f["a"], "ev": [plant(h, rng.randrange(S), f["say"])]})
    for f in J.get("event", [])[:2]:
        if ok(f, "say", "q") and "{WHEN}" in f["say"]:
            s = rng.randrange(S); ph, d = when(h["sessions"][s]["dt"].date())
            qs.append({"kind": "event", "q": f["q"], "a": fmt(d), "rel": ph, "ev": [plant(h, s, f["say"].replace("{WHEN}", ph))]})
    for f in J.get("multi", [])[:1]:
        if isinstance(f, dict) and isinstance(f.get("says"), list) and len(f["says"]) >= 2 and ok(f, "q", "a") and S >= len(f["says"]) and all(isinstance(t, str) for t in f["says"]) \
                and all(any(has(t, it) for t in f["says"]) for it in f["a"].split(",")):
            ss = rng.sample(range(S), len(f["says"][:3]))
            qs.append({"kind": "multi", "q": f["q"], "a": f["a"], "ev": [plant(h, s, t) for s, t in zip(ss, f["says"][:3]) if isinstance(t, str)]})
    for f in J.get("absent", [])[:2]:
        if ok(f, "q"):
            qs.append({"kind": "absent", "q": f["q"], "a": "Not mentioned in the conversation: the correct reply says this was never mentioned.", "ev": []})
    for q in qs:
        q["qdate"] = qdate
    return qs


msgs = []
for h in H:
    tp = [t["text"][:160] for s in h["sessions"] for t in s["turns"] if t["role"] == "user"]
    msgs.append([{"role": "user", "content": PROMPT.format(topics="\n".join(f"- {x}" for x in rng.sample(tp, min(8, len(tp)))))}])
t0 = time.time()
if os.environ.get("BACKEND") == "luna":                          # smoke only: check the prompt and the assembly with a strong model while the GPU is busy
    import concurrent.futures as cf
    from openai import OpenAI
    cl = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
    one = lambda m: cl.chat.completions.create(model="gpt-5.6-luna", max_completion_tokens=3000, messages=m).choices[0].message.content or ""
    with cf.ThreadPoolExecutor(8) as ex:
        TXT = list(ex.map(one, msgs))
else:
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.environ.get("MODEL", "Qwen/Qwen3-14B-AWQ"), max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
              seed=0, max_num_seqs=64, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1100, seed=0)
    TXT = [o.outputs[0].text for o in llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})]
n_ok = n_q = 0; dst = f"{OUT}/hist_{SHARD}{'_smoke' if os.environ.get('SMOKE') else ''}.jsonl"
with open(dst, "w") as f:
    for h, o in zip(H, TXT):
        m = re.search(r"\{.*\}", o, re.S)
        try:
            J = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            J = None
        if not isinstance(J, dict):
            continue
        qs = assemble(h, J)
        if len(qs) < 5:
            continue
        for s in h["sessions"]:
            s.pop("dt")
        h["qs"] = qs; n_ok += 1; n_q += len(qs)
        f.write(json.dumps(h, ensure_ascii=False) + "\n")
print(f"shard {SHARD}: {n_ok}/{len(H)} histories kept, {n_q} questions, {time.time() - t0:.0f}s", flush=True)
sys.stdout.flush(); os._exit(0)
