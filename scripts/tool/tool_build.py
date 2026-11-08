#!/usr/bin/env python3
"""Training rows for MiniCPM3-4B's NATIVE tool calls (2026-11-08, user: "go ahead"; prefers one rewritten question per missing hop).
Each row is a chat: system (+ the search tool in MiniCPM3's own JSON) -> user (chopper notes + question) -> assistant turns, each either a native call
search(query="<self-contained hop question>") followed by a tool turn with notes for that hop, or the final answer.
MuSiQue hop questions come from its published decomposition ("When was #1 founded?" with #1 replaced by hop 1's answer; triple-style hop 1
"X >> rel" becomes "What is the rel of X?"). Kinds: (a) every hop already in the notes -> answer; (b) notes cut at hop k (fragments of the
hop >= k passages removed) -> one call per missing hop, tool turn = the continued chopper's fragments of that passage (else the gold paragraph,
700 chars) -> answer; (c) a hop really missing from the pool -> calls from there, tool turns from the gold paragraphs. HotpotQA / TriviaQA rows:
(a) only, so answering directly stays the default when nothing is missing.
usage: tool_build.py <mm dir> <rr dir> <out jsonl>"""
import json, random, re, sys
MM, RR, OUT = sys.argv[1:4]; R = random.Random(5)
TOOLS = [{"type": "function", "function": {"name": "search", "description": "Search the document collection and return passages relevant to a query.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "a self-contained question for one missing fact"}}, "required": ["query"]}}}]
SYS = "Answer the question from the notes. If a fact you need is not in the notes, call search with one self-contained question for that fact."
TRIP = json.load(open(f"{RR}/tool_triples.json"))     # tool_triples.py: Qwen3-14B rewrites of "X >> rel"
RAW = {}
for f in ("train", "dev"):
    for l in open(f"/media/wii/bigdata/musique_raw/musique_ans_v1.0_{f}.jsonl"):
        r = json.loads(l); RAW[r["question"].strip()] = r


def hop_questions(r):
    out, ans = [], []
    for i, h in enumerate(r["question_decomposition"]):
        q = h["question"]
        if ">>" in q:
            w = TRIP.get(q)
            if not w or any(f"#{j}" in q and f"#{j}" not in w for j in range(1, 6)):     # a rewrite that lost "#1" would ask about nothing
                w = "What is the {1} of {0}?".format(*[x.strip() for x in q.split(">>", 1)])
            q = w
        for j, a in enumerate(ans):
            q = q.replace(f"#{j + 1}", a)
        out.append(q); ans.append(h["answer"])
    return out, ans


def call(q, thought):
    return {"role": "assistant", "thought": thought, "content": "", "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": q}}}]}


notes_of = lambda frs: "\n".join(f"- {' '.join(f.split())}" for f in frs) if frs else "(none)"
rows = []; stat = {"a": 0, "b": 0, "c": 0, "other_a": 0}
for sp in ("train", "train2", "train3"):
    P = {(x["dataset"], x["row"]): x for x in map(json.loads, open(f"{RR}/chopfc_{sp}/preds_val.jsonl"))}
    for c in map(json.loads, open(f"{MM}/ctx_{sp}.jsonl")):
        k = (c["dataset"], c["row"]); pr = P.get(k)
        if not pr:
            continue
        frs = [(f[0], f[3]) for f in pr["fragments"]]
        user = lambda fl: {"role": "user", "content": f"Notes:\n{notes_of([t for _, t in fl])}\n\nQuestion: {c['question']}"}
        final = {"role": "assistant", "thought": "The notes now contain every fact the question needs.", "content": c["answer"]}
        if c["dataset"] != "musique":
            if c["allin"] == 1.0 and R.random() < .5:
                rows.append({"key": k, "messages": [user(frs), final]}); stat["other_a"] += 1
            continue
        raw = RAW.get(c["question"].strip())
        if not raw:
            continue
        hq, ha = hop_questions(raw); H = len(hq)
        gold_par = [raw["paragraphs"][h["paragraph_support_idx"]] for h in raw["question_decomposition"]]
        hop_pos = [{i for i, p in enumerate(c["passages"]) if p["group"] == g} for g in range(H)]
        tool_text = lambda g: notes_of([t for i, t in frs if i in hop_pos[g]] or [(gold_par[g]["title"] + ". " + gold_par[g]["paragraph_text"])[:700]])
        if c["allin"] == 1.0:
            if R.random() < .35:
                rows.append({"key": k, "messages": [user(frs), final]}); stat["a"] += 1; continue
            cut = R.randint(1, H - 1); keep = [(i, t) for i, t in frs if not any(i in hop_pos[g] for g in range(cut, H))]; kind = "b"
        else:
            have = [any(i in hop_pos[g] for i, _ in frs) for g in range(H)]
            if all(have):
                continue
            cut = have.index(False); keep = [(i, t) for i, t in frs if not any(i in hop_pos[g] for g in range(cut, H))]; kind = "c"
        msgs = [user(keep)]
        for g in range(cut, H):
            msgs.append(call(hq[g], f"The notes do not say: {hq[g]}"))
            msgs.append({"role": "tool", "content": tool_text(g)})
        msgs.append(final); rows.append({"key": k, "messages": msgs, "hops": H, "cut": cut}); stat[kind] += 1
R.shuffle(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps({"key": list(r["key"]), "system": SYS, "tools": TOOLS, "messages": r["messages"]}, ensure_ascii=False) + "\n")
print(len(rows), stat)
