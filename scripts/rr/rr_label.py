#!/usr/bin/env python3
"""Step 4b of the rerank dataset pilot: Qwen3-14B-AWQ labels which WHOLE SENTENCES of a passage are material for the
query (user 2026-11-05: the answer-bearing sentence(s) plus what they depend on, 1-3 sentences, never fragments) and
grades the passage 0-3. Read: the seed, every passage with reranker p > TAU (rr_rerank.py), and a 5% random audit of the
rest (how="audit": its positives = what the prefilter would have missed). Unread passages get grade 0, keep [].
Rows are dropped when an answer-type row's seed has no material, or a no-answer row has material anywhere.
The metadata tag is not labelled: a selected passage keeps its whole tag, rendered at training time.
Writes <dir>/lab_<lang>.jsonl = rows_<lang> + per passage {sents, grade, keep, how}.
usage: rr_label.py <dir> [lang ...] [SMOKE=N TAU=0.3 env]"""
import json, os, random, re, sys, time
from collections import Counter
from vllm import LLM, SamplingParams

D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]
SMOKE = int(os.environ.get("SMOKE", "0")); TAU = float(os.environ.get("TAU", "0.3"))
rng = random.Random(0)


def sents(t):
    out = []
    for s in re.split(r"(?<=[.!?])\s+|(?<=[。！？])|\n+", t):
        s = s.strip()
        if not s:
            continue
        if out and len(s) < 4:                       # stray "." / list markers join the sentence before
            out[-1] += " " + s
        else:
            out.append(s)
    return out


PROMPT = """Query: {q}

Passage, one sentence per line:
{lines}

Which sentences state the answer? Pick the FEWEST whole sentences that do: usually 1-2, at most 3. Add a neighbouring
sentence only when the answer cannot be understood without it (it names what the answer refers to). Do not pick
sentences that are merely on the same topic. The query may be in another language than the passage.
Grade the passage:
3 = it states the full answer to this query (for the same organisation, product, place or case, if the query names or
    implies one; a similar thing elsewhere is not an answer),
2 = it states the answer to one part of a multi-part query,
1 = same topic, but the answer to this query is not in it,
0 = unrelated.
Reply exactly:
GRADE: <0-3>
SENTENCES: <comma-separated sentence numbers, or NONE when the grade is 0 or 1>"""
GROUND = """Query: {q}

Reference: {ref}

Passage, one sentence per line:
{lines}

Does this passage itself state the answer to the query? It counts when it states the same facts as the reference, or,
only if the query names no specific organisation, product, place or case, a different but equally direct answer of the
same kind. Being on the same topic, mentioning the same words, or answering a related question does NOT count.
If it counts, pick the FEWEST whole sentences that state the answer (usually 1-2, at most 3).
Grade: 3 = states the full answer, 2 = states the answer to one part of a multi-part query, 1 = same topic but no
answer, 0 = unrelated.
Reply exactly:
GRADE: <0-3>
SENTENCES: <comma-separated sentence numbers, or NONE when the grade is 0 or 1>"""
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-14B-AWQ")
llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.90, seed=0, max_num_seqs=128, enable_prefix_caching=True)
sp = SamplingParams(temperature=0, max_tokens=40)
TOK = llm.get_tokenizer()


def lines(ss):                                   # numbered sentences, cut at ~3,400 tokens (max_model_len 4096)
    out, n = [], 0
    for k, s in enumerate(ss):
        n += len(TOK(s, add_special_tokens=False)["input_ids"]) + 4
        if n > 3400:
            break
        out.append(f"[{k + 1}] {s}")
    return "\n".join(out)
for lang in LANGS:
    dst = f"{D}/lab_{lang}{'_smoke' if SMOKE else ''}{os.environ.get('TAG', '')}.jsonl"
    R = [json.loads(l) for l in open(f"{D}/rows_{lang}.jsonl")][:SMOKE or None]
    RR = json.load(open(f"{D}/rr_{lang}{'_smoke' if SMOKE else ''}.json"))
    jobs = []
    for ri, r in enumerate(R):
        for pi, p in enumerate(r["passages"]):
            p["sents"] = sents(p["text"]); p.update(grade=0, keep=[], how="unread")
            s = RR[r["rid"]][pi]
            how = "seed" if pi == r["seed"] else "rr" if s > TAU else "audit" if rng.random() < 0.05 else None
            if how:
                p["how"] = how
                jobs.append((ri, pi))
    bad = 0

    def run(js, msg):                            # v1/v2 graded ~90% of everything 2-3 (audit: 24/42 positives at p<0.3,
        global bad                               # mostly wrong by hand) -> others are judged against the seed's answer
        outs = llm.chat([[{"role": "user", "content": msg(ri, pi)}] for ri, pi in js], sp, use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": False})
        for (ri, pi), o in zip(js, outs):
            p = R[ri]["passages"][pi]; t = o.outputs[0].text
            g = re.search(r"GRADE:\s*([0-3])", t); s = re.search(r"SENTENCES:\s*(.+)", t)
            if not g or not s:
                p["how"] += "-unparsed"; bad += 1; continue
            k = sorted({int(x) - 1 for x in re.findall(r"\d+", s.group(1)) if 0 < int(x) <= len(p["sents"])})
            p["grade"] = int(g.group(1)); p["keep"] = k if p["grade"] >= 2 else []

    t0 = time.time()
    run([j for j in jobs if R[j[0]]["passages"][j[1]]["how"] == "seed"],
        lambda ri, pi: PROMPT.format(q=R[ri]["query"], lines=lines(R[ri]["passages"][pi]["sents"])))

    def ref(r):
        sd = r["passages"][r["seed"]]
        if r["qtype"] == "no_answer" or not sd["keep"]:
            return "(none: the passage this query was written from does NOT contain the answer)"
        return "the answer, as stated in another passage: \"" + " ".join(sd["sents"][k] for k in sd["keep"]) + "\""
    run([j for j in jobs if R[j[0]]["passages"][j[1]]["how"] != "seed"],
        lambda ri, pi: GROUND.format(q=R[ri]["query"], ref=ref(R[ri]), lines=lines(R[ri]["passages"][pi]["sents"])))
    keep, drop = [], Counter()
    for r in R:
        pos = [p for p in r["passages"] if p["keep"]]
        if r["qtype"] == "no_answer" and pos:
            drop["no_answer_has_material"] += 1
        elif r["qtype"] != "no_answer" and not r["passages"][r["seed"]]["keep"]:
            drop["seed_no_material"] += 1
        elif any(p["how"].endswith("unparsed") for p in r["passages"]):
            drop["unparsed"] += 1
        else:
            keep.append(r)
    with open(dst + ".tmp", "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in keep)
    os.rename(dst + ".tmp", dst)
    P = [p for r in keep for p in r["passages"]]
    aud = [p for p in P if p["how"] == "audit"]
    print(f"{lang}: {len(jobs)} reads in {time.time() - t0:.0f}s ({len(jobs) / (time.time() - t0):.1f}/s) | rows kept {len(keep)}/{len(R)} "
          f"dropped {dict(drop)} | positives/row {sum(bool(p['keep']) for p in P) / max(1, len(keep)):.2f} "
          f"| sentences/positive {sum(len(p['keep']) for p in P) / max(1, sum(bool(p['keep']) for p in P)):.2f} "
          f"| audit positives {sum(bool(p['keep']) for p in aud)}/{len(aud)} | unparsed {bad}", flush=True)
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
