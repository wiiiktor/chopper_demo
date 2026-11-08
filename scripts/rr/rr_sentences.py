#!/usr/bin/env python3
"""Step 4c of the rerank dataset pilot: Qwen3-4B picks the material sentences of every passage luna accepted
(rr_luna_label.py). The yes/no decision is luna's; Qwen only chooses WHICH whole sentences carry the answer (user
2026-11-05: answer material, 1-3 whole sentences, never fragments; the metadata tag is kept whole at training time).
Rows: an answer-type row needs its seed accepted with >= 1 sentence; a no-answer row needs no accepted passage.
Writes <dir>/lab_<V>_<lang>.jsonl = rows + per passage {sents, luna (True/False/None = unread), keep, how}.
usage: rr_sentences.py <dir> [lang ...]   (env V=v2 MODEL=Qwen/Qwen3-4B SMOKE=N)"""
import json, os, re, sys, time
from collections import Counter
from vllm import LLM, SamplingParams

D = sys.argv[1]; LANGS = sys.argv[2:] or ["pl", "en", "zh", "es", "de", "fr", "ja", "pt"]
V = os.environ.get("V", "v2"); SMOKE = int(os.environ.get("SMOKE", "0"))
FINAL = V == "v2" and not SMOKE
while FINAL and not os.path.exists(f"{D}/luna_final_ok"):       # recheck + generic-row read-all (rrextra_chain.sh) come first
    time.sleep(30)


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

This passage answers the query. Which sentences state the answer? List at most 3, the most direct one FIRST. Usually 1 or 2
are enough. Add a neighbouring sentence only when the answer cannot be understood without it (it names what the answer
refers to). Do not list sentences that are merely on the topic. The query may be in another language than the passage.
Reply exactly:
SENTENCES: <comma-separated sentence numbers, best first>"""
llm = LLM(model=os.environ.get("MODEL", "Qwen/Qwen3-4B"), max_model_len=4096, gpu_memory_utilization=0.90, seed=0,
          max_num_seqs=128, enable_prefix_caching=True)
sp = SamplingParams(temperature=0, max_tokens=30)
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
    R = [json.loads(l) for l in open(f"{D}/rows_{V}_{lang}.jsonl")][:SMOKE or None]
    L = {}
    for fn in (f"luna_{V}_{lang}", f"luna_{V}_extra_{lang}", f"luna_{V}_recheck_{lang}", f"luna_{V}_g6_{lang}"):   # later files override: recheck saw the source, g6 = GPT-6 Luna
        if os.path.exists(f"{D}/{fn}.jsonl"):
            L |= {(x["rid"], x["i"]): x for x in map(json.loads, open(f"{D}/{fn}.jsonl"))}
    if FINAL:                                    # user 2026-11-05: common-knowledge (and vague) queries go
        C = {x["rid"]: x["c"] for x in map(json.loads, open(f"{D}/qclass_{V}_{lang}.jsonl"))}
        n0 = len(R); R = [r for r in R if C.get(r["rid"]) == "SOURCE"]; print(f"{lang}: query filter kept {len(R)}/{n0}", flush=True)
    jobs = []
    for ri, r in enumerate(R):
        for i, p in enumerate(r["passages"]):
            x = L.get((r["rid"], i))
            p.update(sents=sents(p["text"]), luna=x["v"] if x else None, how=x["how"] if x else "unread", keep=[])
            if p["luna"]:
                jobs.append((ri, i))
    t0 = time.time()
    outs = llm.chat([[{"role": "user", "content": PROMPT.format(q=R[ri]["query"], lines=lines(R[ri]["passages"][i]["sents"]))}]
                     for ri, i in jobs], sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    bad = 0
    for (ri, i), o in zip(jobs, outs):
        p = R[ri]["passages"][i]; m = re.search(r"SENTENCES:\s*(.+)", o.outputs[0].text)
        # Qwen3-4B ignored "at most 3" (smoke: 37 of 45 lists longer, mean 6.1), so it ranks and the cap is applied here
        k = [int(x) - 1 for x in re.findall(r"\d+", m.group(1)) if 0 < int(x) <= len(p["sents"])] if m else []
        k = sorted(set(list(dict.fromkeys(k))[:3]))
        p["keep"] = k; bad += not k
    keep, drop = [], Counter()
    for r in R:
        sd = r["passages"][r["seed"]]
        if any(p["how"] != "unread" and p["luna"] is None for p in r["passages"]):
            drop["luna_missing"] += 1
        elif r["qtype"] == "no_answer":
            if any(p["luna"] for p in r["passages"]):
                drop["no_answer_has_answer"] += 1
            else:
                keep.append(r)
        elif not sd["luna"]:
            drop["seed_rejected_by_luna"] += 1
        elif not sd["keep"]:
            drop["seed_no_sentences"] += 1
        else:
            for p in r["passages"]:                  # an accepted passage with no parsable sentences cannot be a target
                if p["luna"] and not p["keep"]:
                    p["luna"] = None; p["how"] += "-nosent"
            keep.append(r)
    dst = f"{D}/lab_{V}_{lang}{'_smoke' if SMOKE else ''}.jsonl"
    with open(dst + ".tmp", "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in keep)
    os.rename(dst + ".tmp", dst)
    P = [p for r in keep for p in r["passages"] if p["keep"]]
    aud = [p for r in keep for p in r["passages"] if p["how"] == "audit"]
    print(f"{lang}: {len(jobs)} accepted passages in {time.time() - t0:.0f}s | rows kept {len(keep)}/{len(R)} dropped {dict(drop)} "
          f"| positives/row {len(P) / max(1, len(keep)):.2f} | sentences/positive {sum(len(p['keep']) for p in P) / max(1, len(P)):.2f} "
          f"| >3 sentences {sum(len(p['keep']) > 3 for p in P)} | no sentences {bad} | audit YES {sum(bool(p['luna']) for p in aud)}/{len(aud)}", flush=True)
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
