#!/usr/bin/env python3
"""Step 3 of the rerank dataset pilot: seeds, 16-passage pools and queries.
Seeds: 625 per language x area = every synthetic document of the cell (rr_synth.py) + verified web passages
(area_<lang>.jsonl: area agreed, quality 2) for the rest. Pools: the seed + 15 nearest passages of the SAME source kind
(web seed -> web distractors from the whole language pool, synthetic -> synthetic) by Qwen3-Embedding cosine: top 8 +
7 drawn from ranks 9-60, same-url and near-duplicates (cos > DUP, 0.9 since v2) removed, seed at a random slot. Same-kind pools keep
the source tags from telling the positive apart (taxonomy labels.tag_leak_audit). Queries: Qwen3-14B-AWQ writes one
query of the row's type from the seed alone; 10% cross-lingual (English query, or a random other language for EN).
Every passage keeps its metadata (url/domain/title guess for web; title/author/date/section/type for synthetic); tags
are rendered at training time, so tag on/off and field dropout stay free choices there.
Writes <dir>/rows_<lang>.jsonl {rid, lang, area, qtype, qlang, query, seed, passages:[{id, kind, text, meta}]}.
usage: rr_queries.py <dir> [N_SMOKE]"""
import json, os, random, re, sys, time
from collections import Counter
from urllib.parse import urlparse
import numpy as np

D = sys.argv[1]; SMOKE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.json")))
NEED = TX["pilot"]["rows_per_language_area"]
LANGS = {"en": "English", "zh": "Simplified Chinese", "es": "Spanish", "de": "German", "fr": "French", "ja": "Japanese",
         "pt": "Portuguese", "pl": "Polish"}
QT = list(TX["query_types"]); QW = [TX["query_types"][k] for k in QT]
rng = random.Random(0)
# v2 (2026-11-05): v1's generic queries over pools of near-identical boilerplate (Polish shop T&Cs, accessibility
# statements) had no single right label: every labeller said "answers" to ~90% of reads. v2 names the seed's own
# organisation / product / case in SPECIFIC of the queries and drops template copies (cos > 0.9) from the pools.
V = os.environ.get("V", "v2"); DUP = 0.97 if V == "v1" else 0.9; SPECIFIC = 0.0 if V == "v1" else 0.7


def web_meta(c):
    first = c["text"].split("\n", 1)[0].strip()
    title = first if len(first.split()) <= 12 and len(first) <= 90 and not re.search(r"[.!?。！？:]$", first) else None
    return {"url": c["url"], "domain": urlparse(c["url"]).netloc.removeprefix("www."), "title": title}


def pools(E, seeds, meta):
    """15 distractors per seed row index of E (same source kind), seed at a random slot."""
    Ef = E.astype(np.float32); out = []        # CPU: CUDA touched here would make vLLM spawn and re-run this module
    for i in range(0, len(seeds), 256):
        S = Ef[seeds[i:i + 256]] @ Ef.T
        k = min(80, len(E))
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), axis=1), 1)
        top = type("T", (), {"values": np.take_along_axis(S, order, 1), "indices": order})
        for s, vals, idx in zip(seeds[i:i + 256], top.values.tolist(), top.indices.tolist()):
            seen = {meta[s].get("url") or s}; cand = []
            for v, j in zip(vals, idx):
                key = meta[j].get("url") or j
                if j == s or v > DUP or key in seen:
                    continue
                seen.add(key); cand.append(j)
            pick = cand[:8] + rng.sample(cand[8:60], min(7, len(cand[8:60])))
            pick = pick + cand[len(pick):len(pick) + 15 - len(pick)] if len(pick) < 15 else pick
            pos = rng.randrange(len(pick) + 1)
            out.append(pick[:pos] + [s] + pick[pos:])
    return out


rows = []
for lang in LANGS:
    C = [json.loads(l) for l in open(f"{D}/chunks_{lang}.jsonl")]
    Y = [json.loads(l) for l in open(f"{D}/synth_{lang}.jsonl")]
    ok = {}
    for x in map(json.loads, open(f"{D}/area_{lang}.jsonl")):
        if x["area"] == x["cand_area"] and x["q"] == 2:
            ok.setdefault(x["area"], []).append(x["row"])
    wseed, yseed = [], []
    for a in range(1, 9):
        ys = [i for i, y in enumerate(Y) if y["area"] == a][:NEED]
        ws = rng.sample(ok.get(a, []), min(NEED - len(ys), len(ok.get(a, []))))
        yseed += [(a, i) for i in ys]; wseed += [(a, r) for r in ws]
    if SMOKE:
        wseed, yseed = rng.sample(wseed, SMOKE // 2), rng.sample(yseed, SMOKE // 2)
    wmeta = [{"url": c["url"]} for c in C]
    ymeta = [{} for _ in Y]
    for kind, seeds, E, meta in (("web", wseed, np.load(f"{D}/emb_{lang}.npy"), wmeta),
                                 ("syn", yseed, np.load(f"{D}/emb_syn_{lang}.npy"), ymeta)):
        for (a, s), P in zip(seeds, pools(E, [s for _, s in seeds], meta)):
            ps = []
            for j in P:
                if kind == "web":
                    ps.append({"id": C[j]["id"], "kind": "web", "text": C[j]["text"], "meta": web_meta(C[j])})
                else:
                    y = Y[j]; ps.append({"id": y["id"], "kind": "syn", "text": y["text"],
                                         "meta": {k: y[k] for k in ("title", "author", "date", "section", "type")}})
            qt = rng.choices(QT, QW)[0]
            ql = lang if rng.random() >= TX["cross_lingual_share"] else ("en" if lang != "en" else rng.choice([l for l in LANGS if l != "en"]))
            rows.append({"rid": f"{lang}-{kind}-{a}-{s}", "lang": lang, "area": a, "qtype": qt, "qlang": ql,
                         "seed": P.index(s), "passages": ps})
    print(f"{lang}: {len(wseed)} web + {len(yseed)} synthetic seeds", flush=True)

ASK = {"keywords": "a short keyword query of 2-5 words in total, as typed into a search box (one phrase, no commas, no full sentence)",
       "natural_question": "one natural question a user would ask whose answer is in the passage",
       "long_multi_condition": "one long query (25-50 words) in which the user describes their situation with 2-3 conditions or details, answered by the passage",
       "compound": "one question that asks two different things, both answered in the passage",
       "no_answer": "one question a user of the same site would plausibly ask about the same topic, whose answer is NOT in the passage, not even partly"}
PROMPT = """Passage ({src}):
{text}

Write {ask}. Write it in {qlang}.{spec} Use your own words: do not copy any phrase longer than three words from the passage,
except names, codes and numbers, and mention at most two names, codes or numbers. Do not mention "the passage" or "the document". Reply with the query only, one line."""
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-14B-AWQ", max_model_len=3072, gpu_memory_utilization=0.90, seed=0, max_num_seqs=128, enable_prefix_caching=True)
sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=120, seed=0)
TOK = llm.get_tokenizer()
cut = lambda t: TOK.decode(TOK(t, add_special_tokens=False)["input_ids"][:2400])   # a URL-list "chunk" hit 3,073 tokens
msgs = []
for r in rows:
    p = r["passages"][r["seed"]]
    src = p["meta"].get("domain") or p["meta"].get("title") or p["meta"].get("section") or "web"
    r["specific"] = rng.random() < SPECIFIC
    msgs.append([{"role": "user", "content": PROMPT.format(src=src, text=cut(p["text"]), ask=ASK[r["qtype"]], qlang=LANGS[r["qlang"]],
                 spec=" The query must name the specific organisation, product, place, person or case the passage is about "
                      "(take the name from the passage or the source line; name a website only when it is that organisation, "
                      "never \"as described on ...\"), so that it is answered by this passage and not by a similar page "
                      "of another organisation." if r["specific"] else "")}])
t0 = time.time()
outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
bad = Counter(); per = {l: [] for l in LANGS}
for r, o in zip(rows, outs):
    q = o.outputs[0].text.strip().strip('"“”「」').strip()
    if o.outputs[0].finish_reason != "stop" or not q or "\n" in q or re.search(r"\b(the|this) (passage|document)\b", q, re.I):
        bad["format"] += 1; continue
    r["query"] = q; per[r["lang"]].append(r)
print(f"{sum(map(len, per.values()))}/{len(rows)} queries in {time.time() - t0:.0f}s | rejected {dict(bad)} | types "
      + " ".join(f"{k}:{v}" for k, v in Counter(r['qtype'] for R in per.values() for r in R).items()), flush=True)
for lang, R in per.items():
    with open(f"{D}/rows_{'' if V == 'v1' else V + '_'}{lang}{'_smoke' if SMOKE else ''}.jsonl", "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in R)
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
