#!/usr/bin/env python3
"""Chopper vs XProvence / Provence / LLMLingua-2 on THE XPROVENCE PAPER'S OWN BENCHMARKS (user 2026-11-15: "run also on
their benchmarks").

Source: XProvence (arXiv 2601.18886, ECIR'26) released its evaluation runs (Bergen, naver/bergen scripts/xprovence, Google
Drive 1MT6WUqFLpPFPdE1_fSMr33KAi0VSMTTi -> /media/wii/bigdata/pruners_theirs/xprov_runs.tar.gz). Every no-pruning run's
eval_dev_out.json holds, per question, the FULL prompt the generator saw, i.e. their exact retrieved top-5 (BGE-m3 retrieval
+ bge-reranker-v2-m3 over Wikipedia 100-word chunks for MKQA / TyDi QA; oracle provenance for XPQA / MedExpQA), their
translated system prompt and their gold labels. We rebuild that prompt with each pruner's output in place of the documents,
so the contexts, prompts, labels and metric are theirs; only the generator differs.

Faithful: datasets, languages, retrieved documents, prompt text, labels, metric (Bergen Recall_char3gram = the paper's
  main axis; also M = label substring match; MedExpQA: accuracy of "Answer: N"), compression = % of document characters
  removed (their compression_ratio).
Generators: Qwen3-14B-AWQ (default) and, with PT_GEN_TAG=aya_ PT_GEN_MODEL=CohereLabs/aya-expanse-8b, the paper's own aya-expanse-8b
  (bf16 here, int4 in the paper; user accepted the terms 2026-11-15). Old note: generator = Qwen3-14B-AWQ (vLLM, greedy, no thinking) instead of aya-expanse-8b int4 (gated + non-commercial,
  not accepted on this account); system+user text go through Qwen's chat template. aya's own per-row no-pruning scores on
  the same rows are reported as `aya_full_ref` for orientation. Samples capped (CAP per set, first CAP rows by a fixed seed).
Methods (per query, docs in their order):
  full         their 5 retrieved documents untouched        closed       no documents at all (closed-book floor)
  chopper      released mmBERT-base chopper, one pass over all docs, kept chops joined by " … " per document
  chopper+t    same + the document title (text before the first ". ") when anything is kept (wiki sets only; XProvence
               always keeps the title on these sets)
  xprov@t      naver/xprovence-reranker-bgem3-v2, t = 0.1 / 0.3 / 0.7 (title = first sentence on wiki sets, none otherwise)
  provence     naver/provence-reranker-debertav3-v1 @0.1, English sets only
  llml2        LLMLingua-2 (xlm-roberta-large meetingbank), rate 0.33, question-agnostic
Sets (CAP=150 each): MKQA en de es fr pl pt ja zh ar ru | TyDi QA en ru fi | XPQA de es fr pl pt ja zh | MedExpQA en es fr it

usage: compare_pruners_theirs.py prep|prune|gen|score [CAP]
  prep  (any python)       sets.jsonl from the extracted runs          prune (.venv-prune)  ctx_<method>.jsonl
  gen   (.venv-vllm)       ans_<method>.jsonl                          score (any python)   results.json + table
"""
import json, os, random, re, string, sys, time
from collections import Counter, defaultdict

D0 = "/media/wii/bigdata/pruners_theirs"
D = os.environ.get("PT_DIR", D0)                       # outputs (+ sets.jsonl); PT_DIR for smoke tests
EXP = f"{D0}/experiments_final_with_devout"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SETS = ([f"mkqa_{l}" for l in "en de es fr pl pt ja zh ar ru".split()] + [f"tydiqa_{l}" for l in "en ru fi".split()]
        + [f"xpqa_{l}" for l in "de es fr pl pt ja zh".split()] + [f"medexpqaexp2_{l}" for l in "en es fr it".split()])
WIKI = ("mkqa_", "tydiqa_")
DOC = re.compile(r"Document (\d+): ")


def run_dir(s):
    pre = "bgem3_bgem3" if s.startswith(WIKI) else "oracle_provenance"
    return f"{EXP}/{pre}_noprovence00_ayaexpanse8b_{s}_btl"


# ------------------------------------------------------------------ prep: parse their prompts
def prep():
    out = open(f"{D}/sets.jsonl", "w"); n = Counter()
    for s in SETS:
        rows = json.load(open(f"{run_dir(s)}/eval_dev_out.json"))
        random.Random(0).shuffle(rows)
        for x in rows:
            if n[s] >= CAP:
                break
            ins = x["instruction"]
            sysp = ins.split("<|SYSTEM_TOKEN|>")[1].split("<|END_OF_TURN_TOKEN|>")[0]
            u = ins.split("<|USER_TOKEN|>")[1].split("<|END_OF_TURN_TOKEN|>")[0]
            m = list(DOC.finditer(u))
            qi = u.rfind(x["question"][:40].strip())
            if not m or qi < 0:
                continue
            k = u.rfind("\n\n", 0, qi)
            if k < m[-1].end():
                continue
            head, tail = u[:m[0].start()], u[k:]
            region = u[m[-1].end():k]; gap = region[len(region.rstrip()):]
            docs = [u[a.end():(m[j + 1].start() if j + 1 < len(m) else k)].strip() for j, a in enumerate(m)]
            out.write(json.dumps({"set": s, "qid": x["q_id"], "system": sysp, "head": head, "gap": gap, "tail": tail, "docs": docs,
                                  "question": x["question"], "labels": eval(x["label"]) if isinstance(x["label"], str) else x["label"],
                                  "aya_full": {"M": float(x["M"]), "R3": float(x["Recall_char3gram"])}}, ensure_ascii=False) + "\n")
            n[s] += 1
    print(dict(n), sum(n.values()))


def load_sets():
    return [json.loads(l) for l in open(f"{D}/sets.jsonl")]


# ------------------------------------------------------------------ prune
def prune():
    import numpy as np, torch
    sys.path.insert(0, "release/integrations/chopper-rag")
    from chopper_rag.core import Chopper
    R = load_sets(); dev = "cuda"; times = {}
    import spacy
    nlp = spacy.load("xx_sent_ud_sm")

    def save(name, ctxs):
        with open(f"{D}/ctx_{name}.jsonl", "w") as f:
            for r, c in zip(R, ctxs):
                f.write(json.dumps({"qid": r["qid"], "set": r["set"], "docs": c}, ensure_ascii=False) + "\n")
        print(name, "saved", flush=True)

    def title(d):
        i = d.find(". ")
        return d[:i + 1] if 0 < i < 200 else ""

    # chopper: raw chops (snap="none", as the published numbers) and the package default since 0.1.7 (snap="sentence")
    ch = Chopper.from_pretrained(variant="base", device=dev)
    for suf, snap in (("", "none"), ("_sent", "sentence")):
        ch.snap = snap
        A, B = [], []; t = time.time()
        for r in R:
            o = ch.run(r["question"], r["docs"])
            per = defaultdict(list)
            for c in o.chops:
                per[c.passage_index].append(c.text)
            a = [" … ".join(per[i]) for i in range(len(r["docs"]))]
            A.append(a)
            B.append([(title(d) + " " + x).strip() if x and r["set"].startswith(WIKI) and not x.startswith(title(d)[:-1] or "\0")
                      else x for d, x in zip(r["docs"], a)])
        times["chopper" + suf] = (time.time() - t) / len(R)
        save("chopper" + suf, A); save("chopper+t" + suf, B)
    del ch; torch.cuda.empty_cache()

    from transformers import AutoModel

    def provence_like(name, thrs, only_en=False):
        m = AutoModel.from_pretrained(name, trust_remote_code=True).to(dev).eval()
        res = {t_: [] for t_ in thrs}; t0 = time.time(); n = 0
        for r in R:
            if only_en and not r["set"].endswith("_en"):
                for t_ in thrs:
                    res[t_].append(None)
                continue
            wiki = r["set"].startswith(WIKI); n += 1
            for t_ in thrs:
                with torch.no_grad():
                    o = m.process([r["question"]], [r["docs"]], threshold=t_, always_select_title=wiki,
                                  title="first_sentence" if wiki else None, enable_warnings=False)
                res[t_].append([str(x) for x in o["pruned_context"][0]])
        del m; torch.cuda.empty_cache()
        return res, (time.time() - t0) / max(n, 1)

    res, times["xprov"] = provence_like("naver/xprovence-reranker-bgem3-v2", [0.1, 0.3, 0.7])
    for t_, v in res.items():
        save(f"xprov@{t_}", v)
    res, times["provence"] = provence_like("naver/provence-reranker-debertav3-v1", [0.1], only_en=True)
    save("provence", res[0.1])

    from llmlingua import PromptCompressor
    lc = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map=dev)
    C = []; t = time.time()
    for r in R:
        C.append([lc.compress_prompt(d, rate=0.33, force_tokens=["\n"], drop_consecutive=True)["compressed_prompt"] if d.strip() else ""
                  for d in r["docs"]])
    times["llml2"] = (time.time() - t) / len(R)
    save("llml2", C)
    save("full", [r["docs"] for r in R]); save("closed", [[] for _ in R])
    json.dump(times, open(f"{D}/prune_sec_per_query.json", "w"), indent=1)


TAG = os.environ.get("PT_GEN_TAG", "")                 # "" = Qwen3-14B-AWQ (ans_<m>), "aya_" = aya-expanse-8b (ans_aya_<m>)
METHODS = ["full", "closed", "chopper", "chopper+t", "chopper_sent", "chopper+t_sent", "xprov@0.1", "xprov@0.3", "xprov@0.7", "provence", "llml2"]


def build_user(r, docs):
    if not docs:                                           # closed book: their prompt with the documents block removed
        return (r["head"].rstrip() + r["tail"]) if r["head"].strip() else r["tail"].lstrip()
    return r["head"] + "\n".join(f"Document {i + 1}: {d}" for i, d in enumerate(docs)) + r["gap"] + r["tail"]


# ------------------------------------------------------------------ gen (vLLM, Qwen3-14B-AWQ)
def gen():
    from vllm import LLM, SamplingParams
    R = load_sets()
    ML = int(os.environ.get("MAXLEN", "8192"))
    llm = LLM(model=os.environ.get("PT_GEN_MODEL", "Qwen/Qwen3-14B-AWQ"), max_model_len=ML,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")), seed=0)
    tok = llm.get_tokenizer()
    for meth in METHODS:
        fo = f"{D}/ans_{TAG}{meth}.jsonl"
        if os.path.exists(fo):
            continue
        C = [json.loads(l) for l in open(f"{D}/ctx_{meth}.jsonl")]
        idx = [i for i, c in enumerate(C) if c["docs"] is not None]
        prompts, sps = [], []
        for i in idx:
            r = R[i]
            msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": build_user(r, C[i]["docs"])}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        **({} if TAG else {"enable_thinking": False}))
            ids = tok(p, add_special_tokens=False)["input_ids"]
            mx = 256 if r["set"].startswith("medexp") else 128
            if len(ids) > ML - mx:                                  # rare: keep the end (question + last docs)
                p = tok.decode(ids[-(ML - mx):])
            prompts.append(p); sps.append(SamplingParams(temperature=0, max_tokens=mx))
        t = time.time()
        outs = llm.generate(prompts, sps)
        with open(fo, "w") as f:
            for i, o in zip(idx, outs):
                f.write(json.dumps({"qid": R[i]["qid"], "set": R[i]["set"], "response": o.outputs[0].text}, ensure_ascii=False) + "\n")
        print(meth, len(idx), f"{time.time() - t:.0f}s", flush=True)


# ------------------------------------------------------------------ score (Bergen metrics, copied from naver/bergen modules/metrics.py)
def normalize(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def ngrams(s, n=3):
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s.lower()); out = []
    for w in s.split():
        out += [w] if len(w) < n else [w[i:i + n] for i in range(len(w) - n + 1)]
    return out


def recall3(pred, gts):
    best = 0.0
    for gt in gts:
        p, g = ngrams(normalize(pred)), ngrams(normalize(gt))
        same = sum((Counter(p) & Counter(g)).values())
        best = max(best, same / len(g) if same and g else 0.0)
    return best


def match(pred, gts):
    return float(any(normalize(g) in normalize(pred) for g in gts))


def medexp_acc(pred, gts):
    m = re.findall(r"Answer:\s*\**\s*([1-5])", pred)
    return float(bool(m) and m[-1] == gts[0].strip()[0])


def score():
    R = {f"{r['qid']}|{r['set']}": r for r in load_sets()}
    res = {}; table = defaultdict(dict)
    for meth in METHODS:
        if not os.path.exists(f"{D}/ans_{TAG}{meth}.jsonl"):
            continue
        ctx = {f"{c['qid']}|{c['set']}": c["docs"] for c in map(json.loads, open(f"{D}/ctx_{meth}.jsonl"))}
        per = defaultdict(lambda: defaultdict(list))
        for a in map(json.loads, open(f"{D}/ans_{TAG}{meth}.jsonl")):
            k = f"{a['qid']}|{a['set']}"; r = R[k]; s = a["set"]
            full = sum(len(d) for d in r["docs"]); kept = sum(len(d) for d in (ctx[k] or []))
            per[s]["R3"].append(recall3(a["response"], r["labels"])); per[s]["M"].append(match(a["response"], r["labels"]))
            per[s]["comp"].append(100 * (1 - kept / full) if full else 0.0)
            if s.startswith("medexp"):
                per[s]["acc"].append(medexp_acc(a["response"], r["labels"]))
            per[s]["aya_full_R3"].append(r["aya_full"]["R3"])
        res[meth] = {s: {k: round(sum(v) / len(v), 4) for k, v in d.items()} | {"n": len(d["R3"])} for s, d in per.items()}
    fam = lambda s: s.split("_")[0]
    for meth, d in res.items():
        for f in sorted({fam(s) for s in d}):
            ss = [s for s in d if fam(s) == f]
            key = "acc" if f.startswith("medexp") else "R3"
            table[meth][f] = (round(sum(d[s][key] for s in ss) / len(ss), 4), round(sum(d[s]["comp"] for s in ss) / len(ss), 1), len(ss))
    json.dump({"per_set": res, "by_family": table}, open(f"{D}/results{TAG and '_' + TAG.rstrip('_')}.json", "w"), indent=1)
    fams = sorted({f for t in table.values() for f in t})
    print("method".ljust(11) + "".join(f"{f[:12]:>22}" for f in fams) + "   (score / % removed / #sets)")
    for meth in METHODS:
        if meth in table:
            print(meth.ljust(11) + "".join(f"{str(table[meth].get(f, '-')):>22}" for f in fams))


if __name__ == "__main__":
    {"prep": prep, "prune": prune, "gen": gen, "score": score}[sys.argv[1]]()
