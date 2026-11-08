#!/usr/bin/env python3
"""Step 2 of the rerank dataset pilot: synthetic documents where the web pool is thin. A cell (language x area) needs
625 seed passages that Qwen3-4B verified (area agreed, quality 2, area_<lang>.jsonl); the shortfall, and at least half
of every support / internal-knowledge cell (taxonomy pilot.synthetic_share_areas_1_2), is written by Qwen3-14B-AWQ.
Every document carries the full metadata set (title, author, date, section, type) for the source tags of step 4.
Writes <dir>/synth_<lang>.jsonl {id, lang, area, title, author, date, section, type, text}.
usage: rr_synth.py <dir> [N_SMOKE]"""
import json, os, random, re, sys, time
from collections import Counter
from vllm import LLM, SamplingParams

D = sys.argv[1]; SMOKE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.json")))
NEED = TX["pilot"]["rows_per_language_area"]
LANGS = {"en": "English", "zh": "Simplified Chinese", "es": "Spanish", "de": "German", "fr": "French", "ja": "Japanese",
         "pt": "Portuguese", "pl": "Polish"}
DOC = {   # document kinds per area; with a random sector they keep 10k documents from collapsing onto one template
    1: ["help-centre article", "troubleshooting guide for an error message", "returns and refunds policy page",
        "billing FAQ", "account settings how-to", "shipping and delivery FAQ", "warranty claim instructions",
        "subscription cancellation guide", "known-issue notice with workaround", "password and login help article"],
    2: ["HR policy page on an intranet wiki", "IT onboarding instructions", "expense reimbursement procedure",
        "remote-work policy", "travel booking rules for employees", "benefits overview for new hires",
        "VPN and laptop setup guide", "incident escalation procedure", "parental leave policy", "meeting room booking guide",
        "internal tool release notes", "security awareness rules"],
    4: ["contract clause explanation", "terms and conditions excerpt", "privacy policy section", "court decision summary",
        "regulation explainer"],
    6: ["API reference page", "configuration guide", "CLI tool documentation", "developer FAQ", "migration guide"],
    7: ["patient information leaflet", "clinical guideline summary", "drug interaction notice", "condition explainer from a clinic",
        "research finding summary"],
    8: ["user manual section", "installation instructions", "maintenance schedule", "spare-parts list with part numbers",
        "technical specification sheet", "safety instructions", "error-code table explanation", "quick start guide"],
}
SECTORS = ["telecom", "bank", "online retailer", "airline", "insurance company", "software startup", "hospital network",
           "university", "logistics company", "car manufacturer", "home appliance maker", "energy utility",
           "city administration", "pharmacy chain", "hotel group's corporate office", "game studio", "law firm",
           "manufacturing plant", "public transport operator", "cloud hosting provider", "fitness chain", "publisher",
           "agricultural machinery maker", "medical device maker", "e-bike maker", "accounting firm"]
PROMPT = """Write one realistic {kind} for a fictional {sector}, in {lang}, as it would appear on its website or intranet.
Topic area: {area}. The page is dated {date}; any dates in it must be consistent with that.
Rules: 150-300 words ({cjk}); concrete and searchable facts (names of options, menu paths, numbers, deadlines, amounts,
part or error codes, conditions, exceptions); invented organisation, product and component names, never real brands; plain text,
no markdown, no bullets with symbols (numbered steps are fine); natural style for {lang} documents of this kind.
Reply in exactly this format, field names in English, values in {lang}:
TITLE: <page title>
AUTHOR: <{author}>
SECTION: <breadcrumb, e.g. Help > Billing>
TEXT:
<the document>"""

cells = []
for lang in LANGS:
    ok = Counter(x["area"] for x in map(json.loads, open(f"{D}/area_{lang}.jsonl")) if x["area"] == x["cand_area"] and x["q"] == 2)
    for a in range(1, 9):
        n = max(NEED - ok[a], int(NEED * TX["pilot"]["synthetic_share_areas_1_2"]) if a in (1, 2) else 0)
        cells += [(lang, a)] * n
rng = random.Random(0); rng.shuffle(cells)
if SMOKE:
    cells = cells[:SMOKE]
jobs = []
for i, (lang, a) in enumerate(cells):
    kind = rng.choice(DOC.get(a, [TX["areas"][str(a)].split(":")[0]]))
    jobs.append({"i": i, "lang": lang, "area": a, "kind": kind, "sector": rng.choice(SECTORS),
                 # Qwen's own dates collapse onto the 15th of 2023 (smoke: 42/48) -> a synthetic-vs-web shortcut; drawn here
                 "date": f"{rng.randint(2016, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                 "author": rng.choice(["a full personal name typical for the language", "a team or department name"])})
print(f"{len(jobs)} documents | per area " + " ".join(f"{k}:{v}" for k, v in sorted(Counter(j['area'] for j in jobs).items())), flush=True)

llm = LLM(model="Qwen/Qwen3-14B-AWQ", max_model_len=2048, gpu_memory_utilization=0.90, seed=0, max_num_seqs=128)
sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=900, seed=0)
msgs = [[{"role": "user", "content": PROMPT.format(kind=j["kind"], sector=j["sector"], lang=LANGS[j["lang"]], area=TX["areas"][str(j["area"])], author=j["author"], date=j["date"],
                                                   cjk="characters" if j["lang"] in ("zh", "ja") else "words")}] for j in jobs]
t0 = time.time()
outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
bad = Counter(); per = {l: [] for l in LANGS}
for j, o in zip(jobs, outs):
    t = o.outputs[0].text
    m = re.search(r"TITLE:\s*(.+)\nAUTHOR:\s*(.+)\nSECTION:\s*(.+)\nTEXT:\s*\n(.+)", t, re.S)
    if not m:
        bad["format"] += 1; continue
    text = m.group(4).strip()
    size = len(re.sub(r"\s", "", text)) / 2 if j["lang"] in ("zh", "ja") else len(text.split())   # 2 CJK chars ~ 1 word
    if not 100 <= size <= 450:
        bad["length"] += 1; continue
    if o.outputs[0].finish_reason != "stop":
        bad["truncated"] += 1; continue
    per[j["lang"]].append({"id": f"syn-{j['lang']}-{j['i']}", "lang": j["lang"], "area": j["area"], "title": m.group(1).strip(),
                           "author": m.group(2).strip(), "date": j["date"], "section": m.group(3).strip(), "type": j["kind"], "text": text})
n = sum(map(len, per.values()))
print(f"{n}/{len(jobs)} kept in {time.time() - t0:.0f}s | rejected {dict(bad)}", flush=True)
for lang, R in per.items():
    with open(f"{D}/synth_{lang}{'_smoke' if SMOKE else ''}.jsonl", "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in R)
sys.stdout.flush(); os._exit(0)                        # vLLM V1 never exits on its own
