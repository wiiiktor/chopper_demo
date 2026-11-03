"""Shared parts of the operator (operator-for-llm-with-mmbert.py), 2026-10-31.

  Corpus     your documents → chunks ("Title\\nparagraph…", ≤ ~1,000 characters, like the training passages)
  Retriever  Qwen3-Embedding-0.6B top-100 → bge-reranker-v2-m3 → top 16 (the pipeline the choppers were trained on);
             chunk embeddings are cached next to the corpus
  Session    the answering model with ONE KV cache for the whole work session: prefill, generate, and crop (rollback)
  answer()   the loop: retrieve → chop → roll back → prefill only the notes → answer, or [search] <question> → again

Assumptions about the answering model ("any transformer"): a Hugging Face decoder-only causal LM whose cache is a
DynamicCache that supports crop() (Llama, Mistral, Gemma-2-style full-attention models; not sliding-window-only or
recurrent caches). Without an answerer LoRA the model follows a short instruction.
"""
import glob, hashlib, json, os, re, time
import torch
import torch.nn.functional as F

TASK = "Given a question, retrieve passages that contain facts needed to answer it"
SEARCH = "[search]"
INSTRUCTION = ("You answer questions using the notes given before each question. If the notes contain the answer, "
               "reply with the answer only (a few words). If they do not, reply exactly: [search] <one short question "
               "that would find the missing fact>.\n\n")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def device(arg):
    return arg if arg != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def fetch(weights, folder, files=("*",)):
    """a folder of the weights: a local directory, or downloaded from a Hugging Face repo (e.g. wiiiktor/chopper_mmbert)"""
    if os.path.isdir(os.path.join(weights, folder)):
        return os.path.join(weights, folder)
    from huggingface_hub import snapshot_download
    return os.path.join(snapshot_download(weights, allow_patterns=[f"{folder}/{f}" for f in files]), folder)


# ---------------------------------------------------------------- corpus
def chunk_text(title, text, size=1000):
    """paragraphs merged up to ~size characters; every chunk starts with the document title"""
    out, cur = [], ""
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        while len(p) > size:                                   # one very long paragraph: cut at a sentence end
            cut = p.rfind(". ", 0, size); cut = cut + 1 if cut > size // 3 else size
            if cur:
                out.append(cur); cur = ""
            out.append(p[:cut].strip()); p = p[cut:].strip()
        if cur and len(cur) + len(p) + 1 > size:
            out.append(cur); cur = ""
        cur = (cur + " " + p).strip() if p else cur
    if cur:
        out.append(cur)
    return [f"{title}\n{c}" for c in out if c]


def load_corpus(path):
    """a directory of .txt / .md files (title = file name), or a .jsonl with "text" (+ optional "title") per line"""
    docs = []
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True) + glob.glob(os.path.join(path, "**", "*.md"), recursive=True)):
            docs.append((os.path.splitext(os.path.basename(f))[0].replace("_", " "), open(f, errors="replace").read()))
    else:
        for l in open(path):
            d = json.loads(l); docs.append((d.get("title", ""), d["text"]))
    return [c for t, x in docs for c in chunk_text(t, x)]


# ---------------------------------------------------------------- retrieval
class Retriever:
    def __init__(self, chunks, cache_path, dev, K=16, R=100):
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        self.chunks, self.dev, self.K, self.R = chunks, dev, K, R
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        self.etk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
        self.emo = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=dt).to(dev).eval()
        self.rtk = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        self.rmo = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=dt).to(dev).eval()
        key = hashlib.sha1("\x00".join(chunks).encode()).hexdigest()[:16]
        if os.path.exists(cache_path) and torch.load(cache_path)["key"] == key:
            self.E = torch.load(cache_path)["E"].to(dev)
        else:
            t0 = time.time(); self.E = self.embed(chunks)
            torch.save({"key": key, "E": self.E.cpu()}, cache_path)
            log(f"embedded {len(chunks):,} chunks in {time.time() - t0:.0f} s → {cache_path}")

    @torch.no_grad()
    def embed(self, texts, query=False, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            t = [f"Instruct: {TASK}\nQuery:{x}" for x in texts[i:i + bs]] if query else texts[i:i + bs]
            b = self.etk(t, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
            out.append(F.normalize(self.emo(**b).last_hidden_state[:, -1].float(), dim=-1).half())
        return torch.cat(out)

    @torch.no_grad()
    def __call__(self, query):
        q = self.embed([query], query=True)[0].float()
        top = (self.E.float() @ q).argsort(descending=True)[:self.R].tolist()
        s = []
        for i in range(0, len(top), 50):
            b = self.rtk([[query, self.chunks[k]] for k in top[i:i + 50]], padding=True, truncation=True, max_length=640,
                         return_tensors="pt").to(self.dev)
            s.append(self.rmo(**b).logits.view(-1).float())
        order = torch.cat(s).argsort(descending=True).tolist()
        return [self.chunks[top[k]] for k in order[:self.K]]


# ---------------------------------------------------------------- decoding the KEEP probabilities
def runs(m):
    import numpy as np
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def shape(q, gap=5, mn=4):
    """the decode the results were measured with: merge kept runs ≤ gap tokens apart, drop runs shorter than mn"""
    q = q.copy()
    rr = runs(q)
    for (a0, b0), (a1, b1) in zip(rr, rr[1:]):
        if a1 - b0 <= gap:
            q[b0:a1] = True
    for a0, b0 in runs(q):
        if b0 - a0 < mn:
            q[a0:b0] = False
    return q


def kept_text(s, offsets, q):
    """kept tokens → contiguous runs of the original string s"""
    import numpy as np
    cm = np.zeros(len(s) + 1, dtype=bool)
    for (a, b), k in zip(offsets, q):
        if k:
            cm[a:b] = True
    cm[1:-1] |= cm[:-2] & cm[2:]                               # bridge the space between two kept tokens
    return [(int(a), int(b)) for a, b in runs(cm)]


# ---------------------------------------------------------------- the answering model and its session cache
class Session:
    def __init__(self, model, tok, dev, instruct):
        from transformers import DynamicCache
        self.m, self.tk, self.dev = model, tok, dev
        self.cache, self.len = DynamicCache(), 0
        self.eos = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")] if isinstance(i, int) and i >= 0}
        self.stop_nl = instruct                               # instruction-following models: stop at the end of the line
        if instruct:
            self.prefill(INSTRUCTION)

    @torch.no_grad()
    def prefill(self, text):
        ids = self.tk.encode(text, add_special_tokens=False)
        if not ids:
            return None
        x = torch.tensor([ids], device=self.dev)
        o = self.m(input_ids=x, past_key_values=self.cache, use_cache=True,
                   position_ids=torch.arange(self.len, self.len + len(ids), device=self.dev)[None])
        self.cache = o.past_key_values; self.len += len(ids)
        return o.logits[0, -1]

    def crop(self, n):
        """ROLLBACK: forget everything after position n (free; nothing before n is recomputed)"""
        self.cache.crop(n); self.len = n

    @torch.no_grad()
    def generate(self, last_logits, max_new=64):
        out, t = [], int(last_logits.argmax())
        for _ in range(max_new):
            if t in self.eos:
                break
            out.append(t)
            if self.stop_nl and "\n" in self.tk.decode([t]):
                break
            o = self.m(input_ids=torch.tensor([[t]], device=self.dev), past_key_values=self.cache, use_cache=True,
                       position_ids=torch.tensor([[self.len]], device=self.dev))
            self.cache = o.past_key_values; self.len += 1; t = int(o.logits[0, -1].argmax())
        return self.tk.decode(out).strip()                    # the stop token is not cached; answer() closes the turn


def notes_prompt(q, notes):
    """the answerer's training format (mm_qwen_answer.py)"""
    return "Notes:\n" + ("\n".join(f"- {' '.join(n.split())}" for n in notes) if notes else "(none)") + f"\n\nQuestion: {q}\nAnswer:"


def answer(question, sess, retrieve, chop, max_rounds=3, verbose=True):
    """one user question. chop(question, passages) → fragments. Returns the reply and per-round stats."""
    start = sess.len; notes, query, stats = [], question, []
    for r in range(max_rounds):
        passages = retrieve(query)
        frags = chop(question, passages)
        notes += [f for f in frags if f not in notes]
        sess.crop(start)                                       # ROLLBACK: drop the previous round's prompt and reply
        last = sess.prefill(notes_prompt(question, notes))
        reply = sess.generate(last)
        raw = sum(len(sess.tk.encode(p, add_special_tokens=False)) for p in passages)
        stats.append({"round": r + 1, "query": query, "fragments": len(frags), "passage_tokens": raw, "session_tokens": sess.len})
        if verbose:
            log(f"round {r + 1}: query {query!r} → 16 passages ({raw:,} tokens) → {len(frags)} fragments; reply: {reply!r}")
            for f in frags:
                print(f"      · {' '.join(f.split())}")
        if reply.lower().startswith(SEARCH) and r < max_rounds - 1:
            query = reply[len(SEARCH):].strip() or question
            continue
        break
    sess.prefill("\n\n")
    return reply, stats


def repl(sess, retrieve, chop, args):
    """--ask "q" (repeatable) runs those questions in one session; without it, an interactive prompt"""
    qs = args.ask or []
    it = iter(qs) if qs else None
    total_raw = 0
    while True:
        if it is not None:
            q = next(it, None)
            if q is None:
                break
            print(f"\n>>> {q}")
        else:
            try:
                q = input("\n>>> ").strip()
            except EOFError:
                break
            if q in ("", "exit", "quit"):
                break
        reply, st = answer(q, sess, retrieve, chop, args.max_rounds)
        total_raw += sum(s["passage_tokens"] for s in st)
        print(f"<<< {reply}")
        print(f"    session: {sess.len:,} tokens in the cache; plain RAG would have added {total_raw:,} passage tokens by now")
