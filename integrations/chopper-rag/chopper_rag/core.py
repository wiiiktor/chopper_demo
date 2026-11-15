"""The chopper: a fine-tuned mmBERT encoder that cuts exact quotes ("chops") out of retrieved passages.

Input to the model, one string of at most 8,192 tokens:

    "Question: <question> [SEP] <passage 1> [SEP] ... [SEP] <passage N>"

Per token the KEEP head gives p(KEEP); the decision head on the first token gives SEARCH (0) / ANSWER (1).
Decoding (the one the published results were measured with): keep passage tokens with p > pred_thr, merge kept runs
separated by <= merge_gap tokens, drop runs shorter than min_run tokens, bridge the single character between two kept
tokens, then map each run back to character offsets of the ORIGINAL passage strings. Finally the chops are snapped
(``snap``): "sentence" (default since 0.1.7: every sentence a chop touches is kept whole, so chops read as complete
sentences; on a held-out set ~4.1% of the text is kept instead of 3.4%), "word" (never cut inside a word; F1 unchanged
vs raw, while 24% of raw chops started or ended mid-word) or "none" (raw token runs, as the published results were measured).

Passages that do not fit into one 8,192-token input are processed in several forward passes: passages are packed
greedily, in order, into groups that fit next to the question; a single passage longer than the whole budget is cut
into windows at token boundaries. Each group is chopped exactly like a single input; chop offsets always refer to the
original passage. The model was trained on inputs of up to ~20 passages of ~1,000 characters, so splitting changes
nothing for typical retriever outputs (16-20 passages fit in one pass).

The same pass also ranks the passages (it can replace a reranker): a passage's relevance is the mean p(KEEP) of its 16
most likely tokens. On a held-out rerank set (8 languages, 16 passages per query) this ranks as well as a dedicated
cross-encoder: nDCG@10 0.973 vs bge-reranker-v2-m3 0.984 and Qwen3-Reranker-4B 0.986, at the speed of bge-reranker-v2-m3.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_REPO = "wiiiktor/chopper_mmbert"
VARIANTS = {"base": "mmbert-base", "small": "mmbert-small", "base-pl": "mmbert-base-pl", "small-pl": "mmbert-small-pl"}
MAX_LEN = 8192
TOP_K_TOKENS = 16    # passage relevance = mean p(KEEP) of its TOP_K_TOKENS most likely tokens
_SAFETY = 32          # tokens kept free when packing groups (tokenisation at segment borders may differ slightly)


@dataclass
class Chop:
    """One exact quote: ``passages[passage_index][start:end] == text``."""
    passage_index: int
    start: int
    end: int
    text: str
    score: float | None = None        # mean p(KEEP) over the run's tokens


@dataclass
class ChopResult:
    chops: list[Chop]
    decision: str                     # "ANSWER" (the passages hold the answer) or "SEARCH" (search again)
    p_answer: float                   # p(ANSWER); with several forward passes, the maximum over them
    n_passes: int = 1                 # forward passes used (> 1 when the input did not fit 8,192 tokens)
    texts: list[str] = field(default_factory=list)   # chop texts, in order
    passage_scores: list[float] = field(default_factory=list)   # relevance per input passage (0..1), for ranking

    def ranking(self, top_k: int | None = None) -> list[tuple[int, float]]:
        """(passage_index, relevance), most relevant first: the chopper used as a reranker"""
        r = sorted(enumerate(self.passage_scores), key=lambda x: -x[1])
        return r[:top_k] if top_k else r

    def __iter__(self):
        return iter(self.chops)

    def __len__(self):
        return len(self.chops)


# ------------------------------------------------------------------ snapping chops to words / sentences
SNAPS = ("word", "sentence", "none")
_WORD = re.compile(r"\w")
_SENT_END = re.compile(r"[.!?。！？…]+[\"'”’»)\]]*(?=\s|$)|\n\s*\n|\n(?=\s*[-•*\d])")


def _sentences(text):
    """(start, end) of sentences: ends at . ! ? (plus closing quotes/brackets) before whitespace, blank lines, list items"""
    out, a = [], 0
    for m in _SENT_END.finditer(text):
        b = m.end()
        if text[a:b].strip():
            out.append((a, b))
        a = b
    if text[a:].strip():
        out.append((a, len(text)))
    return out


def snap_chops(chops, passages, mode="word"):
    """extend chops to whole words ("word") or whole sentences ("sentence"); merge chops that then touch or overlap"""
    if mode == "none" or not chops:
        return chops
    sents = {}
    out = []
    for c in chops:
        t = passages[c.passage_index]; a, b = c.start, c.end
        if mode == "sentence":
            ss = sents.setdefault(c.passage_index, _sentences(t))
            a = min([x for x, y in ss if y > a and x <= a] + [a]); b = max([y for x, y in ss if x < b and y >= b] + [b])
        while a > 0 and _WORD.match(t[a - 1]) and _WORD.match(t[a]):
            a -= 1
        while b < len(t) and _WORD.match(t[b - 1]) and _WORD.match(t[b]):
            b += 1
        while a < b and t[a].isspace():
            a += 1
        while b > a and t[b - 1].isspace():
            b -= 1
        if out and out[-1].passage_index == c.passage_index and a <= out[-1].end:
            p = out[-1]; p.end = max(p.end, b); p.text = t[p.start:p.end]
            continue
        out.append(Chop(c.passage_index, a, b, t[a:b], c.score))
    return out


# ------------------------------------------------------------------ decoding (identical to the reference operator)
def _runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def shape(q, gap=5, mn=4):
    """merge kept runs <= gap tokens apart, drop runs shorter than mn"""
    q = q.copy()
    rr = _runs(q)
    for (a0, b0), (a1, b1) in zip(rr, rr[1:]):
        if a1 - b0 <= gap:
            q[b0:a1] = True
    for a0, b0 in _runs(q):
        if b0 - a0 < mn:
            q[a0:b0] = False
    return q


def kept_text(s, offsets, q):
    """kept tokens -> contiguous character runs of the string s"""
    cm = np.zeros(len(s) + 1, dtype=bool)
    for (a, b), k in zip(offsets, q):
        if k:
            cm[a:b] = True
    cm[1:-1] |= cm[:-2] & cm[2:]                     # bridge the space between two kept tokens
    return [(int(a), int(b)) for a, b in _runs(cm)]


def _fetch(repo, folder):
    """a local copy (repo is a directory holding mmbert-base/ ...) or a Hugging Face download"""
    if os.path.isdir(os.path.join(repo, folder)):
        return os.path.join(repo, folder)
    from huggingface_hub import snapshot_download
    # the root config.json is fetched too: the Hub counts a repo's downloads by it
    return os.path.join(snapshot_download(repo, allow_patterns=[f"{folder}/*", "config.json"]), folder)


class Chopper:
    """Load with :meth:`from_pretrained`; call :meth:`chop` (list of :class:`Chop`) or :meth:`run` (+ decision)."""

    def __init__(self, tokenizer, encoder, heads, device, threshold=0.5, merge_gap=5, min_run=4, max_length=MAX_LEN,
                 snap="sentence"):
        self.tokenizer, self.encoder, self.device = tokenizer, encoder, device
        self.heads = {k: v.to(device).float() for k, v in heads.items()}
        self.threshold, self.merge_gap, self.min_run, self.max_length = threshold, merge_gap, min_run, max_length
        self.sep = f" {tokenizer.sep_token} "
        if snap not in SNAPS:
            raise ValueError(f"snap must be one of {SNAPS}, got {snap!r}")
        self.snap = snap

    @classmethod
    def from_pretrained(cls, repo: str = DEFAULT_REPO, variant: str = "base", device: str = "auto",
                        threshold: float | None = None, dtype: torch.dtype | None = None, snap: str = "sentence") -> "Chopper":
        """repo: HF repo id or a local directory containing ``mmbert-base/`` / ``mmbert-small/``.
        variant: "base" (0.31B, recommended) or "small" (0.14B); "base-pl" / "small-pl" for Polish documents. threshold: p(KEEP) cut-off (default from
        chopper_config.json). dtype: default bfloat16 on CUDA, float32 on CPU. snap: "sentence" (default), "word" or "none"."""
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}, got {variant!r}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        d = _fetch(repo, VARIANTS[variant])
        try:
            tok = AutoTokenizer.from_pretrained(f"{d}/encoder")
        except ValueError:                           # transformers 4.x cannot resolve the class name saved by 5.x
            from transformers import PreTrainedTokenizerFast
            cfg = json.load(open(f"{d}/encoder/tokenizer_config.json"))
            sp = {k: cfg[k] for k in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token") if cfg.get(k)}
            tok = PreTrainedTokenizerFast(tokenizer_file=f"{d}/encoder/tokenizer.json", model_max_length=cfg.get("model_max_length", 8192),
                                          padding_side=cfg.get("padding_side", "right"),
                                          model_input_names=cfg.get("model_input_names", ["input_ids", "attention_mask"]), **sp)
        import transformers
        kw = {"dtype": dtype} if int(transformers.__version__.split(".")[0]) >= 5 else \
            {"torch_dtype": dtype, "reference_compile": False}      # transformers 4.x: old dtype name, no torch.compile (needs triton)
        enc = AutoModel.from_pretrained(f"{d}/encoder", **kw).to(device).eval()
        heads = load_file(f"{d}/heads.safetensors")
        dec = json.load(open(f"{d}/chopper_config.json")).get("decode", {})
        thr = dec.get("pred_thr", 0.5) if threshold is None else threshold
        return cls(tok, enc, heads, device, threshold=thr, merge_gap=dec.get("merge_gap", 5),
                   min_run=dec.get("min_run", 4), snap=snap)

    # -------------------------------------------------------------- public API
    def chop(self, question: str, passages: list[str]) -> list[Chop]:
        """the chops (exact quotes) of ``passages`` for ``question``, in passage order"""
        return self.run(question, passages).chops

    def run(self, question: str, passages: list[str]) -> ChopResult:
        """chops plus the ANSWER / SEARCH decision"""
        passages = [p if isinstance(p, str) else str(p) for p in passages]
        head = f"Question: {question}"
        segs = self._segments(head, passages)
        chops, p_ans, n = [], 0.0, 0
        rel = [0.0] * len(passages)
        for group in self._groups(head, segs):
            c, pa = self._forward(head, group, passages, rel)
            chops += c; p_ans = max(p_ans, pa); n += 1
        if n == 0:                                                     # no passages: nothing to keep, search
            p_ans = self._forward(head, [], passages, rel)[1]; n = 1
        chops.sort(key=lambda c: (c.passage_index, c.start))
        chops = snap_chops(chops, passages, self.snap)
        return ChopResult(chops, "ANSWER" if p_ans >= 0.5 else "SEARCH", p_ans, n, [c.text for c in chops],
                          [round(x, 4) for x in rel])

    __call__ = run

    def rerank(self, question: str, passages: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        """(passage_index, relevance) most relevant first: the chopper as a reranker (chops come free with :meth:`run`)"""
        return self.run(question, passages).ranking(top_k)

    # -------------------------------------------------------------- internals
    def _ntok(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _segments(self, head, passages):
        """(passage_index, char offset in the passage, text): stripped passages, too-long ones cut into windows"""
        budget = self.max_length - self._ntok(head) - 2 - _SAFETY       # [CLS]/[SEP] + one separator
        if budget < 16:
            raise ValueError("the question alone fills the 8,192-token input")
        segs = []
        for i, p in enumerate(passages):
            s = p.strip()
            if not s:
                continue
            lead = len(p) - len(p.lstrip())
            e = self.tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
            off = e["offset_mapping"]
            if len(off) <= budget:
                segs.append((i, lead, s, len(off)))
                continue
            for w in range(0, len(off), budget):                       # one very long passage: token windows
                a, b = off[w][0], off[min(w + budget, len(off)) - 1][1]
                t = s[a:b]
                tl = len(t) - len(t.lstrip())
                if t.strip():
                    segs.append((i, lead + a + tl, t.strip(), min(budget, len(off) - w)))
        return segs

    def _groups(self, head, segs):
        """greedy, in order: as many segments per forward pass as fit into max_length"""
        limit = self.max_length - self._ntok(head) - 2 - _SAFETY
        group, used = [], 0
        for sg in segs:
            cost = sg[3] + 2                                           # the passage + its " [SEP] "
            if group and used + cost > limit:
                yield group; group, used = [], 0
            group.append(sg); used += cost
        if group:
            yield group

    @torch.no_grad()
    def _forward(self, head, group, passages, rel=None):
        s, spans = head, []
        for (i, off, text, _) in group:
            s += self.sep
            spans.append((len(s), len(s) + len(text), i, off))
            s += text
        e = self.tokenizer(s, return_offsets_mapping=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt")
        if len(group) > 1 and e["input_ids"].shape[1] >= self.max_length:  # estimate was off: split the group
            m = len(group) // 2
            c1, p1 = self._forward(head, group[:m], passages, rel)
            c2, p2 = self._forward(head, group[m:], passages, rel)
            return c1 + c2, max(p1, p2)
        off = e.pop("offset_mapping")[0].tolist()
        h = self.encoder(**{k: v.to(self.device) for k, v in e.items()}).last_hidden_state[0].float()
        p = torch.sigmoid(h @ self.heads["keep.weight"].T + self.heads["keep.bias"])[:, 0].cpu().numpy()
        p_ans = float(torch.softmax(h[0] @ self.heads["decision.weight"].T + self.heads["decision.bias"], -1)[1])
        body = np.array([a >= len(head) and b > a for a, b in off])
        if rel is not None and spans:                                  # passage relevance: mean of its top tokens' p(KEEP)
            st = np.array([a for a, _ in off]); en = np.array([b for _, b in off])
            for (sa, sb, i, _) in spans:
                v = p[(st >= sa) & (en <= sb) & (en > st)]
                if len(v):
                    rel[i] = max(rel[i], float(np.sort(v)[-TOP_K_TOKENS:].mean()))
        q = shape((p > self.threshold) & body, self.merge_gap, self.min_run) & body
        out = []
        for a, b in kept_text(s, off, q):
            for (sa, sb, i, poff) in spans:                            # a run may cross a separator: cut per passage
                x, y = max(a, sa), min(b, sb)
                if y <= x:
                    continue
                t = s[x:y]
                x += len(t) - len(t.lstrip()); y -= len(t) - len(t.rstrip())
                if y - x <= 2:
                    continue
                ts = [k for k, (u, v) in enumerate(off) if q[k] and u < y and v > x]
                start, end = poff + (x - sa), poff + (y - sa)
                out.append(Chop(i, start, end, passages[i][start:end], float(p[ts].mean()) if ts else None))
        return out, p_ans
