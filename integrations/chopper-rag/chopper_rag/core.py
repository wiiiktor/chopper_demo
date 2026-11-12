"""The chopper: a fine-tuned mmBERT encoder that cuts exact quotes ("chops") out of retrieved passages.

Input to the model, one string of at most 8,192 tokens:

    "Question: <question> [SEP] <passage 1> [SEP] ... [SEP] <passage N>"

Per token the KEEP head gives p(KEEP); the decision head on the first token gives SEARCH (0) / ANSWER (1).
Decoding (the one the published results were measured with): keep passage tokens with p > pred_thr, merge kept runs
separated by <= merge_gap tokens, drop runs shorter than min_run tokens, bridge the single character between two kept
tokens, then map each run back to character offsets of the ORIGINAL passage strings.

Passages that do not fit into one 8,192-token input are processed in several forward passes: passages are packed
greedily, in order, into groups that fit next to the question; a single passage longer than the whole budget is cut
into windows at token boundaries. Each group is chopped exactly like a single input; chop offsets always refer to the
original passage. The model was trained on inputs of up to ~20 passages of ~1,000 characters, so splitting changes
nothing for typical retriever outputs (16-20 passages fit in one pass).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_REPO = "wiiiktor/chopper_mmbert"
VARIANTS = {"base": "mmbert-base", "small": "mmbert-small"}
MAX_LEN = 8192
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

    def __iter__(self):
        return iter(self.chops)

    def __len__(self):
        return len(self.chops)


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
    return os.path.join(snapshot_download(repo, allow_patterns=[f"{folder}/*"]), folder)


class Chopper:
    """Load with :meth:`from_pretrained`; call :meth:`chop` (list of :class:`Chop`) or :meth:`run` (+ decision)."""

    def __init__(self, tokenizer, encoder, heads, device, threshold=0.5, merge_gap=5, min_run=4, max_length=MAX_LEN):
        self.tokenizer, self.encoder, self.device = tokenizer, encoder, device
        self.heads = {k: v.to(device).float() for k, v in heads.items()}
        self.threshold, self.merge_gap, self.min_run, self.max_length = threshold, merge_gap, min_run, max_length
        self.sep = f" {tokenizer.sep_token} "

    @classmethod
    def from_pretrained(cls, repo: str = DEFAULT_REPO, variant: str = "base", device: str = "auto",
                        threshold: float | None = None, dtype: torch.dtype | None = None) -> "Chopper":
        """repo: HF repo id or a local directory containing ``mmbert-base/`` / ``mmbert-small/``.
        variant: "base" (0.31B, recommended) or "small" (0.14B). threshold: p(KEEP) cut-off (default from
        chopper_config.json). dtype: default bfloat16 on CUDA, float32 on CPU."""
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}, got {variant!r}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        d = _fetch(repo, VARIANTS[variant])
        tok = AutoTokenizer.from_pretrained(f"{d}/encoder")
        enc = AutoModel.from_pretrained(f"{d}/encoder", dtype=dtype).to(device).eval()
        heads = load_file(f"{d}/heads.safetensors")
        dec = json.load(open(f"{d}/chopper_config.json")).get("decode", {})
        thr = dec.get("pred_thr", 0.5) if threshold is None else threshold
        return cls(tok, enc, heads, device, threshold=thr, merge_gap=dec.get("merge_gap", 5),
                   min_run=dec.get("min_run", 4))

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
        for group in self._groups(head, segs):
            c, pa = self._forward(head, group, passages)
            chops += c; p_ans = max(p_ans, pa); n += 1
        if n == 0:                                                     # no passages: nothing to keep, search
            p_ans = self._forward(head, [], passages)[1]; n = 1
        chops.sort(key=lambda c: (c.passage_index, c.start))
        return ChopResult(chops, "ANSWER" if p_ans >= 0.5 else "SEARCH", p_ans, n, [c.text for c in chops])

    __call__ = run

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
    def _forward(self, head, group, passages):
        s, spans = head, []
        for (i, off, text, _) in group:
            s += self.sep
            spans.append((len(s), len(s) + len(text), i, off))
            s += text
        e = self.tokenizer(s, return_offsets_mapping=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt")
        if len(group) > 1 and e["input_ids"].shape[1] >= self.max_length:  # estimate was off: split the group
            m = len(group) // 2
            c1, p1 = self._forward(head, group[:m], passages)
            c2, p2 = self._forward(head, group[m:], passages)
            return c1 + c2, max(p1, p2)
        off = e.pop("offset_mapping")[0].tolist()
        h = self.encoder(**{k: v.to(self.device) for k, v in e.items()}).last_hidden_state[0].float()
        p = torch.sigmoid(h @ self.heads["keep.weight"].T + self.heads["keep.bias"])[:, 0].cpu().numpy()
        p_ans = float(torch.softmax(h[0] @ self.heads["decision.weight"].T + self.heads["decision.bias"], -1)[1])
        body = np.array([a >= len(head) and b > a for a, b in off])
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
