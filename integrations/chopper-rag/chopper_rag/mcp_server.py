"""Chopper as an MCP server: Claude (Desktop, Code, any MCP client) hands it a question plus retrieved text and gets
back only the exact quotes that answer it, plus an ANSWER / SEARCH decision.

Run:      chopper-mcp                      (stdio; installed by  pip install "chopper-rag[mcp]")
          python -m chopper_rag.mcp_server
Settings (environment variables):
    CHOPPER_VARIANT       base (default) | small | base-pl | small-pl
    CHOPPER_DEVICE        auto (default) | cpu | cuda
    CHOPPER_SNAP          sentence (default: whole sentences) | word (never cut inside a word) | none (raw)
    CHOPPER_REPO          Hugging Face repo or local directory (default wiiiktor/chopper_mmbert)
    CHOPPER_ALLOWED_DIRS  directories chop_files may read, separated by os.pathsep (default: the home directory)

Tools:
    chop_passages(question, passages)   chop a list of retrieved passages (search results, tool outputs, notes)
    chop_files(question, paths)         read local text files, split them into ~1,000-character passages, chop them
    rank_passages(question, passages)   the chopper as a reranker: passages ordered by relevance (same single pass)
Every chop result also carries `ranking` (passages by relevance), so one call both reranks and chops.
"""
from __future__ import annotations

import os
import re
import sys
import threading

try:                                    # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .core import Chopper

PASSAGE_CHARS = 1000                    # the model was trained on passages of ~1,000 characters
MAX_FILE_BYTES = 5_000_000

INSTRUCTIONS = (
    "Chopper cuts exact quotes that answer a question out of long retrieved text (2-5% of it is typically kept). "
    "Call it BEFORE reading many search results, documents or files in full: pass the user's question and the raw "
    "text, then read the returned quotes instead of the whole text. Quotes are verbatim, with their source and "
    "character offsets. decision == 'SEARCH' means the text most likely does not contain the answer: search again "
    "or retrieve other documents rather than answering from it. The same call ranks the passages by relevance "
    "(`ranking`), so no separate reranker is needed; use rank_passages when whole passages, not quotes, are wanted."
)

mcp = _Server("chopper", instructions=INSTRUCTIONS)
_model: Chopper | None = None
_lock = threading.Lock()


def _chopper() -> Chopper:
    global _model
    with _lock:
        if _model is None:
            kw = {"variant": os.environ.get("CHOPPER_VARIANT", "base"), "device": os.environ.get("CHOPPER_DEVICE", "auto"),
                  "snap": os.environ.get("CHOPPER_SNAP", "sentence")}
            if os.environ.get("CHOPPER_REPO"):
                kw["repo"] = os.environ["CHOPPER_REPO"]
            print(f"chopper-mcp: loading {kw}", file=sys.stderr)
            _model = Chopper.from_pretrained(**kw)
        return _model


def _result(question: str, passages: list[str], sources: list[tuple[str, int]]) -> dict:
    """run the chopper; sources[i] = (label, character offset of passage i inside its source)"""
    r = _chopper().run(question, passages)
    chops = [{"source": sources[c.passage_index][0], "start": sources[c.passage_index][1] + c.start,
              "end": sources[c.passage_index][1] + c.end, "text": c.text,
              "score": None if c.score is None else round(float(c.score), 3)} for c in r.chops]
    n_in = sum(len(p) for p in passages)
    n_out = sum(len(c["text"]) for c in chops)
    context = "\n".join(f"[{c['source']} @{c['start']}] {c['text']}" for c in chops)
    ranking = [{"passage": i, "source": sources[i][0], "start": sources[i][1], "relevance": round(float(v), 3)}
               for i, v in r.ranking(20)]
    return {"decision": r.decision, "p_answer": round(float(r.p_answer), 3), "chops": chops, "context": context,
            "ranking": ranking,
            "stats": {"passages": len(passages), "input_chars": n_in, "kept_chars": n_out,
                      "kept_share": round(n_out / n_in, 4) if n_in else 0.0, "forward_passes": r.n_passes}}


@mcp.tool()
def chop_passages(question: str, passages: list[str]) -> dict:
    """Cut the exact quotes that answer `question` out of `passages` (retrieved snippets, search results, tool
    outputs). Returns the quotes (verbatim, with passage number and character offsets), a ready-to-read `context`
    string, and decision ANSWER (the passages hold the answer) or SEARCH (search again)."""
    passages = [str(p) for p in passages]
    return _result(question, passages, [(f"passage {i}", 0) for i in range(len(passages))])


@mcp.tool()
def rank_passages(question: str, passages: list[str], top_k: int = 5) -> dict:
    """Rank `passages` by relevance to `question` (the chopper used as a reranker: on a held-out 8-language set it
    ranks within ~1 nDCG point of bge-reranker-v2-m3 at the same speed). Returns the top_k passages, most relevant
    first, with their index and relevance (0..1), plus decision ANSWER / SEARCH."""
    passages = [str(p) for p in passages]
    r = _chopper().run(question, passages)
    return {"decision": r.decision, "p_answer": round(float(r.p_answer), 3),
            "ranking": [{"passage": i, "relevance": round(float(v), 3), "text": passages[i]}
                        for i, v in r.ranking(top_k)]}


def _allowed(path: str) -> bool:
    roots = os.environ.get("CHOPPER_ALLOWED_DIRS") or os.path.expanduser("~")
    real = os.path.realpath(path)
    return any(real == r or real.startswith(r.rstrip(os.sep) + os.sep)
               for r in (os.path.realpath(os.path.expanduser(x)) for x in roots.split(os.pathsep) if x))


def split_passages(text: str, size: int = PASSAGE_CHARS) -> list[tuple[int, str]]:
    """(offset, passage): paragraphs packed up to ~size characters; one longer paragraph is cut at sentence or word ends"""
    out, start, end = [], None, None
    for m in re.finditer(r"\S(?:.*?\S)?(?=\s*\n\s*\n|\s*$)", text, flags=re.S):
        a, b = m.start(), m.end()
        while b - a > size:                                     # long paragraph: cut near size at '. ' or ' '
            cut = text.rfind(". ", a, a + size)
            cut = cut + 1 if cut > a + size // 2 else (text.rfind(" ", a, a + size) if text.rfind(" ", a, a + size) > a else a + size)
            if start is not None:
                out.append((start, text[start:end])); start = None
            out.append((a, text[a:cut]))
            a = cut
            while a < b and text[a].isspace():
                a += 1
        if start is not None and b - start > size:
            out.append((start, text[start:end])); start = None
        if start is None:
            start = a
        end = b
    if start is not None:
        out.append((start, text[start:end]))
    return [(o, p) for o, p in out if p.strip()]


@mcp.tool()
def chop_files(question: str, paths: list[str]) -> dict:
    """Read local text files (notes, docs, logs, code, exported pages), split them into ~1,000-character passages and
    cut out the exact quotes that answer `question`. Each quote carries its file path and character offset in that
    file. Use it instead of reading long files in full when only part of them is relevant."""
    passages, sources, skipped = [], [], []
    for p in paths:
        path = os.path.expanduser(p)
        if not _allowed(path):
            skipped.append({"path": p, "reason": "outside CHOPPER_ALLOWED_DIRS"}); continue
        if not os.path.isfile(path):
            skipped.append({"path": p, "reason": "not a file"}); continue
        if os.path.getsize(path) > MAX_FILE_BYTES:
            skipped.append({"path": p, "reason": f"larger than {MAX_FILE_BYTES} bytes"}); continue
        try:
            text = open(path, encoding="utf-8").read()
        except UnicodeDecodeError:
            skipped.append({"path": p, "reason": "not UTF-8 text"}); continue
        for off, s in split_passages(text):
            passages.append(s); sources.append((p, off))
    res = _result(question, passages, sources)
    res["skipped"] = skipped
    return res


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
