"""Chopper demo (CPU): cut the exact quotes that answer a question out of long text, and rank the passages, in one pass.
Model: huggingface.co/wiiiktor/chopper_mmbert (MIT). Code: github.com/wiiiktor/chopper_demo, pip install chopper-rag."""
import time

import gradio as gr
import torch

from chopper_rag.core import Chopper
from chopper_rag.mcp_server import split_passages

torch.set_num_threads(max(1, torch.get_num_threads()))
_models = {}


def model(variant):
    if variant not in _models:
        _models[variant] = Chopper.from_pretrained(variant=variant, device="cpu")
    return _models[variant]


def run(question, text, variant):
    if not question.strip() or not text.strip():
        return [], "Enter a question and some text.", []
    parts = split_passages(text, 500)          # ~paragraphs, so the ranking is visible
    offs = [o for o, _ in parts]; passages = [p for _, p in parts]
    t = time.time()
    r = model(variant).run(question, passages)
    dt = time.time() - t
    spans = sorted((offs[c.passage_index] + c.start, offs[c.passage_index] + c.end) for c in r.chops)
    hl, cur = [], 0
    for a, b in spans:
        if a > cur:
            hl.append((text[cur:a], None))
        hl.append((text[a:b], "chop")); cur = b
    if cur < len(text):
        hl.append((text[cur:], None))
    kept = sum(b - a for a, b in spans)
    verdict = ("ANSWER: the text seems to hold the answer" if r.decision == "ANSWER"
               else "SEARCH: the answer is probably not here, search again")
    summary = (f"**{verdict}** (p = {r.p_answer:.2f})  \n"
               f"Kept **{kept:,}** of {len(text):,} characters ({100 * kept / max(len(text), 1):.1f}%) in {len(spans)} chops, "
               f"{len(passages)} passages, {dt:.1f} s on a shared CPU (about 0.1 s on a GPU).")
    rank = [[i + 1, round(s, 3), passages[i][:120].replace("\n", " ") + ("..." if len(passages[i]) > 120 else "")]
            for i, s in r.ranking()]
    return hl, summary, rank


EX_EN = """Maria Skłodowska-Curie was born in Warsaw on 7 November 1867, the youngest of five children of Władysław Skłodowski and Bronisława Boguska. Her father taught mathematics and physics.

In 1891 she moved to Paris and enrolled at the Sorbonne, where she earned degrees in physics (1893) and mathematics (1894). There she met Pierre Curie, whom she married in 1895.

The Curies discovered two elements: polonium, named after her native Poland, and radium. In 1903 she shared the Nobel Prize in Physics with Pierre Curie and Henri Becquerel for their work on radioactivity.

In 1911 she received a second Nobel Prize, in Chemistry, for the discovery of polonium and radium and the isolation of radium. She remains the only person to win Nobel Prizes in two different sciences.

During the First World War she developed mobile X-ray units, known as "petites Curies", that served field hospitals near the front. She died in 1934 in Passy, France, of aplastic anaemia, probably caused by long exposure to radiation."""

EX_PL = """Wisła jest najdłuższą rzeką Polski. Jej długość wynosi 1047 km, a powierzchnia dorzecza około 194 tys. km², z czego większość leży w granicach kraju.

Rzeka bierze początek na zachodnim stoku Baraniej Góry w Beskidzie Śląskim, a uchodzi do Zatoki Gdańskiej. Przepływa przez Kraków, Warszawę, Toruń i Bydgoszcz (Fordon).

Największymi dopływami Wisły są Bug (prawy, przez Narew) oraz San. Wisła należy do zlewiska Morza Bałtyckiego."""

with gr.Blocks(title="Chopper") as demo:
    gr.Markdown(
        "# Chopper\n"
        "Paste a question and long text (search results, documents, notes). The chopper keeps only the **exact quotes** "
        "that answer it, typically 2-5% of the text, and ranks the passages, in **one pass** of a small open model "
        "(mmBERT, 0.3B, 8 languages + Polish versions). **MIT: commercial use allowed.**  \n"
        "[GitHub](https://github.com/wiiiktor/chopper_demo) · [model](https://huggingface.co/wiiiktor/chopper_mmbert) · "
        "`pip install chopper-rag` (LangChain, LlamaIndex, Haystack) · MCP server: `uvx chopper-rag`")
    with gr.Row():
        with gr.Column():
            q = gr.Textbox(label="Question", lines=1)
            txt = gr.Textbox(label="Text (passages separated by blank lines; long ones are split at ~500 characters)", lines=14)
            v = gr.Radio(["small", "base", "small-pl", "base-pl"], value="small", label="Model (small is faster on CPU; -pl for Polish)")
            b = gr.Button("Chop", variant="primary")
        with gr.Column():
            s = gr.Markdown()
            hl = gr.HighlightedText(label="Kept text (highlighted)", color_map={"chop": "#2f7d6d"}, combine_adjacent=True)
            rk = gr.Dataframe(headers=["passage", "relevance", "text"], label="Passages ranked by relevance (the chopper as a reranker)", wrap=True)
    gr.Examples([["Which two elements did the Curies discover?", EX_EN, "small"],
                 ["Who did Maria marry, and when?", EX_EN, "base"],
                 ["Gdzie uchodzi Wisła?", EX_PL, "base-pl"]], [q, txt, v])
    b.click(run, [q, txt, v], [hl, s, rk])
    q.submit(run, [q, txt, v], [hl, s, rk])

demo.queue(max_size=16).launch()
