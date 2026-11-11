# Chopper with a few iterations

[← back to the README](../README.md)

The main results page uses **one database retrieval per question**, even at 4 hops: the chopper cuts the 20 retrieved
paragraphs once and Qwen3-14B answers once. This page covers the iterative version:

1. The chopper cuts the question's 20 paragraphs into notes.
2. Qwen3-14B (4-bit, no fine-tuning, prompt only) reads the notes and either answers or writes **one** short follow-up
   question about the first missing link in the chain (for example, *"Who is Steve Hillage's spouse?"*).
3. The chopper cuts the same 20 paragraphs again with that follow-up question; the new notes are added to the old ones.
4. Up to 3 follow-ups, then Qwen3-14B must answer. gpt-5.6-luna judges the answer.

The same loop runs with Claude Opus 5.5 (answer-blind) as the note-taker, for comparison. The 20 paragraphs already contain
every fact each hop needs, so this test measures how far follow-up questions let each note-taker chain hops, without retrieval
in the way. Searching the whole database with each follow-up is a separate test.

## Results

*Pending: the run is queued for 2026-11-11 evening; the table and diagrams go here.*

| note-taker | follow-ups | 1-hop | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|---|
| multi-hop chopper | 0 (one cut, as on the README) | | | | |
| multi-hop chopper | up to 3 | | | | |
| Opus 5.5 | 0 | | | | |
| Opus 5.5 | up to 3 | | | | |

Script: [`scripts/hopstudy/iter_loop.py`](../scripts/hopstudy/iter_loop.py).
