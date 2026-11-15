---
title: Chopper with and without
emoji: 🔪
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Qwen3-14B on 2-hop questions, with and without Chopper
models:
- wiiiktor/chopper_mmbert
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Qwen3-14B answers 2-hop MuSiQue questions twice: from all 20 retrieved paragraphs, and from Chopper's exact quotes only
(about 12x fewer tokens). 20 precomputed examples (stratified to match 28% -> 48% on all 100 two-hop questions); for your
own question Chopper runs in your browser and Qwen3-14B answers through a small proxy. MIT, commercial use allowed.
Code: https://github.com/wiiiktor/chopper_demo
