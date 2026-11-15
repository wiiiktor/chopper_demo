---
title: Chopper
emoji: 🔪
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Exact quotes that answer a question + passage ranking
models:
- wiiiktor/chopper_mmbert
---

Chopper keeps only the exact quotes that answer a question (typically 2-5% of the text) and ranks the passages, in one
pass of a small open model. MIT, commercial use allowed. Code: https://github.com/wiiiktor/chopper_demo
