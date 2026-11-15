---
title: Chopper Playground
emoji: 🔪
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Exact quotes that answer a question, in your browser
models:
- wiiiktor/chopper_mmbert
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Chopper keeps only the exact quotes that answer a question (typically 2-5% of the text) and ranks the passages, in one
pass of a small open model. This page runs the `small` chopper (int8 ONNX) in your browser with onnxruntime-web:
nothing is sent to a server. MIT, commercial use allowed. Code: https://github.com/wiiiktor/chopper_demo
