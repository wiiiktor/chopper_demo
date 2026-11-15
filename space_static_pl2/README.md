---
title: Chopper PL
emoji: 🔪
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Qwen3-14B na polskich pytaniach, z i bez Choppera
models:
- wiiiktor/chopper_mmbert
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Qwen3-14B odpowiada na polskie pytania (PoQuAD, PolQA) dwa razy: z 16 znalezionych fragmentów i tylko z cytatów wyciętych przez
polski Chopper (ok. 15× mniej tokenów, ta sama trafność). 20 przykładów policzonych wcześniej; własne pytania na żywo.
