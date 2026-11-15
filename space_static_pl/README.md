---
title: Chopper PL
emoji: 🔪
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Dokładne cytaty odpowiadające na pytanie, w przeglądarce
models:
- wiiiktor/chopper_mmbert
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Chopper zostawia tylko dokładne cytaty, które odpowiadają na pytanie (zwykle 2-5% tekstu), i porządkuje fragmenty według
trafności, w jednym przebiegu małego otwartego modelu. Ta strona uruchamia polski model `small-pl` (ONNX) w przeglądarce:
nic nie jest wysyłane na serwer. MIT, także do użytku komercyjnego. Kod: https://github.com/wiiiktor/chopper_demo
