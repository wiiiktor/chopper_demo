#!/usr/bin/env python3
"""Static SVGs for the README that mirror the results page: (1) the size cube corner (mmBERT one dot vs a 32x32x32 cube of dots
~ Opus 5.5 at a rough 10,000B), (2) the Qwen3-14B notes table with mini bars.
usage: readme_static.py <out dir>"""
import math, os, sys
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
INK, MUTE, FLOW, LOSS, LINE, BG = "#1b211f", "#5d6863", "#2f7d6d", "#c0533d", "#d9ddd6", "#ffffff"
FONT = 'font-family="Helvetica, Arial, sans-serif"'

# 1. cube corner
W, H, n, st, c30 = 900, 300, 32, 20, math.cos(math.pi / 6); cx0, cy0 = 34, 112
P = lambda x, y, z: (cx0 + (x - (z - (n - 1))) * c30 * st, cy0 + (x + (z - (n - 1))) * 0.5 * st - (y - (n - 1)) * st)
o = [f'<rect width="100%" height="100%" fill="{BG}"/>']
for a in range(n):
    for b in range(n):
        for (x, y, z) in ((a, n - 1, b), (a, b, n - 1)):
            px, py = P(x, y, z)
            if -5 <= px <= W + 5 and -5 <= py <= H + 5:
                d = math.hypot(px - cx0, py - cy0) / math.hypot(W, H); op = max(0.12, 0.9 - 1.1 * d)
                o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{FLOW}" fill-opacity="{op:.2f}"/>')
qx, qy = P(0, n - 1, n - 1)
o += [f'<circle cx="{qx:.1f}" cy="{qy:.1f}" r="7" fill="{LOSS}"/>', f'<circle cx="{qx:.1f}" cy="{qy:.1f}" r="13" fill="none" stroke="{LOSS}" stroke-width="1.5"/>',
      f'<line x1="{qx:.1f}" y1="{qy - 14:.1f}" x2="{qx:.1f}" y2="30" stroke="{LOSS}" stroke-width="1.5"/>',
      f'<text x="{qx + 8:.1f}" y="22" font-size="15" font-weight="700" fill="{INK}">mmBERT: this one dot, 0.31B</text>',
      f'<rect x="{W - 640}" y="{H - 34}" width="630" height="28" rx="4" fill="{BG}" fill-opacity="0.92"/>',
      f'<text x="{W - 16}" y="{H - 14}" font-size="14" text-anchor="end" fill="{MUTE}">… one corner of a 32 × 32 × 32 cube of such dots ≈ Opus 5.5 (~10,000B, a rough estimate)</text>']
open(f"{OUT}/size_cube.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" {FONT}>' + "".join(o) + "</svg>")

# 2. Qwen3-14B notes table with mini bars
# (all correct, complete-notes-only correct in brackets; None = notes are always complete)
ROWS = [("All notes retrieved from database", 2620, [(84, None), (28, None), (22, None), (12, None)]),
        ("Chopper notes (mmBERT 0.31B)", 228, [(83, 82), (48, 44), (45, 35), (29, 21)]),
        ("Opus 5.5's notes", 144, [(90, 90), (64, 60), (55, 40), (43, 30)]),
        ("Gold notes (each hop's gold sentences)", 217, [(86, None), (63, None), (52, None), (37, None)])]
COLS = ["Qwen3-14B answering from", "notes length", "1-hop", "2-hop", "3-hop", "4-hop"]
X = [16, 330, 460, 570, 680, 790]; RH = 46; W2 = 900; H2 = 50 + RH * len(ROWS) + 10
o = [f'<rect width="100%" height="100%" fill="{BG}"/>']
for i, (x, c) in enumerate(zip(X, COLS)):
    xa, an = (x, "start") if i == 0 else (x + 90, "end")
    o.append(f'<text x="{xa}" y="30" font-size="12" font-weight="700" text-anchor="{an}" fill="{MUTE}" letter-spacing=".5">{c.upper()}</text>')
o.append(f'<line x1="10" y1="42" x2="{W2 - 10}" y2="42" stroke="{LINE}"/>')
for i, (name, tok, v) in enumerate(ROWS):
    y = 50 + i * RH
    o.append(f'<text x="{X[0]}" y="{y + 28}" font-size="14" fill="{INK}">{name.replace(chr(39), "&#39;")}</text>')
    for j, (val, mx, lab) in enumerate([(tok, 2620, f"{tok:,}")] + [(a, 100, f"{a} ({b})" if b is not None else str(a)) for a, b in v]):
        x = X[j + 1]; w = 90
        o.append(f'<text x="{x + w}" y="{y + 24}" font-size="14" text-anchor="end" fill="{INK}" font-family="Menlo, Consolas, monospace">{lab}</text>')
        o.append(f'<rect x="{x}" y="{y + 31}" width="{w}" height="5" rx="2" fill="{LINE}"/>')
        o.append(f'<rect x="{x}" y="{y + 31}" width="{max(w * val / mx, 2):.1f}" height="5" rx="2" fill="{FLOW}"/>')
    o.append(f'<line x1="10" y1="{y + RH}" x2="{W2 - 10}" y2="{y + RH}" stroke="{LINE}"/>')
open(f"{OUT}/qwen_notes_table.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}" width="{W2}" {FONT}>' + "".join(o) + "</svg>")
print("written")
