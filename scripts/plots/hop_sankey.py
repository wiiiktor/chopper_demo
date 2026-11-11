#!/usr/bin/env python3
"""Static Sankey funnels for the README (GitHub cannot run the artifact's JavaScript): one SVG per hop count, two sources stacked
(the multi-hop chopper vs Claude Opus 5.5 as the teacher), stages = each hop's paragraph kept -> answer in notes -> answered correctly
(Opus 5.5 answering from the notes). Data = hopstudy/funnel.json ("all" subset).
usage: hop_sankey.py <funnel.json> <out dir> [srcA=label] [srcB=label]"""
import json, os, sys
F, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SRC = [a.split("=", 1) for a in sys.argv[3:]] or [["mh0", "Multi-hop chopper (mmBERT, 0.31B)"], ["opusblind", "Claude Opus 5.5 as the note-taker (~10,000B, estimate)"]]
J = json.load(open(F)); D = J["all"] if "all" in J else J          # funnel.json (all/unknown) or qfunnel.json (flat)
INK, MUTE, FLOW, BAND, LOSS, LBAND, BG = "#1b211f", "#5d6863", "#2f7d6d", "#9fcdc2", "#c0533d", "#efc2b6", "#ffffff"


def ribbon(x0, a0, b0, x1, a1, b1):
    m = (x0 + x1) / 2
    return f"M{x0:.1f},{a0:.1f} C{m:.1f},{a0:.1f} {m:.1f},{a1:.1f} {x1:.1f},{a1:.1f} L{x1:.1f},{b1:.1f} C{m:.1f},{b1:.1f} {m:.1f},{b0:.1f} {x0:.1f},{b0:.1f} Z"


def sankey(d, k, y0, title, W=900):
    o = []; top = y0 + 64; sc = 120 / d["n"]; nw = 9; lossTop = top + 120 + 40
    stages = ["Questions"] + [f"Hop {j} kept" for j in range(1, k + 1)] + ["Answer in notes", "Answered correctly"]
    surv = [d["n"]]
    for j in range(k):
        surv.append(surv[-1] - d["lost"][j])
    surv += [d["ans_kept"], d["ans_ok"] or 0]; drops = d["lost"] + [d["ans_drop"], d["ans_kept"] - (d["ans_ok"] or 0)]
    if "tok" in d and d["tok"] > 50:
        title += f" · {d['tok']:,} tokens"
    S = len(stages); x = lambda i: 20 + i * (W - 20 - 120) / (S - 1)
    o.append(f'<text x="20" y="{y0 + 18}" font-size="15" font-weight="700" fill="{INK}">{title}</text>')
    for i in range(1, S):
        x0, x1, s, p, l = x(i - 1) + nw, x(i), surv[i], surv[i - 1], drops[i - 1]
        if s > 0:
            o.append(f'<path d="{ribbon(x0, top, top + s * sc, x1, top, top + s * sc)}" fill="{BAND}"/>')
        if l > 0:
            lh = max(l * sc * 0.6, 2)
            o.append(f'<path d="{ribbon(x0, top + s * sc, top + p * sc, x1, lossTop, lossTop + lh)}" fill="{LBAND}"/>')
            o.append(f'<rect x="{x1}" y="{lossTop}" width="{nw}" height="{lh:.1f}" fill="{LOSS}" rx="2"/>')
            lab = f"lost at hop {i}" if i <= k else ("answer cut" if i == k + 1 else "answered wrong")
            o.append(f'<text x="{x1 + nw + 4}" y="{lossTop + 11}" font-size="11.5" fill="{INK}">{lab}</text>')
            o.append(f'<text x="{x1 + nw + 4}" y="{lossTop + 24}" font-size="11" fill="{MUTE}">{l}%</text>')
    for i in range(S):
        h = max(surv[i] * sc, 1.5); anchor = "start" if i == 0 else "middle"
        o.append(f'<rect x="{x(i)}" y="{top}" width="{nw}" height="{h:.1f}" fill="{FLOW}" rx="2"/>')
        o.append(f'<text x="{x(i)}" y="{top - 22}" font-size="11.5" text-anchor="{anchor}" fill="{INK}">{stages[i]}</text>')
        lab = f"{surv[i]} questions" if i == 0 else (f"{d['ok_all']}% ({surv[i]}%)" if i == S - 1 and d.get("ok_all") is not None else f"{surv[i]}%")
        o.append(f'<text x="{x(i)}" y="{top - 8}" font-size="11" text-anchor="{anchor}" fill="{MUTE}">{lab}</text>')
    extra = (d.get("ok_all") or 0) - (d["ans_ok"] or 0)
    if extra > 0:                                   # correct from incomplete notes: stacked under the full-chain part, so the bar = the table
        ry = top + (d["ans_ok"] or 0) * sc + 2
        o.append(f'<rect x="{x(S - 1)}" y="{ry:.1f}" width="{nw}" height="{max(extra * sc, 1.5):.1f}" fill="{FLOW}" fill-opacity="0.55" rx="2"/>')
        o.append(f'<text x="{x(S - 1) + nw + 4}" y="{ry + 9:.1f}" font-size="11" fill="{MUTE}">+{extra} from</text>')
        o.append(f'<text x="{x(S - 1) + nw + 4}" y="{ry + 22:.1f}" font-size="11" fill="{MUTE}">incomplete notes</text>')
    return o, lossTop + 120 * 0.6 + 34


for k in (1, 2, 3, 4):
    body, y = [], 10
    for src, lab in SRC:
        o, y = sankey(D[src][str(k)], k, y, lab); body += o
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {y:.0f}" width="900" font-family="Helvetica, Arial, sans-serif">'
           f'<rect width="100%" height="100%" fill="{BG}"/>' + "".join(body) + "</svg>")
    open(f"{OUT}/{os.environ.get('PREFIX', 'hop_sankey')}_{k}hop.svg", "w").write(svg)
print("written", OUT)
