#!/usr/bin/env python
"""HTML page of the chopper / answerer training curves (2026-10-30, user: "update charts on an artifact").
Reads release/figures/curves.json (written by plot_curves.py) and writes a self-contained page with the data inlined.
usage: python curves_page.py <out.html>"""
import json, os, sys, datetime
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "curves.json")))
_rp = os.environ.get("RR_REPORT", "/media/wii/bigdata/rr/pilot_report.json")
D["pilot"] = json.load(open(_rp)) if os.path.exists(_rp) else None
D["answers"] = {}
for _w in ("old", "new", "full", "fullc"):
    _f = f"/media/wii/bigdata/rr/ans_{_w}/judged.jsonl"
    if os.path.exists(_f):
        D["answers"][_w] = {x["dataset"]: x["judged"] for x in map(json.loads, open(_f))}
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
html = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chopper Training Curves</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
:root{--bg:#fcfcfb;--panel:#ffffff;--fg:#0b0b0b;--fg2:#52514e;--fg3:#7a7974;--grid:#e9e8e4;--line:#d6d5d0;
 --c0:#2a78d6;--c1:#eb6834;--c2:#1baf7a;--c3:#c2449f;--ca:#6250d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222220;--fg:#f0efec;--fg2:#c3c2b7;--fg3:#8f8e86;--grid:#2f2e2b;--line:#45443f;
 --c0:#3987e5;--c1:#d95926;--c2:#199e70;--c3:#d55fb3;--ca:#9085e9}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222220;--fg:#f0efec;--fg2:#c3c2b7;--fg3:#8f8e86;--grid:#2f2e2b;--line:#45443f;
 --c0:#3987e5;--c1:#d95926;--c2:#199e70;--c3:#d55fb3;--ca:#9085e9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif}
main{max-width:1640px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:24px;margin:0 0 4px;text-wrap:balance}h2{font-size:15px;margin:28px 0 10px;color:var(--fg2);letter-spacing:.04em;text-transform:uppercase}
.sub{color:var(--fg2);max-width:70ch;margin:0 0 14px}.meta{font:12px "IBM Plex Mono",monospace;color:var(--fg3)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:14px 0 4px;font-size:13px;color:var(--fg2)}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:18px;height:3px;border-radius:2px;display:inline-block}
.sw.dash{background:none!important;border-top:2px dashed;height:0}
#g3{display:grid;gap:16px}#g3 .card{max-width:760px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,600px),1fr));gap:16px}
.zoombar{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;margin:10px 0 4px;font-size:13px;color:var(--fg2)}
.zoombar button{font:13px "IBM Plex Mono",monospace;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 10px;cursor:pointer}
.zoombar button:hover{border-color:var(--fg3)}.zoombar button[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.zoombar button:focus-visible,.zoombar input:focus-visible{outline:2px solid var(--c0);outline-offset:2px}
.zoombar .hint{color:var(--fg3)}.zoombar label{display:inline-flex;align-items:center;gap:5px;cursor:pointer}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 12px 6px;position:relative}
.card h3{margin:0;font-size:14px}.card p{margin:2px 0 6px;font-size:12px;color:var(--fg3)}
svg{width:100%;height:auto;display:block;overflow:visible;cursor:crosshair;touch-action:pan-y;user-select:none}
svg text{font:12px "IBM Plex Mono",monospace;fill:var(--fg3)}
.tip{position:absolute;pointer-events:none;background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:6px 8px;
 font:12px "IBM Plex Mono",monospace;color:var(--fg);white-space:nowrap;display:none;z-index:2;box-shadow:0 2px 8px rgba(0,0,0,.12)}
.tip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.empty{font-size:12px;fill:var(--fg3)}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 14px;min-width:190px;flex:1 1 190px}
.stat b{display:block;font:600 24px "IBM Plex Mono",monospace;color:var(--fg)}.stat span{font-size:12px;color:var(--fg2)}
.hw{overflow-x:auto}table.heat{border-collapse:separate;border-spacing:2px;font:13px "IBM Plex Mono",monospace;width:100%;min-width:560px}
.heat th{font-weight:400;color:var(--fg2);text-align:center;padding:4px 6px}.heat th.r{text-align:left;white-space:nowrap}
.heat td{text-align:center;padding:8px 6px;border-radius:3px}
.lg{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--fg2);margin:0 0 4px}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.explain{max-width:78ch;color:var(--fg2);margin:0 0 12px}.explain b{color:var(--fg)}
.desc{margin:10px 0 0;max-width:72ch;color:var(--fg2);line-height:1.5}
</style></head><body><main>
<h1>Chopper training curves</h1>
<p class="sub">The choppers learn which tokens of 16 retrieved passages to keep; the answerer (MiniCPM3-4B LoRA) learns to answer
or ask a follow-up question. The two mmBERTs and the MiniCPM3-4B slice (MLA, multi-head latent attention: every slice head reads the
full model's whole conversation cache) are trained once on all 54k rows mixed (benchmark + Wikipedia); the fourth line continues mmBERT-base on the 8-language rerank pilot, whose labels are whole sentences, so its training numbers are not comparable with the short-span rows before it. The answerer has two arms
with the same targets, trained on the integrated chopper's fragments or on the Qwen3-14B labeller's; both are scored on the chopper's
validation fragments.
Train lines are 500-row rolling means; validation points are the mean over MuSiQue, HotpotQA and TriviaQA (300 questions), read at p&nbsp;&gt;&nbsp;0.5.</p>
<div class="meta">data from the train logs · regenerated __NOW__ Warsaw</div>
<div class="legend" id="legend"></div>
<div class="zoombar" id="zoombar">Zoom:
<button data-r="0,999999">all</button><button data-r="0,27000">0–27k</button><button data-r="27000,55000">27k–54k</button><button data-r="54000,94000">rerank pilot</button>
<label><input type="checkbox" id="fity" checked> fit y to the view</label>
<span class="hint">· or drag across any chart; double-click a chart to reset</span></div>
<div id="pilot"></div>
<h2>Chopper training curves</h2><div class="grid" id="g1"></div>
<h2>Answerer</h2><div id="g3"></div>
</main>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NAMES = Object.keys(DATA.choppers), COLS = ["--c0", "--c1", "--c2", "--c3"];
const REFS = Object.keys(DATA.reference);   // dashed, in the colour of the solid run with the same model
function smooth(a, k = 5) { return a.map((_, i) => { const w = a.slice(Math.max(0, i - k + 1), i + 1).filter(v => v != null); return w.length ? w.reduce((s, v) => s + v, 0) / w.length : null; }); }
function seriesFor(run, part, key) {
  const pts = (run[part] || []).filter(p => p[key] != null);
  const ys = part === "train" ? smooth(pts.map(p => p[key])) : pts.map(p => p[key]);
  return pts.map((p, i) => ({ x: p.rows, y: ys[i], per: p.per }));
}
const lg = document.getElementById("legend");
const LORDER = NAMES.map((n, i) => i).sort((a, b) => (/mmBERT-base/.test(NAMES[b]) - /mmBERT-base/.test(NAMES[a])) || (/mmBERT/.test(NAMES[b]) - /mmBERT/.test(NAMES[a])));   // legend: mmBERT-base, mmBERT-small, then the slices
LORDER.forEach(i => lg.insertAdjacentHTML("beforeend", `<span><i class="sw" style="background:var(${COLS[i]})"></i>${NAMES[i]}</span>`));
REFS.forEach(n => { if ((DATA.reference[n].train || []).length) lg.insertAdjacentHTML("beforeend", `<span><i class="sw dash" style="border-color:var(${COLS[DATA.reference[n].ci]})"></i>${n}</span>`); });
const ANS = Object.keys(DATA.answerer);
const W = 600, H = 330, M = { l: 50, r: 16, t: 10, b: 30 }, XMAX = Math.max(55000, Math.ceil(Math.max(...[DATA.choppers, DATA.reference].flatMap(o => Object.values(o)).flatMap(r => [...(r.train || []), ...(r.val || [])]).map(p => p.rows)) / 9000) * 9000);   // past 54k = a second epoch
window.ZXG = [0, XMAX];                                   // shared x zoom, rows seen
function setZoom(a, b) { window.ZXG = [Math.max(0, a), Math.min(XMAX, b)]; render(); }
document.querySelectorAll("#zoombar button").forEach(b => b.addEventListener("click", () => setZoom(...b.dataset.r.split(",").map(Number))));
document.getElementById("fity").addEventListener("change", () => render());
function xticks(ZX) {
  const span = ZX[1] - ZX[0], step = [250, 500, 1000, 2000, 2500, 5000, 9000, 18000].find(s => span / s <= 7) || 18000, out = [];
  for (let x = Math.ceil(ZX[0] / step) * step; x <= ZX[1]; x += step) out.push(x);
  return out;
}
function panel(host, title, note, series, yr, fmt, xr) {
  const ZX = xr || window.ZXG, wide = !!xr;              // xr: a fixed x range (no zoom), drawn full width
  const card = document.createElement("div"); card.className = "card";
  card.innerHTML = `<h3>${title}</h3><p>${note}</p><div class="tip"></div>`; host.appendChild(card);
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", title); card.insertBefore(svg, card.querySelector(".tip"));
  const all = series.flatMap(s => s.pts.map(p => p.y)), zoomed = !wide && (ZX[0] > 0 || ZX[1] < XMAX);
  const vis = series.flatMap(s => s.pts.filter(p => p.x >= ZX[0] && p.x <= ZX[1]).map(p => p.y));
  const fit = zoomed && document.getElementById("fity").checked && vis.length;
  let [y0, y1] = fit ? [Math.min(...vis), Math.max(...vis)] : yr || [Math.min(...all), Math.max(...all)];
  if ((fit || !yr) && all.length) { const pad = (y1 - y0) * 0.08 || 0.02; y0 -= pad; y1 += pad; }
  const PW = W - M.l - M.r, X = x => M.l + (x - ZX[0]) / (ZX[1] - ZX[0]) * PW, Y = y => H - M.b - (y - y0) / (y1 - y0) * (H - M.t - M.b);
  const cid = "c" + Math.random().toString(36).slice(2);
  let g = "";
  for (let i = 0; i <= 4; i++) { const v = y0 + (y1 - y0) * i / 4; g += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css("--grid")}"/><text x="${M.l - 4}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}</text>`; }
  for (const x of xticks(ZX)) g += `<text x="${X(x)}" y="${H - 8}" text-anchor="middle">${+(x / 1000).toFixed(2)}k</text>`;
  g += `<defs><clipPath id="${cid}"><rect x="${M.l}" y="${M.t - 6}" width="${PW}" height="${H - M.t - M.b + 12}"/></clipPath></defs><g clip-path="url(#${cid})">`;
  if (!all.length) g += `<text class="empty" x="${W / 2}" y="${H / 2}" text-anchor="middle">logged from the next runs on</text>`;
  // a run that logged this metric only from a later point: hatch the unlogged span, open circle where logging starts
  const gaps = series.filter(s => s.pts.length && s.from != null && s.pts[0].x - s.from > 1000);
  gaps.forEach((s, k) => {
    const pid = "h" + Math.random().toString(36).slice(2), x0 = Math.max(M.l, X(s.from)), x1 = Math.min(W - M.r, X(s.pts[0].x));
    if (x1 <= x0) return;
    g += `<defs><pattern id="${pid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="${s.col}" stroke-width="1.2" opacity=".35"/></pattern></defs>`;
    g += `<rect x="${x0}" y="${M.t}" width="${x1 - x0}" height="${H - M.t - M.b}" fill="url(#${pid})"/>`;
    g += `<text x="${x0 + 6}" y="${H - M.b - 22 - 26 * k}" style="fill:var(--fg2)">${s.name.split(" (")[0].replace(/^10% /, "")}:</text><text x="${x0 + 6}" y="${H - M.b - 9 - 26 * k}" style="fill:var(--fg2)">not logged</text>`;
  });
  for (const s of series) {
    if (!s.pts.length) continue;
    const d = s.pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="${s.col}" stroke-width="${s.lw || (s.dash ? 1.3 : 2)}" ${s.dash ? `stroke-dasharray="${typeof s.dash === "string" ? s.dash : "4 3"}"` : ""} stroke-linejoin="round"/>`;
    if (s.end) g += `<text x="${X(s.pts.at(-1).x) + 6}" y="${Y(s.pts.at(-1).y) + 4}" style="fill:var(--fg2)">${s.end}</text>`;
    if (s.dots) for (const p of s.pts) g += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4" fill="${s.col}" stroke="${css("--panel")}" stroke-width="2"/>`;
    if (gaps.includes(s)) g += `<circle cx="${X(s.pts[0].x)}" cy="${Y(s.pts[0].y)}" r="4" fill="${css("--panel")}" stroke="${s.col}" stroke-width="2"/>`;
  }
  g += `</g>`;
  const missing = series.filter(s => !s.pts.length && s.from != null).map(s => s.name.split(/ \(|,/)[0]).filter((n, i, A) => A.indexOf(n) === i);
  if (missing.length && all.length) card.querySelector("p").insertAdjacentHTML("beforeend", ` · <i>not logged by ${missing.join(", ")} (older trainer)</i>`);
  g += `<rect class="brush" y="${M.t}" height="${H - M.t - M.b}" fill="${css("--fg3")}" opacity=".18" visibility="hidden"/>`;
  g += `<line class="xh" y1="${M.t}" y2="${H - M.b}" stroke="${css("--fg2")}" stroke-width="1" visibility="hidden"/>`;
  svg.innerHTML = g;
  const tip = card.querySelector(".tip"), xh = svg.querySelector(".xh"), br = svg.querySelector(".brush");
  const toX = e => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) / r.width * W; };
  const toRows = sx => ZX[0] + (sx - M.l) / PW * (ZX[1] - ZX[0]);
  let drag = null;                                         // drag across the plot = zoom every chart to that span
  svg.addEventListener("pointerdown", e => { if (e.button) return; drag = Math.min(Math.max(toX(e), M.l), W - M.r); svg.setPointerCapture(e.pointerId); });
  svg.addEventListener("pointerup", e => {
    if (drag == null) return; const a = drag, b = Math.min(Math.max(toX(e), M.l), W - M.r); drag = null; br.setAttribute("visibility", "hidden");
    if (Math.abs(b - a) > 8) setZoom(Math.round(toRows(Math.min(a, b))), Math.round(toRows(Math.max(a, b))));
  });
  if (wide) { card.classList.add("wide"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); }
  svg.addEventListener("dblclick", () => setZoom(0, XMAX));
  svg.addEventListener("pointermove", e => {
    if (drag != null) { const b = Math.min(Math.max(toX(e), M.l), W - M.r); br.setAttribute("x", Math.min(drag, b)); br.setAttribute("width", Math.abs(b - drag)); br.setAttribute("visibility", "visible"); }
    const sx = toX(e), tol = (ZX[1] - ZX[0]) / XMAX;
    const xv = toRows(sx); if (xv < ZX[0] || xv > ZX[1]) { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); return; }
    let rows = [], snap = null;
    for (const s of series) {
      if (!s.pts.length) continue;
      const p = s.pts.reduce((a, b) => Math.abs(b.x - xv) < Math.abs(a.x - xv) ? b : a);
      if (Math.abs(p.x - xv) > (s.dots ? 1500 : 600) * Math.max(tol, 0.1)) continue;
      snap = snap ?? p.x;
      const per = p.per ? ` <span style="color:var(--fg3)">(${Object.values(p.per).map(v => v.toFixed(2)).join(" / ")})</span>` : "";
      rows.push(`<div><i style="background:${s.col}"></i>${s.name}: <b>${fmt(p.y)}</b>${per}</div>`);
    }
    if (!rows.length) { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); return; }
    xh.setAttribute("x1", X(snap)); xh.setAttribute("x2", X(snap)); xh.setAttribute("visibility", "visible");
    tip.innerHTML = `<div style="color:var(--fg3)">${snap.toLocaleString()} rows</div>` + rows.join("");
    tip.style.display = "block";
    const cr = card.getBoundingClientRect(), left = e.clientX - cr.left + 12;
    tip.style.left = Math.min(left, cr.width - tip.offsetWidth - 4) + "px"; tip.style.top = (e.clientY - cr.top + 12) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); });
}
function lgd(S) { return `<div class="lg">${S.map(x => `<span><i style="background:${x.col}"></i>${x.name}</span>`).join("")}</div>`; }
function card(host, title, note, body, S, desc) {
  host.insertAdjacentHTML("beforeend", `<div class="card"><h3>${title}</h3><p>${note}</p>${S ? lgd(S) : ""}${body}${desc ? `<p class="desc" style="font-size:13px;margin-bottom:6px">${desc}</p>` : ""}</div>`);
}
function bars(host, title, note, cats, S, yMax, desc) {
  const W = 600, H = 230, L = 34, R = 8, T = 18, B = 34, y = v => T + (H - T - B) * (1 - v / yMax), gw = (W - L - R) / cats.length, bw = Math.min(26, (gw - 10 - 2 * (S.length - 1)) / S.length);
  let g = "";
  for (let t = 0; t <= 4; t++) { const v = yMax * t / 4; g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(2).replace(/^0/, "")}</text>`; }
  cats.forEach((c, i) => {
    const x0 = L + gw * i + (gw - S.length * bw - 2 * (S.length - 1)) / 2;
    S.forEach((s, j) => { const v = s.vals[i]; if (v == null) return; const x = x0 + j * (bw + 2), h = y(0) - y(v);
      g += `<path d="M${x},${y(0)}v${-Math.max(h - 3, 0)}q0,-3 3,-3h${bw - 6}q3,0 3,3v${Math.max(h - 3, 0)}z" fill="${s.col}"><title>${c} · ${s.name}: ${v.toFixed(3)}</title></path>`;
      if (cats.length <= 6 || s.lab) g += `<text x="${x + bw / 2}" y="${y(v) - 4}" text-anchor="middle" style="fill:var(--fg2);font-size:11px">${v.toFixed(2).replace(/^0/, "")}</text>`; });
    g += `<text x="${L + gw * (i + .5)}" y="${H - 14}" text-anchor="middle" style="fill:var(--fg2)">${c}</text>`;
  });
  card(host, title, note, `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}" style="cursor:default">${g}</svg>`, S, desc);
}
function lines(host, title, note, S, xl, yl, xr, yr, desc) {
  const W = 600, H = 276, L = 42, R = 14, T = 28, B = 40, X = v => L + (W - L - R) * (v - xr[0]) / (xr[1] - xr[0]), Y = v => T + (H - T - B) * (1 - (v - yr[0]) / (yr[1] - yr[0]));
  let g = "";
  for (let t = 0; t <= 4; t++) { const v = yr[0] + (yr[1] - yr[0]) * t / 4, u = xr[0] + (xr[1] - xr[0]) * t / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2).replace(/^0/, "")}</text><text x="${X(u)}" y="${H - 22}" text-anchor="middle">${xl.fmt(u)}</text>`; }
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle" style="fill:var(--fg2)">${xl.t}</text><text x="${L}" y="14" style="fill:var(--fg2)">${yl}</text>`;
  for (const s of S) { g += `<polyline fill="none" stroke="${s.col}" stroke-width="2" points="${s.pts.map(p => `${X(p[0])},${Y(p[1])}`).join(" ")}"/>`;
    for (const p of s.pts) { g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4.5" fill="${s.col}" stroke="var(--panel)" stroke-width="2"><title>${s.name}: ${p[2] || ""} ${xl.fmt(p[0])} → ${p[1].toFixed(3)}</title></circle>`; if (p[2]) g += `<text x="${X(p[0]) + 7}" y="${Y(p[1]) - 7}" style="font-size:11px;fill:var(--fg2)">${p[2]}</text>`; } }
  card(host, title, note, `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}" style="cursor:default">${g}</svg>`, S, desc);
}
function renderPilot() {
  const P = DATA.pilot, root = document.getElementById("pilot"); root.innerHTML = ""; if (!P) return;
  const c0 = css("--c0"), c1 = css("--c1"), c2 = css("--c2"), c3 = css("--c3"), OLD = "old 54k chopper", NEW = "pilot chopper";
  const sec = (w, a, k) => P.holdout[w][a].sec[k], A = "rrval_notag", T = "rrval_tag";
  root.insertAdjacentHTML("beforeend", `<h2>Rerank pilot: did the new data help?</h2>
  <p class="explain">Both choppers cut 16 passages down to what answers the query. <b>Precision</b> = how much of what it keeps is really answer material. <b>Recall</b> = how much of the answer material it keeps. <b>F1</b> combines the two. Scored on 2,000 rows it never trained on, 250 per language, passage text only, at the release settings (p &gt; .8).</p>
  <div class="stats">
   <div class="stat"><b>${sec("old", A, "all")[0].toFixed(2).replace(/^0/, "")} → ${sec("new", A, "all")[0].toFixed(2).replace(/^0/, "")}</b><span>F1 on the new 8-language rows, no tags</span></div>
   <div class="stat"><b>${sec("old", A, "all")[2].toFixed(2).replace(/^0/, "")} → ${sec("new", A, "all")[2].toFixed(2).replace(/^0/, "")}</b><span>recall: nearly all of the gain</span></div>
   <div class="stat"><b>${P.holdout.new[A].noans[1]} of ${P.holdout.new[A].noans[0]}</b><span>no-answer rows where it still keeps text (problem)</span></div>
   <div class="stat"><b>~$44</b><span>judging spend for the whole pilot</span></div></div><div class="grid" id="g4"></div>
  <h2>Window test: can a normal transformer run on a fixed-size memory?</h2>
  <p class="explain">A transformer normally remembers every token it has read, so its memory grows forever. The test cuts that memory: anything older than the window is deleted. We hide a fact (a 7-digit number) in 32,000 tokens of web text at a set distance from the end, then ask for it. <b>1.00 = found every time, 0 = never.</b> Qwen3-4B, no training.</p>
  <div class="hw card" id="g5"></div>
  <p class="explain" style="margin-top:10px"><b>How to read it.</b> Blue up to the window's size, then zero: a hard window is perfect for what it still holds and blind to the rest. "No pin" also deletes the very start of the prompt, and the model stops working entirely, so those first tokens must always stay. "Averaged tiers" keeps older text blurred instead of deleting it, using the same memory as the 8K window: it loses exact numbers much sooner and reaches no further, so plain averaging is not worth it. Training that blur is the next idea.</p>`);
  const g4 = document.getElementById("g4");
  bars(g4, "Precision, recall, F1", "new 8-language holdout, no tags", ["precision", "recall", "F1"], [{ name: OLD, col: c0, vals: [sec("old", A, "all")[1], sec("old", A, "all")[2], sec("old", A, "all")[0]] }, { name: NEW, col: c1, vals: [sec("new", A, "all")[1], sec("new", A, "all")[2], sec("new", A, "all")[0]] }], 1, "It finds far more of the answer material but is no more precise: about two-thirds of what it keeps is not labelled material.");
  const L = ["pl", "de", "en", "fr", "ja", "pt", "es", "zh"];
  bars(g4, "F1 by language", "no tags · worst to best", L, [{ name: OLD, col: c0, vals: L.map(l => sec("old", A, "rr_" + l)[0]) }, { name: NEW, col: c1, lab: true, vals: L.map(l => sec("new", A, "rr_" + l)[0]) }], .6, "Every language improves. Polish is weakest and German next.");
  const Q = [["keywords", "keywords"], ["natural_question", "question"], ["long_multi_condition", "long"], ["compound", "compound"], ["kind web", "web text"], ["kind syn", "synthetic"]];
  bars(g4, "F1 by query type and source", "no tags", Q.map(q => q[1]), [{ name: OLD, col: c0, vals: Q.map(q => sec("old", A, q[0])[0]) }, { name: NEW, col: c1, vals: Q.map(q => sec("new", A, q[0])[0]) }], .6, "No query type is failing, and synthetic documents score about the same as real web text.");
  const G = [["rrval_tag", "with tags"], ["rrval_notag", "no tags"], ["rrval_swap", "tags swapped"]];
  bars(g4, "Do the metadata tags matter?", "F1 on passage text only", G.map(g => g[1]), [{ name: OLD, col: c0, vals: G.map(g => sec("old", g[0], "all")[0]) }, { name: NEW, col: c1, vals: G.map(g => sec("new", g[0], "all")[0]) }], .6, "Tags neither help nor hurt. Deliberately wrong tags cost only .02, so the model is not relying on them.");
  const B = [["musique", "MuSiQue"], ["hotpot", "HotpotQA"], ["triviaqa", "TriviaQA"], ["wiki2hop", "Wiki 2-hop"], ["wiki1hop", "Wiki 1-hop"]];
  bars(g4, "Old benchmarks: answer kept", "share of questions whose answer survives the cut", B.map(b => b[1]), [{ name: OLD, col: c0, vals: B.map(b => P.bench.old[b[0]].ans) }, { name: NEW, col: c1, vals: B.map(b => P.bench.new[b[0]].ans) }], 1, "The independent check: on data it did not train on there is no real gain, and TriviaQA drops .06.");
  bars(g4, "Old benchmarks: F1", "against the short Qwen3-14B spans", B.map(b => b[1]), [{ name: OLD, col: c0, vals: B.map(b => P.bench.old[b[0]].f1) }, { name: NEW, col: c1, vals: B.map(b => P.bench.new[b[0]].f1) }], 1, "Slightly better on four of five.");
  const sw = (w, col, n) => ({ name: n, col, pts: P.sweep[w].map(x => [x[2], x[1], "p>" + String(x[0]).replace(/^0/, "")]) });
  lines(g4, "Trade-off: what a stricter threshold buys", "no tags · token-level, each point is a keep threshold · further up and right is better", [sw("old", c0, OLD), sw("new", c1, NEW)], { t: "recall: share of the answer material kept", fmt: v => v.toFixed(2).replace(/^0/, "") }, "precision", [0, 1], [0, .6], "At any precision the pilot chopper keeps far more of the material, but it cannot get above about .4 precision without giving up recall. Averaged over the 8 languages.");
  if (P.curve.length) lines(g4, "Does more data help?", "F1 on the same holdout vs new rows used · this decides the 2M run · points appear as runs finish", [{ name: "no tags", col: c2, pts: P.curve.map(x => [x.rows, x.notag]) }, { name: "with tags", col: c3, pts: P.curve.map(x => [x.rows, x.tag]) }], { t: "new training rows (0 = the old chopper)", fmt: v => (v / 1000).toFixed(0) + "k" }, "F1", [0, 30000], [.2, .6], "If this line flattens, buying 2M rows is not worth it. Old-chopper point is not fully comparable: it was never trained on this label style.");
  if (P.holdout.full && P.noans) {
    const WH = [["old", "old 54k", c0], ["new", "pilot", c1], ["full", "fresh retrain", c2]].concat(P.holdout.fullc ? [["fullc", "continued retrain", c3]] : []);
    const ser = f => WH.map(w => ({ name: w[1], col: w[2], vals: f(w[0]) }));
    bars(g4, "Retrain: rows with an answer", "new 8-language holdout, no tags", ["precision", "recall", "F1"], ser(w => { const a = sec(w, A, "all"); return [a[1], a[2], a[0]]; }), 1, "The retrain gives up recall to stop keeping text when there is no answer.");
    bars(g4, "Retrain: says &quot;search&quot; when there is no answer", "higher is better · deleted = the hard test · last group: wrongly says search on rows with an answer, lower is better", ["deleted", "made", "real", "wrong"], ser(w => { const n = P.noans[w]; return [n.del.search, n.made.search, n.notag.search, n.wrong]; }), 1, "From never to about two-thirds on the hard test, at the cost of wrongly searching on one row in eight.");
    bars(g4, "Retrain: old benchmarks, answer kept", "pilot and old at p &gt; .8, retrain at p &gt; .5 for its new score scale", ["MuSiQue", "HotpotQA", "TriviaQA", "Wiki 2-hop", "Wiki 1-hop"], ser(w => ["musique", "hotpot", "triviaqa", "wiki2hop", "wiki1hop"].map(k => P.bench[w] && P.bench[w][k] ? P.bench[w][k].ans : null)), 1, "MuSiQue is what the retrain has to win back.");
  }
  if (DATA.answers && DATA.answers.fullc) {
    const WA = [["old", "old 54k", c0], ["new", "pilot", c1], ["full", "fresh retrain", c2], ["fullc", "continued retrain", c3]];
    bars(g4, "What matters: correct answers", "MiniCPM3-4B answerer reading each chopper's fragments · judged by gpt-5.6-luna · 62 / 96 / 98 questions, so differences under about .08 are noise", ["MuSiQue", "HotpotQA", "TriviaQA"], WA.map(w => ({ name: w[1], col: w[2], vals: ["musique", "hotpot", "triviaqa"].map(k => DATA.answers[w[0]][k]) })), 1, "The continued retrain ties the old chopper on answers while keeping 10-23% less text, and it adds no-answer handling. The fresh retrain loses MuSiQue.");
  }
  const AR = [["full", "full memory, 32K"], ["w16k", "window 16K"], ["w8k", "window 8K"], ["w4k", "window 4K"], ["w8k_nopin", "window 8K, no pin"], ["tiers8k", "averaged tiers, 8K memory"]];
  const C = [["single_age1000", "1K"], ["single_age3000", "3K"], ["single_age6000", "6K"], ["single_age12000", "12K"], ["single_age24000", "24K"], ["multivalue", "4 scattered facts"], ["vt", "5-hop chain"]];
  let t = `<h3>Share found, by how far back the fact sits (in tokens)</h3><p>darker = found more often · hard windows go from 1.00 to 0 at their size</p><table class="heat"><tr><th class="r">memory</th>${C.map(c => `<th>${c[1]}</th>`).join("")}</tr>`;
  for (const [a, n] of AR) { const R = P.window[a]; if (!R) continue; t += `<tr><th class="r">${n}</th>${C.map(c => { const v = R[c[0]]; const pct = Math.round(v * 100);
      return `<td style="background:color-mix(in srgb,var(--c0) ${pct}%,var(--panel));color:${pct > 55 ? "#fff" : "var(--fg)"}" title="${n} · ${c[1]}: ${v.toFixed(3)}">${v === 1 ? "1.00" : v === 0 ? "0" : v.toFixed(2).replace(/^0/, "")}</td>`; }).join("")}</tr>`; }
  document.getElementById("g5").innerHTML = t + "</table>";
}
function render() {
  renderPilot();
  document.querySelectorAll("#zoombar button").forEach(b => { const [a, c] = b.dataset.r.split(",").map(Number); b.setAttribute("aria-pressed", a === ZXG[0] && Math.min(c, XMAX) === ZXG[1]); });
  document.getElementById("g1").innerHTML = "";
  const f2 = v => v.toFixed(2), f3 = v => v.toFixed(3), pc = v => (v * 100).toFixed(0) + "%";
  const P = [["loss", "Train KEEP loss", "balanced BCE (binary cross-entropy), lower is better", "train", null, f2],
             ["f1", "Train KEEP F1", "at p > 0.5, against the Qwen3-14B spans", "train", [0, 1], f2],
             ["bal", "Train balanced accuracy", "mean of recall on kept and on dropped tokens", "train", [0.5, 1], f2],
             ["ans", "Val answer-in-kept", "share of questions whose answer survives the chop · per dataset M / H / T", "val", [0.5, 1], f2],
             ["kept", "Val share of text kept", "lower = less context spent", "val", [0, 0.25], pc]];
  for (const [key, t, note, part, yr, fmt] of P) {
    const from = run => (run[part] || []).length ? run[part][0].rows : null;     // first row the run logged anything
    const S = NAMES.map((n, i) => ({ name: n, col: css(COLS[i]), dots: part === "val", from: part === "train" ? from(DATA.choppers[n]) : null, pts: seriesFor(DATA.choppers[n], part, key) }));
    REFS.forEach(n => S.push({ name: n, col: css(COLS[DATA.reference[n].ci]), dash: true, dots: part === "val", from: part === "train" ? from(DATA.reference[n]) : null, pts: seriesFor(DATA.reference[n], part, key) }));
    panel(document.getElementById("g1"), t, note, S, yr, fmt);
  }
  // 2026-11-04 (user: "all numbers from table on one chart, with all lines as growing, so convert CE to sth that grows"):
  // per arm, every val number on the chopper's notes; 2026-11-04 (user: "run luna verification instead of exact that is
  // misleading (remove it from charts)"): exact match is replaced by the gpt-5.6-luna verdict; CE shown as exp(−CE) = the mean per-token probability of the target
  const g3 = document.getElementById("g3"); g3.innerHTML = "";
  const MET = [["judged", "luna-judged correct", "--ca", v => v], ["tok", "tok", "--c0", v => v], ["ce", "exp(−CE)", "--c2", v => Math.exp(-v)], ["search", "follow-up asked when needed", "--c1", v => v]];
  // 2026-11-04 (user: "put m h t benchmarks on separate charts, chopper fragments and qwen fragments together … describe each
  // benchmark specifics below the chart"): one chart per benchmark; colour = metric, line style = training arm
  const ARMS = [[ANS.find(n => /chopper/.test(n)), "trained on chopper fragments", null], [ANS.find(n => /Qwen/.test(n)), "trained on Qwen3-14B fragments", "7 4"]];
  const DSN = [["musique", "MuSiQue", `<b>MuSiQue</b> — multi-hop questions built by chaining 2–4 single-hop Wikipedia questions
    (“Who is the spouse of the Green performer?”), composed so that no hop can be skipped. Of the 100 validation questions, 62 have
    every needed paragraph among the 16 passages and are graded on the answer; the other 38 miss a hop, so the right move is a
    <code>[search]</code> follow-up — the orange line is shown only here. The hardest set: the answer string is in the chopper's
    fragments for .74 of the 62, and the answerer still has to combine the facts.`],
    ["hotpot", "HotpotQA", `<b>HotpotQA</b> — two-hop questions written by crowdworkers over two linked Wikipedia paragraphs: either a
    bridge (find an entity, then a fact about it) or a comparison of two entities (which is older, are both …). 96 of the 100 have both
    paragraphs among the 16. The answer string is in the chopper's fragments for .83 of them; the answerer gets about 95% of those.`],
    ["triviaqa", "TriviaQA", `<b>TriviaQA</b> — quiz questions written independently of any text, with evidence matched to them later:
    one fact, but asked with indirect quiz-style clues, and many answers have several valid names (where exact match was most wrong).
    98 of the 100 are graded. The answer string is in the chopper's fragments for .72 of them; the answerer scores slightly above that,
    because MiniCPM3-4B already knows some answers.`]];
  const leg = d => MET.filter(([k]) => k !== "search" || d === "musique").map(([, lab, col]) => `<span style="white-space:nowrap"><i class="sw" style="background:var(${col});display:inline-block;width:14px;height:3px;vertical-align:middle"></i> ${lab}</span>`).join(" · ");
  for (const [d, dl, desc] of DSN) {
    const S = [];
    for (const [n, al, dash] of ARMS) {
      const V = ((n && DATA.answerer[n].val) || []).filter(p => p.m && p.m[d]);
      for (const [k, lab, col, f] of MET) {
        if (k === "search" && d !== "musique") continue;     // HotpotQA / TriviaQA have 4 and 2 such questions
        S.push({ name: `${lab} · ${al}`, col: css(col), dash, lw: 2, dots: false, pts: V.filter(p => p.m[d][k] != null).map(p => ({ x: p.rows, y: f(p.m[d][k]) })) });
      }
    }
    panel(g3, `${dl}: answerer on the chopper's val fragments`, `colour = metric: ${leg(d)} · line = training arm: solid = chopper fragments, dashed = Qwen3-14B fragments · all higher-is-better`,
          S, [0, 1], f2, [0, 27000]);
    g3.lastChild.insertAdjacentHTML("beforeend", `<p class="desc">${desc}</p>`);
  }
}
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
</script></body></html>"""
open(sys.argv[1], "w").write(html.replace("__DATA__", json.dumps(D)).replace("__NOW__", now))
print("PAGE", sys.argv[1])
