#!/usr/bin/env python
"""HTML page of the chopper / answerer training curves (2026-10-30, user: "update charts on an artifact").
Reads release/figures/curves.json (written by plot_curves.py) and writes a self-contained page with the data inlined.
usage: python curves_page.py <out.html>"""
import json, os, sys, datetime
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "curves.json")))
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
#g3{display:grid;gap:16px;margin-top:16px}#g3 .card{max-width:760px}
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
</style></head><body><main>
<h1>Chopper training curves</h1>
<p class="sub">The choppers learn which tokens of 16 retrieved passages to keep; the answerer (MiniCPM3-4B LoRA) learns to answer
or ask a follow-up question. The two mmBERTs and the MiniCPM3-4B slice (MLA, multi-head latent attention: every slice head reads the
full model's whole conversation cache) are trained once on all 54k rows mixed (benchmark + Wikipedia). The answerer has two arms
with the same targets, trained on the integrated chopper's fragments or on the Qwen3-14B labeller's; both are scored on the chopper's
validation fragments.
Train lines are 500-row rolling means; validation points are the mean over MuSiQue, HotpotQA and TriviaQA (300 questions), read at p&nbsp;&gt;&nbsp;0.5.</p>
<div class="meta">data from the train logs · regenerated __NOW__ Warsaw</div>
<div class="legend" id="legend"></div>
<div class="zoombar" id="zoombar">Zoom:
<button data-r="0,999999">all</button><button data-r="0,27000">0–27k</button><button data-r="27000,55000">27k–54k</button>
<label><input type="checkbox" id="fity" checked> fit y to the view</label>
<span class="hint">· or drag across any chart; double-click a chart to reset</span></div>
<h2>Chopper</h2><div class="grid" id="g1"></div>
<h2>Answerer</h2><div class="grid" id="g2"></div><div id="g3"></div>
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
const ANS = Object.keys(DATA.answerer), ACOL = ["--ca", "--c3"];
ANS.forEach((n, i) => lg.insertAdjacentHTML("beforeend", `<span><i class="sw" style="background:var(${ACOL[i]})"></i>MiniCPM3-4B ${n}${(DATA.answerer[n].train || []).length ? "" : " (queued)"}</span>`));
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
function render() {
  document.querySelectorAll("#zoombar button").forEach(b => { const [a, c] = b.dataset.r.split(",").map(Number); b.setAttribute("aria-pressed", a === ZXG[0] && Math.min(c, XMAX) === ZXG[1]); });
  document.getElementById("g1").innerHTML = ""; document.getElementById("g2").innerHTML = "";
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
  const AS = (part, key) => ANS.map((n, i) => ({ name: n, col: css(ACOL[i]), dots: part === "val", pts: seriesFor(DATA.answerer[n], part, key) }));
  panel(document.getElementById("g2"), "Train CE", "cross-entropy on the answer / follow-up tokens", AS("train", "ce"), null, f3);
  panel(document.getElementById("g2"), "Train token accuracy", "share of target tokens predicted exactly", AS("train", "tok"), null, f3);
  panel(document.getElementById("g2"), "Val exact (chopper notes)", "per dataset M / H / T in the tooltip", AS("val", "exact"), [0, 1], f2);
  // 2026-11-04 (user: "all numbers from table on one chart, with all lines as growing, so convert CE to sth that grows"):
  // per arm, every val number on the chopper's notes; CE shown as exp(−CE) = the mean per-token probability of the target
  const g3 = document.getElementById("g3"); g3.innerHTML = "";
  const MET = [["exact", "exact", "--ca", v => v], ["tok", "tok", "--c0", v => v], ["ce", "exp(−CE)", "--c2", v => Math.exp(-v)], ["search", "follow-up asked when needed", "--c1", v => v]];
  const DSN = [["musique", "MuSiQue", null], ["hotpot", "HotpotQA", "7 4"], ["triviaqa", "TriviaQA", "2 3"]];
  ANS.forEach(n => {
    const V = (DATA.answerer[n].val || []).filter(p => p.m), S = [];
    if (!V.length) { g3.insertAdjacentHTML("beforeend", `<p class="sub">${n}: every val number on one chart — queued</p>`); return; }
    for (const [k, lab, col, f] of MET) for (const [d, dl, dash] of DSN) {
      if (k === "search" && d !== "musique") continue;     // HotpotQA / TriviaQA have 4 and 2 such questions
      const pts = V.filter(p => p.m[d] && p.m[d][k] != null).map(p => ({ x: p.rows, y: f(p.m[d][k]) }));
      S.push({ name: `${lab} · ${dl}`, col: css(col), dash, lw: 2, dots: false, pts });
    }
    const leg = MET.map(([, lab, col]) => `<span style="white-space:nowrap"><i class="sw" style="background:var(${col});display:inline-block;width:14px;height:3px;vertical-align:middle"></i> ${lab}</span>`).join(" · ");
    panel(g3, `${n}: every val number, all higher-is-better`, `on the integrated chopper's val fragments · colour = metric: ${leg} · line = dataset: solid MuSiQue, dashed HotpotQA, dotted TriviaQA (follow-up: MuSiQue only, 38 questions)`,
          S, [0, 1], f2, [0, 27000]);
  });
}
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
</script></body></html>"""
open(sys.argv[1], "w").write(html.replace("__DATA__", json.dumps(D)).replace("__NOW__", now))
print("PAGE", sys.argv[1])
