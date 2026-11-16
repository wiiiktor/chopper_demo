// Chopper in the browser: a port of chopper_rag/core.py (tokenisation with offsets, packing, decoding, ranking).
// The model is one ONNX graph (mmBERT encoder + KEEP head + decision head) giving p(KEEP) per token and p(ANSWER).
export const MAX_LEN = 8192, TOP_K_TOKENS = 16, SAFETY = 32;

export class Chopper {
  constructor(tokenizer, session, ort, cfg = {}) {
    this.tok = tokenizer; this.session = session; this.ort = ort;
    const d = cfg.decode || {};
    this.snap = "sentence";   // "sentence" (default, as chopper-rag 0.1.7) | "word"
    this.threshold = d.pred_thr ?? 0.5; this.mergeGap = d.merge_gap ?? 5; this.minRun = d.min_run ?? 4;
    this.sep = ` ${tokenizer.sep_token} `;
    const v = tokenizer.get_vocab(); this.inv = [];
    for (const [t, i] of (v instanceof Map ? v.entries() : Object.entries(v))) this.inv[i] = t;
    this.special = new Set(tokenizer.all_special_tokens);
    const ids = v instanceof Map ? v : new Map(Object.entries(v));
    this.bosId = ids.get(tokenizer.bos_token ?? tokenizer.cls_token) ?? 2;   // mmBERT <bos> = 2, pl-ModernBERT <s> = 0
    // pl-ModernBERT (Unigram, WhitespaceSplit + Metaspace, NFKC): tokens never carry whitespace, so offsets are found by aligning
    // each token's text to the input after skipping whitespace (chopper_config.json "js": {"offsets": "align"}).
    this.align = (cfg.js || {}).offsets === "align";
  }

  _encodeAlign(text, ids, off) {
    const pid = this.tok.encode(text, { add_special_tokens: false }); let c = 0;
    const txt = id => (this.inv[id] ?? "").replace(/▁/g, "");
    for (let k = 0; k < pid.length; k++) {
      const id = pid[k], t = this.inv[id] ?? "", s = txt(id);
      ids.push(id);
      let a = c; while (a < text.length && /\s/.test(text[a])) a++;
      if (!s) { off.push([a, Math.min(a + 1, text.length)]); continue; }   // a lone "▁": the Rust tokenizer gives it the next character
      if (t !== this.tok.unk_token && text.startsWith(s, a)) { off.push([a, a + s.length]); c = a + s.length; continue; }
      // <unk> (e.g. "Q" is not in the pl-ModernBERT vocabulary) or NFKC-changed text: up to where the next token's text starts,
      // at most to the end of this run of non-space characters
      let r = a + 1; while (r < text.length && !/\s/.test(text[r])) r++;
      const nid = pid[k + 1], nx = k + 1 < pid.length && this.inv[nid] !== this.tok.unk_token ? txt(nid) : "";
      const j = nx ? text.indexOf(nx, a + 1) : -1;
      const e = j > a && j <= r ? j : r;
      off.push([a, e]); c = e;
    }
  }


  // ---------------------------------------------------------------- tokenisation with character offsets
  // transformers.js merges a run of spaces into one "▁▁" token; the Rust tokenizer (used in training) starts a new word
  // at every space. Encoding each word-group separately (split before a space that follows a space) reproduces it.
  _encodeText(text, base, ids, off) {
    if (this.align) return this._encodeAlign(text, ids, off);
    const pieces = []; let st = 0;
    for (let i = 1; i < text.length; i++) if (text[i] === " " && text[i - 1] === " ") { pieces.push([st, i]); st = i; }
    pieces.push([st, text.length]);
    for (const [a, b] of pieces) {
      const piece = text.slice(a, b);
      if (!piece) continue;
      const pid = this.tok.encode(piece, { add_special_tokens: false });
      let c = a, bytes = [], bstart = -1, btoks = [];
      const flush = () => {
        if (!btoks.length) return;
        const str = new TextDecoder().decode(new Uint8Array(bytes));
        const e = Math.min(b, bstart + str.length);
        for (const k of btoks) off[k] = [bstart, e];
        c = e; bytes = []; btoks = []; bstart = -1;
      };
      for (const id of pid) {
        const t = this.inv[id] ?? "";
        ids.push(id); off.push([c, c]);
        const k = off.length - 1;
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(t);
        if (m) {                                             // byte fallback: tokens share the span of their character(s)
          if (bstart < 0) bstart = c;
          bytes.push(parseInt(m[1], 16)); btoks.push(k);
          try { new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(bytes)); flush(); } catch (_) { /* incomplete */ }
          continue;
        }
        flush();
        let s = t.replace(/▁/g, " ");
        if (s === " " && text[c] !== " " && c < b) { off[k] = [c, c + 1]; continue; }   // a lone prepended "▁": Rust gives it the next char
        if (s.startsWith(" ") && text[c] !== " ") s = s.slice(1);   // "▁" prepended by the tokenizer: no character
        const e = Math.min(b, c + s.length);
        off[k] = [c + 0, e]; c = e;
      }
      flush();
    }
  }

  encode(s) {
    // s = head + (sep + passage)*: split at the separator token, encode the text between them
    const ids = [this.bosId], off = [[0, 0]];                // <bos> / <s>
    const sepTok = this.tok.sep_token; let pos = 0;
    while (pos <= s.length) {
      let j = s.indexOf(sepTok, pos); if (j < 0) j = s.length;
      if (j > pos) this._encodeTextAt(s, pos, j, ids, off);
      if (j >= s.length) break;
      ids.push(this.tok.sep_token_id); off.push([j, j + sepTok.length]); pos = j + sepTok.length;
    }
    ids.push(this.tok.eos_token_id); off.push([0, 0]);      // <eos>
    return { ids, off };
  }

  _encodeTextAt(s, a, b, ids, off) {
    const n0 = off.length;
    this._encodeText(s.slice(a, b), 0, ids, off);
    for (let k = n0; k < off.length; k++) off[k] = [off[k][0] + a, off[k][1] + a];
  }

  ntok(text) { return this.tok.encode(text, { add_special_tokens: false }).length; }

  // ---------------------------------------------------------------- model
  async _probs(ids) {
    const T = this.ort.Tensor, n = ids.length;
    const feeds = { input_ids: new T("int64", BigInt64Array.from(ids.map(BigInt)), [1, n]),
                    attention_mask: new T("int64", new BigInt64Array(n).fill(1n), [1, n]) };
    const o = await this.session.run(feeds);
    return { p: Float32Array.from(o.p_keep.data), pAns: Number(o.p_answer.data[0]) };
  }

  // ---------------------------------------------------------------- public API (same results as Chopper.run in Python)
  async run(question, passages) {
    const head = `Question: ${question}`;
    const budget = (this.maxLen || MAX_LEN) - this.ntok(head) - 2 - SAFETY;   // maxLen < 8192 in browsers: wasm memory (attention is quadratic)
    const segs = [];
    passages.forEach((p, i) => {
      const s = p.trim(); if (!s) return;
      const lead = p.length - p.trimStart().length;
      segs.push([i, lead, s, this.ntok(s)]);                // (passages longer than the budget are cut by the caller)
    });
    const groups = []; let g = [], used = 0;
    for (const sg of segs) { const cost = sg[3] + 2; if (g.length && used + cost > budget) { groups.push(g); g = []; used = 0; } g.push(sg); used += cost; }
    if (g.length) groups.push(g);
    const rel = passages.map(() => 0); let chops = [], pAns = 0;
    for (const gr of groups) { const r = await this._forward(head, gr, passages, rel); chops = chops.concat(r.chops); pAns = Math.max(pAns, r.pAns); }
    if (!groups.length) pAns = (await this._forward(head, [], passages, rel)).pAns;
    chops.sort((x, y) => x.passage - y.passage || x.start - y.start);
    const ranking = rel.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]);
    return { chops, decision: pAns >= 0.5 ? "ANSWER" : "SEARCH", pAnswer: pAns, scores: rel, ranking };
  }

  async _forward(head, group, passages, rel) {
    let s = head; const spans = [];
    for (const [i, poff, text] of group) { s += this.sep; spans.push([s.length, s.length + text.length, i, poff]); s += text; }
    let { ids, off } = this.encode(s);
    if (ids.length > MAX_LEN) { ids = ids.slice(0, MAX_LEN - 1).concat([this.tok.eos_token_id]); off = off.slice(0, MAX_LEN - 1).concat([[0, 0]]); }
    const { p, pAns } = await this._probs(ids);
    const body = off.map(([a, b]) => a >= head.length && b > a);
    for (const [sa, sb, i] of spans) {                       // relevance: mean of the passage's top-16 token p(KEEP)
      const v = []; off.forEach(([a, b], k) => { if (a >= sa && b <= sb && b > a) v.push(p[k]); });
      if (v.length) { v.sort((x, y) => y - x); const t = v.slice(0, TOP_K_TOKENS); rel[i] = Math.max(rel[i], t.reduce((x, y) => x + y, 0) / t.length); }
    }
    let q = Array.from(p, (x, k) => x > this.threshold && body[k]);
    q = shape(q, this.mergeGap, this.minRun).map((x, k) => x && body[k]);
    const out = [], sentCache = {};
    for (const [a, b] of keptText(s, off, q)) {
      for (const [sa, sb, i, poff] of spans) {
        let x = Math.max(a, sa), y = Math.min(b, sb);
        if (y <= x) continue;
        const t = s.slice(x, y);
        x += t.length - t.trimStart().length; y -= t.length - t.trimEnd().length;
        while (x > sa && WORD.test(s[x - 1]) && WORD.test(s[x])) x--;   // never cut inside a word
        while (y < sb && WORD.test(s[y - 1]) && WORD.test(s[y])) y++;
        if (y - x <= 2) continue;
        const ts = []; off.forEach(([u, v], k) => { if (q[k] && u < y && v > x) ts.push(p[k]); });
        let start = poff + (x - sa), end = poff + (y - sa);
        if (this.snap === "sentence") {                          // whole sentences the chop touches (as snap_chops)
          const P = passages[i], ss = (sentCache[i] ||= sentences(P));
          for (const [u, v] of ss) { if (u <= start && v > start) start = Math.min(start, u); if (u < end && v >= end) end = Math.max(end, v); }
          while (start < end && /\s/.test(P[start])) start++;
          while (end > start && /\s/.test(P[end - 1])) end--;
        }
        const last = out[out.length - 1];
        if (last && last.passage === i && start <= last.end) { last.end = Math.max(last.end, end); last.text = passages[i].slice(last.start, last.end); continue; }
        out.push({ passage: i, start, end, text: passages[i].slice(start, end), score: ts.length ? ts.reduce((m, z) => m + z, 0) / ts.length : null });
      }
    }
    return { chops: out, pAns };
  }
}

function runs(m) {
  const r = []; let st = -1;
  for (let i = 0; i <= m.length; i++) {
    const v = i < m.length && m[i];
    if (v && st < 0) st = i; else if (!v && st >= 0) { r.push([st, i]); st = -1; }
  }
  return r;
}

export function shape(q, gap = 5, mn = 4) {              // merge kept runs <= gap tokens apart, drop runs shorter than mn
  q = q.slice(); const rr = runs(q);
  for (let k = 0; k + 1 < rr.length; k++) if (rr[k + 1][0] - rr[k][1] <= gap) for (let i = rr[k][1]; i < rr[k + 1][0]; i++) q[i] = true;
  for (const [a, b] of runs(q)) if (b - a < mn) for (let i = a; i < b; i++) q[i] = false;
  return q;
}

const WORD = /[\p{L}\p{N}_]/u;
const SENT_END = /[.!?。！？…]+["'”’»)\]]*(?=\s|$)|\n\s*\n|\n(?=\s*[-•*\d])/gu;
export function sentences(t) {                            // (start, end) of sentences, as chopper_rag.core._sentences
  const out = []; let a = 0, m; SENT_END.lastIndex = 0;
  while ((m = SENT_END.exec(t)) !== null) { const b = m.index + m[0].length; if (t.slice(a, b).trim()) out.push([a, b]); a = b; if (m[0].length === 0) SENT_END.lastIndex++; }
  if (t.slice(a).trim()) out.push([a, t.length]);
  return out;
}
export function keptText(s, off, q) {                     // kept tokens -> character runs; bridge one char between kept chars
  const cm = new Array(s.length + 1).fill(false);
  off.forEach(([a, b], k) => { if (q[k]) for (let i = a; i < b; i++) cm[i] = true; });
  const c2 = cm.slice();
  for (let i = 1; i < cm.length - 1; i++) c2[i] = cm[i] || (cm[i - 1] && cm[i + 1]);
  return runs(c2);
}

// ~size-character passages: paragraphs packed; a longer paragraph cut at a sentence or word end (as split_passages)
export function splitPassages(text, size = 500, pack = true) {   // pack=false: every paragraph is its own passage
  const out = []; let start = null, end = null;
  const re = /\S(?:[\s\S]*?\S)?(?=\s*\n\s*\n|\s*$)/g; let m;
  while ((m = re.exec(text)) !== null) {
    let a = m.index; const b = m.index + m[0].length;
    while (b - a > size) {
      let cut = text.lastIndexOf(". ", a + size - 2);
      if (cut > a + Math.floor(size / 2)) cut = cut + 1;
      else { const sp = text.lastIndexOf(" ", a + size - 1); cut = sp > a ? sp : a + size; }
      if (start !== null) { out.push([start, text.slice(start, end)]); start = null; }
      out.push([a, text.slice(a, cut)]); a = cut;
      while (a < b && /\s/.test(text[a])) a++;
    }
    if (!pack) { if (b > a) out.push([a, text.slice(a, b)]); if (m[0].length === 0) re.lastIndex++; continue; }
    if (start !== null && b - start > size) { out.push([start, text.slice(start, end)]); start = null; }
    if (start === null) start = a;
    end = b;
    if (m[0].length === 0) re.lastIndex++;
  }
  if (start !== null) out.push([start, text.slice(start, end)]);
  return out.filter(([, p]) => p.trim());
}
