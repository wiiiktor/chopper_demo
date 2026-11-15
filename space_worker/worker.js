// Chopper demo proxy: lets the static Hugging Face Space ask Qwen3-14B (OpenRouter) without exposing the API key.
// Only this model, short answers, size and rate limits, only the Space's origin. Secret: OPENROUTER_KEY (wrangler secret put).
const MODEL = "qwen/qwen3-14b";
const ALLOWED_ORIGINS = ["https://wiiiktor-chopper.static.hf.space", "https://wiiiktor-chopper-pl.static.hf.space", "https://huggingface.co"];
const MAX_CHARS = 60000;          // question + context; 20 MuSiQue paragraphs are ~12k characters
const INS = "Answer the question using ONLY the notes. Reply with just the answer (a name, date, number or short phrase). " +
            "If the notes do not contain enough to answer, reply exactly: not in notes.\n\n";

function cors(origin) {
  return { "Access-Control-Allow-Origin": ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
           "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type", "Vary": "Origin" };
}
const json = (obj, status, origin) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json", ...cors(origin) } });

export default {
  async fetch(req, env) {
    const origin = req.headers.get("Origin") || "";
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
    if (req.method !== "POST") return json({ error: "POST only" }, 405, origin);
    if (!ALLOWED_ORIGINS.includes(origin)) return json({ error: "origin not allowed" }, 403, origin);
    if (env.LIMITER) {                                   // Workers rate limiting binding: per visitor IP
      const { success } = await env.LIMITER.limit({ key: req.headers.get("CF-Connecting-IP") || "anon" });
      if (!success) return json({ error: "Too many requests, please wait a minute." }, 429, origin);
    }
    let body;
    try { body = await req.json(); } catch { return json({ error: "bad JSON" }, 400, origin); }
    const q = String(body.question || "").slice(0, 500), notes = String(body.notes || "");
    if (!q.trim()) return json({ error: "empty question" }, 400, origin);
    if (q.length + notes.length > MAX_CHARS) return json({ error: "text too long" }, 413, origin);
    const t0 = Date.now();
    const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: { "Authorization": `Bearer ${env.OPENROUTER_KEY}`, "Content-Type": "application/json",
                 "HTTP-Referer": "https://huggingface.co/spaces/wiiiktor/chopper", "X-Title": "Chopper demo" },
      body: JSON.stringify({ model: MODEL, max_tokens: 64, temperature: 0, reasoning: { enabled: false },
                             messages: [{ role: "user", content: INS + `Notes:\n${notes}\n\nQuestion: ${q}` }] }),   // same prompt as the hop study (qwen_answer.py)
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) return json({ error: (d.error && d.error.message) || `upstream ${r.status}` }, 502, origin);
    const text = ((d.choices && d.choices[0] && d.choices[0].message && d.choices[0].message.content) || "").split("</think>").pop().trim();
    const cut = !!(d.choices && d.choices[0] && d.choices[0].finish_reason === "length");
    return json({ answer: text, cut, prompt_tokens: d.usage ? d.usage.prompt_tokens : null, ms: Date.now() - t0 }, 200, origin);
  },
};
