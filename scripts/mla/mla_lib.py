"""10% WIDTH KEEPER for an MLA model (multi-head latent attention), 2026-10-31 (user: "start work on the MiniCPM3-4B",
after "are there any transformer or llm architectures … that would make fast chopper implementation easier than qwen?").
The MLA counterpart of keeper_lib.py, written for openbmb/MiniCPM3-4B (62 layers, hidden 2560, 40 heads, latent 256 +
32 RoPE dims, query compression 768, MLP 6,400).

Why MLA: every head rebuilds its keys/values from ONE shared latent per token, so the cache holds the latent (288 numbers
per token per layer), not per-head K/V. A slice of h heads therefore reads the WHOLE conversation memory — the Qwen GQA
slice sees 1 of 8 KV groups — and, because it keeps the two small full-width compressions (2560→768 for queries,
2560→288 for the latent), the latents it writes for its own tokens have the full model's format.

Pure torch over the checkpoint's safetensors (the repo's remote code targets transformers 4.41 and its cache API is gone
in 5.x). Pieces, mirroring keeper_lib:
  MiniCPM3      the full model: forward(ids, cache, pos0) with a LATENT cache — per layer (c [1,T,256] after kv_a_layernorm,
                kpe [1,T,32] after RoPE); crop = slicing. Optional importance statistics.
  importance()  per HEAD (norm of the head's attention output at the o_proj input) and per MLP NEURON (mean |silu(gate)·up|).
  rewire()      function-preserving permutation of heads (q_b rows, kv_b rows, o_proj columns) and neurons.
  Keeper        the leading h heads + n neurons of every layer, chopper-mode LoRA + KEEP head, reading the full model's
                latent cache of the conversation plus its own latents of the passages."""
import glob, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention.bias import causal_lower_right
from torch.utils.checkpoint import checkpoint


def rms(x, w, eps):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rot_half(x):
    h = x.shape[-1] // 2; return torch.cat([-x[..., h:], x[..., :h]], -1)


class LoRA(nn.Module):
    def __init__(self, din, dout, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, din) / math.sqrt(din)); self.B = nn.Parameter(torch.zeros(dout, r))

    def forward(self, x):
        return (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)


class MiniCPM3:
    """weights as plain tensors; layer i = dict of its tensors. Scales of the architecture (scale_emb on the embedding,
    scale_depth/sqrt(L) on every residual branch, logits / (hidden / dim_model_base)) are applied as in the original."""

    def __init__(self, path, dev="cuda", dt=torch.bfloat16):
        if not os.path.isdir(path):                          # a Hugging Face id, e.g. openbmb/MiniCPM3-4B
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        c = json.load(open(os.path.join(path, "config.json"))); self.cfg = c
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(f))
        for f in [] if sd else sorted(glob.glob(os.path.join(path, "*.bin"))):   # MiniCPM3-4B ships one pytorch_model.bin
            sd.update(torch.load(f, map_location="cpu", weights_only=True, mmap=True))
        g = lambda k: sd.pop(k).to(dev, dt)
        self.H, self.L, self.nh = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
        self.dn, self.dr, self.R = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["kv_lora_rank"]
        self.dv = self.H // self.nh; self.dq = self.dn + self.dr; self.I = c["intermediate_size"]
        self.eps = c["rms_norm_eps"]; self.res = c["scale_depth"] / math.sqrt(self.L)
        self.emb_scale = c["scale_emb"]; self.logit_div = self.H / c["dim_model_base"]
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm_head = g("lm_head.weight") if "lm_head.weight" in sd else self.embed
        names = {"ln1": "input_layernorm", "ln2": "post_attention_layernorm", "q_a": "self_attn.q_a_proj",
                 "q_a_ln": "self_attn.q_a_layernorm", "q_b": "self_attn.q_b_proj", "kv_a": "self_attn.kv_a_proj_with_mqa",
                 "kv_a_ln": "self_attn.kv_a_layernorm", "kv_b": "self_attn.kv_b_proj", "o": "self_attn.o_proj",
                 "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
        self.layers = [{k: g(f"model.layers.{i}.{v}.weight") for k, v in names.items()} for i in range(self.L)]
        rs = c.get("rope_scaling") or {}
        inv = 1.0 / (c.get("rope_theta", 10000.0) ** (torch.arange(0, self.dr, 2).float() / self.dr))
        if rs.get("type") == "longrope":                    # max_position == original → short factors, scale 1
            orig = rs["original_max_position_embeddings"]; mp = c["max_position_embeddings"]
            inv = inv / torch.tensor(rs["long_factor"] if mp > orig else rs["short_factor"], dtype=torch.float32)
            self.rope_mag = math.sqrt(1 + math.log(mp / orig) / math.log(orig))
        else:
            self.rope_mag = 1.0
        self.inv_freq = inv.to(dev); self.dev, self.dt = dev, dt
        self.stats = None                                   # set to a list of [head, neuron] accumulators for importance()

    def rope(self, pos):
        f = torch.outer(pos.float(), self.inv_freq); e = torch.cat([f, f], -1)
        return (e.cos() * self.rope_mag)[None, None], (e.sin() * self.rope_mag)[None, None]   # [1,1,T,dr] fp32

    def latent(self, i, h, cos, sin, lo=None):
        """the per-token latent this layer caches: (c [1,T,R] normed, kpe [1,T,dr] roped)"""
        l = self.layers[i]
        ck = F.linear(h, l["kv_a"]) + (lo["kv_a"](h) if lo is not None else 0)
        c, kpe = ck.split([self.R, self.dr], -1)
        kpe = (kpe.float()[:, None] * cos + rot_half(kpe.float()[:, None]) * sin)[:, 0].to(h.dtype)
        return rms(c, l["kv_a_ln"], self.eps), kpe

    def attn(self, i, h, cos, sin, past, nh, lo=None):
        """attention of the first nh heads (all of them for the full model). past = (c, kpe) of earlier tokens or None.
        Returns (output [1,T,H], the new tokens' latent)."""
        l = self.layers[i]; T = h.shape[1]; dq, dn, dr, dv = self.dq, self.dn, self.dr, self.dv
        qa = rms(F.linear(h, l["q_a"]) + (lo["q_a"](h) if lo is not None else 0), l["q_a_ln"], self.eps)
        q = (F.linear(qa, l["q_b"][:nh * dq]) + (lo["q_b"](qa) if lo is not None else 0)).view(1, T, nh, dq).transpose(1, 2)
        qn, qp = q.split([dn, dr], -1)
        qp = (qp.float() * cos + rot_half(qp.float()) * sin).to(h.dtype)
        c, kpe = self.latent(i, h, cos, sin, lo)
        C, P = (torch.cat([past[0].to(c.dtype), c], 1), torch.cat([past[1].to(kpe.dtype), kpe], 1)) if past is not None else (c, kpe)
        kv = (F.linear(C, l["kv_b"][:nh * (dn + dv)]) + (lo["kv_b"](C) if lo is not None else 0))
        kv = kv.view(1, C.shape[1], nh, dn + dv).transpose(1, 2)
        kn, v = kv.split([dn, dv], -1)
        k = torch.cat([kn, P[:, None].expand(1, nh, -1, dr)], -1); qq = torch.cat([qn, qp], -1)
        o = F.scaled_dot_product_attention(qq, k, v, attn_mask=causal_lower_right(T, k.shape[2]), scale=dq ** -0.5)
        o = o.transpose(1, 2).reshape(1, T, nh * dv)
        if self.stats is not None:
            self.stats[i][0] += o.float().view(T, nh, dv).norm(dim=-1).sum(0).double().cpu()
        return F.linear(o, l["o"][:, :nh * dv]) + (lo["o"](o) if lo is not None else 0), (c, kpe)

    def mlp(self, i, h, n, lo=None):
        l = self.layers[i]
        u = F.silu(F.linear(h, l["gate"][:n]) + (lo["gate"](h) if lo is not None else 0)) * \
            (F.linear(h, l["up"][:n]) + (lo["up"](h) if lo is not None else 0))
        if self.stats is not None:
            self.stats[i][1] += u.float().abs().reshape(-1, n).sum(0).double().cpu()
        return F.linear(u, l["down"][:, :n]) + (lo["down"](u) if lo is not None else 0)

    def block(self, i, x, cos, sin, past, nh, n, lo=None):
        l = self.layers[i]
        a, lat = self.attn(i, rms(x, l["ln1"], self.eps), cos, sin, past, nh, lo)
        x = x + a * self.res
        return x + self.mlp(i, rms(x, l["ln2"], self.eps), n, lo) * self.res, lat

    @torch.no_grad()
    def forward(self, ids, cache=None, pos0=0, logits=True):
        """ids [1,T]; cache = list of per-layer (c, kpe) or None. Returns (logits or final hidden, new cache)."""
        x = F.embedding(ids, self.embed) * self.emb_scale
        cos, sin = self.rope(torch.arange(pos0, pos0 + ids.shape[1], device=ids.device)); out = []
        for i in range(self.L):
            past = cache[i] if cache is not None else None
            x, lat = self.block(i, x, cos, sin, past, self.nh, self.I)
            out.append((torch.cat([past[0], lat[0]], 1), torch.cat([past[1], lat[1]], 1)) if past is not None else lat)
        h = rms(x, self.norm, self.eps)
        return (F.linear(h / self.logit_div, self.lm_head).float() if logits else h), out


def crop(cache, n):
    """ROLLBACK on a latent cache: keep the first n positions"""
    return [(c[:, :n], p[:, :n]) for c, p in cache]


def sizes(m, frac):
    """heads h and neurons n per layer so the slice holds ≈ frac of each layer's parameters. The two compressions
    (q_a, kv_a) are counted in full: the slice keeps them whole so its latents match the full model's."""
    H = m.H; shared = H * m.cfg["q_lora_rank"] + H * (m.R + m.dr)
    per_head = m.cfg["q_lora_rank"] * m.dq + m.R * (m.dn + m.dv) + m.dv * H
    layer = shared + per_head * m.nh + 3 * H * m.I
    h = max(1, round(frac * m.nh)); n = int(max(64, min(m.I, (frac * layer - shared - h * per_head) / (3 * H))))
    n = max(64, n // 64 * 64)
    return h, n, (shared + h * per_head + 3 * H * n) / layer


@torch.no_grad()
def importance(m, batches):
    m.stats = [[torch.zeros(m.nh, dtype=torch.float64), torch.zeros(m.I, dtype=torch.float64)] for _ in range(m.L)]
    cnt = 0
    for ids in batches:
        m.forward(ids.to(m.dev), logits=False); cnt += ids.numel()
    st, m.stats = m.stats, None
    return [(a / cnt, b / cnt) for a, b in st]


@torch.no_grad()
def rewire(m, perms):
    """perms: per layer (head order [nh], neuron order [I]), most important first. In place, function-preserving."""
    dq, dn, dv = m.dq, m.dn, m.dv
    for l, (ph, pn) in zip(m.layers, perms):
        qi = torch.cat([torch.arange(j * dq, (j + 1) * dq) for j in ph.tolist()]).to(m.dev)
        ki = torch.cat([torch.arange(j * (dn + dv), (j + 1) * (dn + dv)) for j in ph.tolist()]).to(m.dev)
        oi = torch.cat([torch.arange(j * dv, (j + 1) * dv) for j in ph.tolist()]).to(m.dev); pn = pn.to(m.dev)
        l["q_b"], l["kv_b"], l["o"] = l["q_b"][qi].contiguous(), l["kv_b"][ki].contiguous(), l["o"][:, oi].contiguous()
        l["gate"], l["up"], l["down"] = l["gate"][pn].contiguous(), l["up"][pn].contiguous(), l["down"][:, pn].contiguous()


def perms_from(imp):
    return [(a.argsort(descending=True), b.argsort(descending=True)) for a, b in imp]


class Keeper(nn.Module):
    """leading h heads + n neurons of every layer of a REWIRED MiniCPM3 + chopper-mode LoRA + KEEP head"""

    def __init__(self, m, h, n, r=32, ckpt=False):
        super().__init__()
        self.m, self.h, self.n, self.ckpt = m, h, n, ckpt; H, qr = m.H, m.cfg["q_lora_rank"]
        dims = {"q_a": (H, qr), "q_b": (qr, h * m.dq), "kv_a": (H, m.R + m.dr), "kv_b": (m.R, h * (m.dn + m.dv)),
                "o": (h * m.dv, H), "gate": (H, n), "up": (H, n), "down": (n, H)}
        self.lora = nn.ModuleList([nn.ModuleDict({k: LoRA(a, b, r) for k, (a, b) in dims.items()}) for _ in range(m.L)])
        self.keep = nn.Linear(H, 1)

    def forward(self, ids, cache, pos0, hidden=False):
        """ids [1,T] passage tokens; cache = the full model's latent cache of the conversation (or None). KEEP logits [T]."""
        m = self.m; x = F.embedding(ids, m.embed) * m.emb_scale
        cos, sin = m.rope(torch.arange(pos0, pos0 + ids.shape[1], device=ids.device))
        for i in range(m.L):
            past = cache[i] if cache is not None else None
            if self.ckpt and torch.is_grad_enabled():       # recompute each block in backward: 62 layers × ~6k tokens
                x = checkpoint(lambda x_, i=i, past=past: m.block(i, x_, cos, sin, past, self.h, self.n, self.lora[i])[0], x, use_reentrant=False)
            else:
                x, _ = m.block(i, x, cos, sin, past, self.h, self.n, self.lora[i])
        h = rms(x, m.norm, m.eps)
        return h if hidden else self.keep(h.float())[0, :, 0]
