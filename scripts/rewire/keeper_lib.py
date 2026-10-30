"""10% WIDTH KEEPER inside an existing LLM (2026-10-29, user: "existing open models like qwen could be rewired to have this
keeper functionality out of the box: a subset of 10% parameters could be chosen (post-training matryoshka rewiring) …
to run keep heads on top of them cheaply"; "the 10% width concept only, no depth, no recompute").

Three pieces, for any Llama-layout decoder (q/k/v/o + optional q_norm/k_norm, gate/up/down; GQA):
  1. importance(): Minitron-style activation scores on calibration text — per KV GROUP (the norm of its query heads'
     outputs at the o_proj input) and per MLP NEURON (mean |silu(gate)·up| at the down_proj input), layer by layer.
  2. rewire(): a FUNCTION-PRESERVING permutation of every layer so the most important groups / neurons come first.
     The full model is unchanged (checked by the CLI); the keeper is then the LEADING SLICES of the same tensors —
     no copy, no extra weights apart from a chopper-mode LoRA and the KEEP head.
  3. Keeper: the width-pruned sub-network (all layers, hidden size kept, g groups + n neurons per layer). It never
     recomputes the conversation: its attention reads the FULL model's cached keys/values of the conversation for the
     kept groups, plus its own keys/values of the snippets. Snippets are read twice (double prefill); KEEP logits come
     from the second copy. The full model never prefills the snippets, so "rollback" is just dropping the keeper's
     tensors.
Hidden size is NOT pruned: that keeps the cached keys/values shape-compatible with the keeper's queries."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.bias import causal_lower_right


def geom(cfg):
    H = cfg.hidden_size; hd = getattr(cfg, "head_dim", None) or H // cfg.num_attention_heads
    G = cfg.num_key_value_heads; qpg = cfg.num_attention_heads // G
    return H, hd, G, qpg, cfg.intermediate_size


def sizes(cfg, frac):
    """kept KV groups g and MLP neurons n per layer so the keeper holds ≈ frac of each layer's parameters"""
    H, hd, G, qpg, I = geom(cfg)
    per_group = H * qpg * hd * 2 + H * hd * 2               # q rows + o cols + k rows + v rows of one group
    layer = per_group * G + 3 * H * I
    g = max(1, round(frac * G)); n = int(max(64, min(I, (frac * layer - g * per_group) / (3 * H))))
    n = max(64, n // 64 * 64)
    return g, n, (g * per_group + 3 * H * n) / layer


@torch.no_grad()
def importance(model, batches):
    """batches: iterable of input_ids [1, T] → per layer (group score [G], neuron score [I])"""
    cfg = model.config; H, hd, G, qpg, I = geom(cfg); L = model.model.layers
    gs = [torch.zeros(G, dtype=torch.float64) for _ in L]; ns = [torch.zeros(I, dtype=torch.float64) for _ in L]
    cnt = [0]; hooks = []
    for i, l in enumerate(L):
        def ho(mod, a, i=i):
            x = a[0].float().view(-1, G, qpg * hd)          # o_proj input: heads of group j are contiguous
            gs[i] += x.norm(dim=-1).sum(0).double().cpu()
        def hd_(mod, a, i=i):
            ns[i] += a[0].float().abs().reshape(-1, I).sum(0).double().cpu()
        hooks += [l.self_attn.o_proj.register_forward_pre_hook(ho), l.mlp.down_proj.register_forward_pre_hook(hd_)]
    for ids in batches:
        model(input_ids=ids.to(model.device)); cnt[0] += ids.numel()
    for h in hooks:
        h.remove()
    return [(g / cnt[0], n / cnt[0]) for g, n in zip(gs, ns)]


@torch.no_grad()
def rewire(model, perms):
    """perms: per layer (group order [G], neuron order [I]) — most important first. In place; exact up to float sums."""
    H, hd, G, qpg, I = geom(model.config)
    for l, (pg, pn) in zip(model.model.layers, perms):
        a, m = l.self_attn, l.mlp
        qi = torch.cat([torch.arange(j * qpg * hd, (j + 1) * qpg * hd) for j in pg.tolist()])
        ki = torch.cat([torch.arange(j * hd, (j + 1) * hd) for j in pg.tolist()])
        a.q_proj.weight.copy_(a.q_proj.weight[qi]); a.o_proj.weight.copy_(a.o_proj.weight[:, qi])
        a.k_proj.weight.copy_(a.k_proj.weight[ki]); a.v_proj.weight.copy_(a.v_proj.weight[ki])
        for p_, idx in ((a.q_proj, qi), (a.k_proj, ki), (a.v_proj, ki)):
            if p_.bias is not None:
                p_.bias.copy_(p_.bias[idx])
        m.gate_proj.weight.copy_(m.gate_proj.weight[pn]); m.up_proj.weight.copy_(m.up_proj.weight[pn])
        m.down_proj.weight.copy_(m.down_proj.weight[:, pn])


def perms_from(imp):
    return [(g.argsort(descending=True), n.argsort(descending=True)) for g, n in imp]


class LoRA(nn.Module):
    def __init__(self, din, dout, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, din) / math.sqrt(din)); self.B = nn.Parameter(torch.zeros(dout, r))

    def forward(self, x):
        return (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)


def rms(x, w, eps):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope(x, cos, sin):
    h = x.shape[-1] // 2; rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


class Keeper(nn.Module):
    """the width slice of a REWIRED model + chopper-mode LoRA + KEEP head; the base weights stay frozen and shared"""

    def __init__(self, model, g, n, r=32, ckpt=True):
        super().__init__()
        self.m = [model]                                    # not a submodule: its weights are not ours to train
        H, hd, G, qpg, I = geom(model.config); self.H, self.hd, self.qpg, self.g, self.n, self.ckpt = H, hd, qpg, g, n, ckpt
        self.eps = model.config.rms_norm_eps; nq = g * qpg * hd; nk = g * hd
        dims = {"q": (H, nq), "k": (H, nk), "v": (H, nk), "o": (nq, H), "gate": (H, n), "up": (H, n), "down": (n, H)}
        self.lora = nn.ModuleList([nn.ModuleDict({k: LoRA(a, b, r) for k, (a, b) in dims.items()})
                                   for _ in model.model.layers])
        self.keep = nn.Linear(H, 1)

    def layer(self, i, x, cos, sin, ck, cv):
        l = self.m[0].model.layers[i]; a, mm, lo = l.self_attn, l.mlp, self.lora[i]
        g, qpg, hd, n = self.g, self.qpg, self.hd, self.n; nq, nk = g * qpg * hd, g * hd; T = x.shape[1]
        h = rms(x, l.input_layernorm.weight, self.eps)
        q = (F.linear(h, a.q_proj.weight[:nq]) + lo["q"](h)).view(1, T, g * qpg, hd)
        k = (F.linear(h, a.k_proj.weight[:nk]) + lo["k"](h)).view(1, T, g, hd)
        v = (F.linear(h, a.v_proj.weight[:nk]) + lo["v"](h)).view(1, T, g, hd).transpose(1, 2)
        if hasattr(a, "q_norm"):
            q, k = rms(q, a.q_norm.weight, self.eps), rms(k, a.k_norm.weight, self.eps)
        q, k = rope(q.transpose(1, 2), cos, sin), rope(k.transpose(1, 2), cos, sin)
        if ck is not None:                                  # the FULL model's conversation keys/values, kept groups
            k, v = torch.cat([ck[:, :g].to(k.dtype), k], 2), torch.cat([cv[:, :g].to(v.dtype), v], 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(T, k.shape[2]), enable_gqa=True)
        o = o.transpose(1, 2).reshape(1, T, nq)
        x = x + F.linear(o, a.o_proj.weight[:, :nq]) + lo["o"](o)
        h = rms(x, l.post_attention_layernorm.weight, self.eps)
        u = F.silu(F.linear(h, mm.gate_proj.weight[:n]) + lo["gate"](h)) * (F.linear(h, mm.up_proj.weight[:n]) + lo["up"](h))
        return x + F.linear(u, mm.down_proj.weight[:, :n]) + lo["down"](u)

    def forward(self, ids, cache_kv, pos0):
        """ids [1, T] snippet tokens (both copies); cache_kv: per layer (k, v) [1, G, Tc, hd] of the conversation from
        the full model (or None); pos0 = Tc. Returns KEEP logits [T]."""
        base = self.m[0].model
        x = base.embed_tokens(ids)
        pos = torch.arange(pos0, pos0 + ids.shape[1], device=ids.device)[None]
        cos, sin = base.rotary_emb(x, pos); cos, sin = cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)
        for i in range(len(base.layers)):
            ck, cv = cache_kv[i] if cache_kv is not None else (None, None)
            if self.ckpt and self.training:
                x = checkpoint(self.layer, i, x, cos, sin, ck, cv, use_reentrant=False)
            else:
                x = self.layer(i, x, cos, sin, ck, cv)
        return self.keep(rms(x, base.norm.weight, self.eps).float())[0, :, 0]


@torch.no_grad()
def conv_cache(model, ids):
    """prefill the conversation with the FULL model; per layer (k, v) after k_norm + RoPE, [1, G, Tc, hd]"""
    c = model(input_ids=ids, use_cache=True).past_key_values
    if hasattr(c, "layers"):
        return [(ly.keys, ly.values) for ly in c.layers]
    return [(c[i][0], c[i][1]) for i in range(len(model.model.layers))]
