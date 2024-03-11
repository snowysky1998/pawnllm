import torch
import torch.nn.functional as F
from torch import nn
import math
from dataclasses import dataclass

from einops import rearrange, reduce, einsum, repeat





class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        assert x.ndim == 3
        x = x.float()
        x_squared = x.pow(2)
        x_mean_squared = reduce(x_squared, "b s_q hd -> b s_q 1", "mean")
        x_root_mean_squared = torch.rsqrt(x_mean_squared + self.eps)
        y = x * x_root_mean_squared
        return y.type_as(x) * self.weight


class Transformer(nn.Module):
    def __init__(self, args, freq_cos, freq_sin):
        super().__init__()
        self.args = args
        # ====================== attention =================================
        self.attention_norm = RMSNorm(args.h_q * args.d, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.h_q * args.d, eps=args.norm_eps)
        self.wq = nn.Linear(args.h_q * args.d, args.h_q * args.d, bias=False)
        self.wk = nn.Linear(args.h_q * args.d, args.h_kv * args.d, bias=False)
        self.wv = nn.Linear(args.h_q * args.d, args.h_kv * args.d, bias=False)
        self.wo = nn.Linear(args.h_q * args.d, args.h_q * args.d, bias=False)

        self.freq_cos = freq_cos
        self.freq_sin = freq_sin

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            causal_mask = torch.full((1, 1, args.s, args.s), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            self.causal_mask = nn.Parameter(causal_mask, requires_grad=False)

        self.attention_dropout = nn.Dropout(args.dropout)
        self.attention_proj_dropout = nn.Dropout(args.dropout)

        # ====================== ffn =================================
        ffn_num_features = math.floor(args.h_q * args.d * 8 / 3)
        ffn_num_features = 2 ** (ffn_num_features.bit_length())
        self.w1 = nn.Linear(args.h_q * args.d, ffn_num_features, bias=False)
        self.w2 = nn.Linear(args.h_q * args.d, ffn_num_features, bias=False)
        self.w3 = nn.Linear(ffn_num_features, args.h_q * args.d, bias=False)
        self.ffn_dropout = nn.Dropout(args.dropout)

    def forward(self, h):
        assert h.ndim == 3
        # ============================== norm + attention + residual ============================
        x = self.attention_norm(h)

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = rearrange(xq, "b s (h_q d) -> b s h_q d", d=self.args.d)
        xk = rearrange(xk, "b s (h_kv d) -> b s h_kv d", d=self.args.d)
        xv = rearrange(xv, "b s (h_kv d) -> b s h_kv d", d=self.args.d)

        # rotery embedding (RoPE)
        xq_r, xq_i = rearrange(xq.float(), "b s h_q (d_half two) -> b s h_q d_half two", two=2).unbind(-1)
        xk_r, xk_i = rearrange(xk.float(), "b s h_kv (d_half two) -> b s h_kv d_half two", two=2).unbind(-1)

        freq_cos = rearrange(self.freq_cos, "s d_half -> 1 s 1 d_half")
        freq_sin = rearrange(self.freq_sin, "s d_half -> 1 s 1 d_half")

        xq_out_r = xq_r * freq_cos - xq_i * freq_sin
        xq_out_i = xq_r * freq_sin + xq_i * freq_cos
        xk_out_r = xk_r * freq_cos - xk_i * freq_sin
        xk_out_i = xk_r * freq_sin + xk_i * freq_cos

        xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)
        xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)

        xq = rearrange(xq_out.float(), "b s h_q  d_half two -> b s h_q  (d_half two)").type_as(xq)
        xk = rearrange(xk_out.float(), "b s h_kv d_half two -> b s h_kv (d_half two)").type_as(xk)

        # (grouped) scaled-dot-product-attention
        xk = repeat(xk, "b s h_kv d -> b s (repeat h_kv) d", repeat=self.args.h_q // self.args.h_kv)
        xv = repeat(xv, "b s h_kv d -> b s (repeat h_kv) d", repeat=self.args.h_q // self.args.h_kv)

        xq = rearrange(xq, "b s h d -> b h s d")
        xk = rearrange(xk, "b s h d -> b h s d")
        xv = rearrange(xv, "b s h d -> b h s d")

        if self.flash:
            o = F.scaled_dot_product_attention(
                xq, xk, xv, is_causal=True, dropout_p=self.args.dropout if self.training else 0.0
            )
        else:
            score = einsum(xq, xk, "b h_q s_q d, b h_q s_kv d -> b h_q s_q s_kv") / math.sqrt(self.args.d)
            score = score + self.causal_mask
            score = F.softmax(score.float(), dim=-1).type_as(xq)
            score = self.attention_dropout(score)
            o = einsum(score, xv, "b h_q s_q s_kv, b h_q s_kv d -> b h_q s_q d")

        o = rearrange(o, "b h s d -> b s (h d)")

        o = self.wo(o)

        o = self.attention_proj_dropout(o)

        # attention residual
        h1 = h + o

        # ============================== norm + ffn + residual ==================================
        x = self.ffn_norm(h1)

        # feed forward network (FFN)
        x = F.silu(self.w1(x)) * self.w2(x)  # b s (hd) -> b s num_feature
        x = self.w3(x)  # b s num_feature -> b s (hd)
        o = self.ffn_dropout(x)

        h2 = h1 + o  # b s (hd)

        return h2


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.h_q * args.d)
        self.dropout = nn.Dropout(p=args.dropout)
        self.norm = RMSNorm(args.h_q * args.d, eps=args.norm_eps)
        self.wvocab = nn.Linear(args.h_q * args.d, args.vocab_size, bias=False)

        # cis means "cos(x) + i * sin(x)"
        def precompute_freqs_cis(d, s, theta=10000.0):
            assert d % 2 == 0
            freqs = torch.arange(0, d, 2).float() / d
            freqs = 1.0 / (theta**freqs)
            t = torch.arange(s).float()
            freqs = einsum(t, freqs, "s, d -> s d")
            freqs_cos = torch.cos(freqs)
            freqs_sin = torch.sin(freqs)
            return freqs_cos, freqs_sin

        freq_cos, freq_sin = precompute_freqs_cis(args.d, args.s)
        self.freq_cos = nn.Parameter(freq_cos, requires_grad=False)
        self.freq_sin = nn.Parameter(freq_sin, requires_grad=False)

        self.transformers = torch.nn.ModuleList(
            [Transformer(args, self.freq_cos, self.freq_sin) for _ in range(args.n_layers)]
        )

    def forward(self, tokens, targets=None):
        assert tokens.ndim == 2
        assert tokens.dtype == torch.int64
        assert targets.ndim == 2 if targets is not None else True
        assert targets.dtype == torch.int64 if targets is not None else True

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for transformer in self.transformers:
            h = transformer(h)

        h = self.norm(h)  # b s (hd)

        if self.train:
            # training: for some desired targets calculate the loss
            logits = self.wvocab(h)
            logits = rearrange(logits, "b s vocab_size -> (b s) vocab_size")
            targets = rearrange(targets, "b s -> (b s)")
            last_loss = F.cross_entropy(logits, targets, ignore_index=-1)
            return logits, last_loss
        else:
            # inference: only forward the output on the very last position
            logits = self.wvocab(h[:, -1:, :])
            return logits
