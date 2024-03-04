import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass
from einops import rearrange, reduce, einsum

# # self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

@dataclass
class ModelArgs:
    # vocab size
    vocab_size: int

    # hidden dim per head
    d: int = 128
    # number of query heads
    h_q: int = 32
    # number of key/value heads
    h_kv: int = 32
    # number of heads x hidden dim per head = hidden dim
    hd: int = 4096
    # maximum sequence length
    s: int = 2048

    # number of layers
    n_layers: int = 32

    # multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    dropout: float = 0.125

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        x_squared = x.pow(2)
        x_mean_squared = reduce(x_squared, "b s_q hd -> b s_q 1", "mean")
        x_root_mean_squared = torch.rsqrt(x_mean_squared + self.eps)
        y = x * x_root_mean_squared
        return y.type_as(x) * self.weight

# cis means "cos(x) + i * sine(x)"
def precompute_freqs_cis(d, s, theta = 10000.0):
    assert d % 2 == 0
    freqs = torch.arange(0, d, 2).float() / d
    freqs = 1.0 / (theta ** freqs)
    t = torch.arange(s).float()
    freqs = einsum(t, freqs, "s, d -> s d")
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hd)
        self.dropout = nn.Dropout(p=args.dropout)

        self.attention_norm = RMSNorm(args.hd, eps=args.norm_eps)
        self.wq = nn.Linear(args.h_q * args.d, args.h_q * args.d, bias=False)
        self.wk = nn.Linear(args.h_q * args.d, args.h_kv * args.d, bias=False)
        self.wv = nn.Linear(args.h_q * args.d, args.h_kv * args.d, bias=False)

        freq_cos, freq_sin = precompute_freqs_cis(args.d, args.s)
        self.freq_cos = nn.Parameter(freq_cos, requires_grad=False)
        self.freq_sin = nn.Parameter(freq_sin, requires_grad=False)

    def forward(self, tokens, targest=None):
        b, seqlen = tokens.size()

        print(f"{tokens.size()=}")
        h1 = self.tok_embeddings(tokens)
        h1 = self.dropout(h1)

        # transformer block
        x = self.attention_norm(h1)

        # attention_forward()
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = rearrange(xq, "b seqlen (h_q  d) -> b seqlen h_q  d", h_q=self.args.h_q,   d=self.args.d)
        xk = rearrange(xk, "b seqlen (h_kv d) -> b seqlen h_kv d", h_kv=self.args.h_kv, d=self.args.d)
        xv = rearrange(xv, "b seqlen (h_kv d) -> b seqlen h_kv d", h_kv=self.args.h_kv, d=self.args.d)

        # apply_rotary_emb
        xq_r, xq_i = rearrange(xq.float(), "b s h_q  (d_half two) -> b s h_q  d_half two", two=2).unbind(-1)
        xk_r, xk_i = rearrange(xk.float(), "b s h_kv (d_half two) -> b s h_kv d_half two", two=2).unbind(-1)

        # # rehape_for_broadcast()
        # def reshape_for_broadcast(freqs_cis, x):
        #     ndim = x.ndim
        #     assert 0 <= 1 < ndim
        #     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        #     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        #     return freqs_cis.view(shape)

        # freqs_cos = reshape_for_broadcast(self.freq_cos, xq_r)
        # freqs_sin = reshape_for_broadcast(self.freq_sin, xq_r)

        # xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        # xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        # xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        # xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos


        # print(f"{xq_r.size()=}")
        # print(f"{xq_r.size()=}")

        # def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        #     print(freqs_cis.size())
        #     ndim = x.ndim
        #     assert 0 <= 1 < ndim
        #     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        #     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        #     return freqs_cis.view(shape)

        # freqs_cos = reshape_for_broadcast(self.freq_cos, xq_r)
        # freqs_sin = reshape_for_broadcast(self.freq_sin, xq_r)

        print(f"{xq.size()=}")
        print(f"{xk.size()=}")
        print(f"{xv.size()=}")

        # from einops import rearrange
        # rearrange(xq, "")

        # position embedding
        #sine cosine

        return h1

    # def forward(self, tokens: torch.Tensor, targets=None):
    #     _bsz, seqlen = tokens.shape
    #     h = self.tok_embeddings(tokens)
    #     h = self.dropout(h)
    #     freqs_cos = self.freqs_cos[:seqlen]
    #     freqs_sin = self.freqs_sin[:seqlen]

    #     for layer in self.layers:
    #         h = layer(h, freqs_cos, freqs_sin)
    #     h = self.norm(h)

    #     if targets is not None:
    #         # if we are given some desired targets also calculate the loss
    #         logits = self.output(h)
    #         self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    #     else:
    #         # inference-time mini-optimization: only forward the output on the very last position
    #         logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
    #         self.last_loss = None

    #     return logits


