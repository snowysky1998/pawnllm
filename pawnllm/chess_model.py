"""Small Chessformer-inspired encoder. WDL classes are loss, draw, win."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

ACTION_SIZE = 4096 + 8 * 8 * 4
FORMAT_VERSION = 1


@dataclass
class ChessConfig:
    history: int = 8
    dim: int = 256
    layers: int = 8
    heads: int = 8
    gab_size: int = 64
    variant: str = 'pilot'
    rating_conditioned: bool = False

    def __post_init__(self):
        if min(self.history, self.dim, self.layers, self.heads, self.gab_size) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.dim % self.heads:
            raise ValueError('dim must be divisible by heads')
        if self.variant not in ('pilot', 'enhanced'):
            raise ValueError('Unknown chess model variant')


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.dim // c.heads
        self.norm = nn.LayerNorm(c.dim)
        self.qkv = nn.Linear(c.dim, 3 * c.dim)
        self.proj = nn.Linear(c.dim, c.dim)
        self.gab = nn.Sequential(nn.Linear(c.dim, c.gab_size), nn.GELU(),
                                 nn.LayerNorm(c.gab_size),
                                 nn.Linear(c.gab_size, c.heads * c.gab_size),
                                 nn.GELU(), nn.LayerNorm(c.heads * c.gab_size))
        self.ffn = nn.Sequential(nn.LayerNorm(c.dim), nn.Linear(c.dim, 2 * c.dim),
                                 nn.GELU(), nn.Linear(2 * c.dim, c.dim))

    def forward(self, x, templates):
        b, s, d = x.shape
        y = self.norm(x)
        q, k, v = self.qkv(y).reshape(b, s, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        mixture = self.gab(y.mean(1)).reshape(b, self.heads, -1)
        bias = (mixture @ templates.to(mixture.dtype)).reshape(b, self.heads, s, s)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(y.transpose(1, 2).reshape(b, s, d))
        return x + self.ffn(x)


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate_up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(2 * dim, dim)

    def forward(self, x):
        gate, up = self.gate_up(self.norm(x)).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class EnhancedBlock(Block):
    """GAB + learned 2-D relative geometry, Q/K normalization and gated MLPs."""
    def __init__(self, c):
        super().__init__(c)
        self.ffn = SwiGLU(c.dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.relative = nn.Parameter(torch.zeros(c.heads, 225))
        squares = torch.arange(64)
        rank, file = squares // 8, squares % 8
        offsets = (rank[None, :] - rank[:, None] + 7) * 15 + file[None, :] - file[:, None] + 7
        self.register_buffer('offsets', offsets, persistent=False)

    def forward(self, x, templates):
        b, s, d = x.shape
        y = self.norm(x)
        q, k, v = self.qkv(y).reshape(b, s, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        mixture = self.gab(y.mean(1)).reshape(b, self.heads, -1)
        bias = (mixture @ templates.to(mixture.dtype)).reshape(b, self.heads, s, s)
        bias = bias + self.relative[:, self.offsets].to(bias.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(y.transpose(1, 2).reshape(b, s, d))
        return x + self.ffn(x)


class ChessModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ChessConfig()
        c = self.config
        # Twelve piece planes per historical board, six global rule features,
        # and a square-local en passant feature.
        self.input = nn.Linear(c.history * 12 + 7, c.dim)
        if c.rating_conditioned:
            self.rating_embedding = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, c.dim))
        self.templates = nn.Parameter(torch.randn(c.gab_size, 4096) * 0.02)
        block_type = EnhancedBlock if c.variant == 'enhanced' else Block
        self.blocks = nn.ModuleList(block_type(c) for _ in range(c.layers))
        self.norm = nn.LayerNorm(c.dim)
        self.from_proj = nn.Linear(c.dim, c.dim)
        self.to_proj = nn.Linear(c.dim, c.dim)
        self.promotion = nn.Linear(c.dim, 4)
        self.value = nn.Sequential(nn.Linear(c.dim, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, pieces, rules, ratings=None):
        # pieces: B,H,64 uint8; rules: B,7 float (castles x4, clock, repetition, ep).
        b = pieces.shape[0]
        planes = F.one_hot(pieces.long(), 13)[..., 1:].permute(0, 2, 1, 3).reshape(b, 64, -1).float()
        globals_ = rules[:, None, :6].expand(-1, 64, -1)
        ep = (torch.arange(64, device=pieces.device)[None, :] == rules[:, 6:7]).float()[..., None]
        x = self.input(torch.cat((planes, globals_, ep), dim=-1))
        if self.config.rating_conditioned:
            if ratings is None or ratings.shape != (b, 2):
                raise ValueError('Expected ratings shaped (batch, 2): side to move, opponent')
            x = x + self.rating_embedding((ratings.float() - 1600) / 1000)[:, None, :]
        for block in self.blocks:
            x = block(x, self.templates)
        x = self.norm(x)
        q, k = self.from_proj(x), self.to_proj(x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.config.dim)
        # Canonical promotions go from rank 7 to rank 8. All four pieces have
        # distinct actions, including queens; ordinary from-to promotion is masked.
        promos = scores[:, 48:56, 56:64, None] + self.promotion(x[:, 56:64])[:, None, :, :]
        policy = torch.cat((scores.flatten(1), promos.flatten(1)), dim=1)
        return policy, self.value(x.mean(1))


def masked_policy(logits, legal):
    # legal contains action indices padded with -1; use an extra sentinel column.
    mask = torch.zeros((len(logits), ACTION_SIZE + 1), dtype=torch.bool, device=logits.device)
    mask.scatter_(1, legal.long().masked_fill(legal < 0, ACTION_SIZE), True)
    return logits.float().masked_fill(~mask[:, :ACTION_SIZE], -torch.inf)
