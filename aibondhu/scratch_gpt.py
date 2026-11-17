"""AIBondhu Scratch GPT: a small decoder-only Transformer initialized from random weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class CausalAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.dropout = dropout
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(z):
            return z.view(b, t, self.heads, c // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))

class TransformerBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalAttention(width, heads, dropout)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(),
                                 nn.Linear(4*width, width), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class ScratchGPT(nn.Module):
    def __init__(self, vocab_size, width=256, layers=6, heads=8, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(block_size, width)
        self.blocks = nn.Sequential(*[TransformerBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, ids, targets=None):
        b, t = ids.shape
        if t > self.block_size:
            raise ValueError("input exceeds block_size")
        pos = torch.arange(t, device=ids.device)
        x = self.token(ids) + self.position(pos)[None, :, :]
        logits = self.lm_head(self.norm(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss
