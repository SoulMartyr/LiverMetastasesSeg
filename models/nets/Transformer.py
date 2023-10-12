import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, qkv_bias: bool = False, dropout_rate: float = 0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.dropout_rate = dropout_rate
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, length, channels = x.shape
        query = self.query(x).reshape(batch_size, length, self.num_heads, channels // self.num_heads).permute(0, 2, 1,
                                                                                                              3)
        key = self.key(x).reshape(batch_size, length, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        value = self.value(x).reshape(batch_size, length, self.num_heads, channels // self.num_heads).permute(0, 2, 1,
                                                                                                              3)

        attn_score = (query @ key.transpose(-2, -1)) * self.scale

        attn_score = attn_score.softmax(dim=-1)

        out = (attn_score @ value).transpose(1, 2).reshape(batch_size, length, channels)

        out = self.proj(out)
        if self.dropout_rate > 0:
            out = self.proj_drop(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, dropout_rate: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.ffn(x)


class PositionEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int):
        super(PositionEncoding, self).__init__()
        if not embed_dim % 2 == 0:
            raise ValueError("Dimension Should Be Divided By 2")

        self.max_length = max_length

        w = torch.exp(torch.arange(0., embed_dim, 2) * (-torch.log(10000.0) / embed_dim))
        t = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, embed_dim)
        pos[0, :, 0::2] = torch.sin(w * t)
        pos[0, :, 1::2] = torch.cos(w * t)
        self.register_buffer('pos', pos)

    def forward(self, x):
        _, t, _ = x.shape
        x = x + self.pos[:, :t]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, seq_length: int):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))  # 8x

    def forward(self, x):
        position_embeddings = self.position_embeddings
        return x + position_embeddings


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, heads: int, mlp_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                Residual(PreNormDrop(embed_dim, SelfAttention(embed_dim, heads=heads, dropout_rate=dropout_rate),
                                     dropout_rate)),
                Residual(PreNorm(embed_dim, FeedForward(embed_dim, mlp_dim, dropout_rate))),
            ])
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    a = torch.randn([5, 64, 128])
    sa = SelfAttention(dim=128)
    print(sa(a).shape)
