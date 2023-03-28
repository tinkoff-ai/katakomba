import torch
import torch.nn as nn
from .pos_emb import generate_fourier_pos_encoding


# standard transformer block, but without a dropout
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_d = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x, q):
        q_norm = self.norm_q(q)
        x_norm = self.norm_d(x)

        attn = self.attn(query=q_norm, key=x_norm, value=x_norm, need_weights=False)[0]

        out = q + attn
        out = out + self.mlp(out)
        return out


class LatentTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.transformers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, l):
        for transformer in self.transformers:
            l = transformer(x=l, q=l)
        return l


class PerceiverBlock(nn.Module):
    def __init__(self, hidden_dim, cross_heads, latent_heads, latent_layers):
        super().__init__()
        self.cross_attention = TransformerBlock(hidden_dim, cross_heads)
        self.latent_transformer = LatentTransformer(
            hidden_dim, latent_heads, latent_layers
        )

    def forward(self, x, l):
        l = self.cross_attention(x, l)
        l = self.latent_transformer(l)
        return l


class Perceiver(nn.Module):
    def __init__(
            self,
            img_shape=(24, 80),
            input_dim=128,
            hidden_dim=128,
            latent_len=128,
            out_dim=512,
            cross_trans_heads=1,
            latent_trans_heads=4,
            latent_trans_layers=1,
            depth=6,
            num_bands=4,
            share_weights=False,
    ):
        super().__init__()
        assert len(img_shape) == 2
        self.register_buffer("pos_embed", generate_fourier_pos_encoding(img_shape, num_bands))

        # input_dim + dims from pos embedding
        input_dim = input_dim + (4 * num_bands + 2)
        self.to_hidden = nn.Linear(input_dim, hidden_dim)

        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.zeros((1, latent_len, hidden_dim)), mean=0, std=0.02, a=-2, b=2)
        )
        self.perceiver_blocks = nn.ModuleList([
            PerceiverBlock(
                hidden_dim,
                cross_trans_heads,
                latent_trans_heads,
                latent_trans_layers
            )
            # we not share 1 block (as in paper), while all others are shared
            for _ in range(depth if not share_weights else 2)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)
        self.depth = depth
        self.share_weights = share_weights

    def forward(self, x):
        # x: [seq_len, batch_size, emb_dim]
        batch_size, seq_len, hidden_dim = x.shape

        x = torch.cat([x, self.pos_embed.expand(batch_size, -1, -1)], dim=-1)
        x = self.to_hidden(x)

        latent = self.latent.expand(batch_size, -1, -1)
        if self.share_weights:
            # first block is always not shared (as in paper)
            latent = self.perceiver_blocks[0](x, latent)
            for d in range(self.depth - 1):
                # reuse block multiple times as in RNN
                latent = self.perceiver_blocks[1](x, latent)
        else:
            for block in self.perceiver_blocks:
                latent = block(x, latent)

        # mean across len
        latent = latent.mean(1)
        out = self.head(latent)

        return out


if __name__ == "__main__":
    model = Perceiver(
        input_dim=32,
        hidden_dim=32,
        latent_len=32,
        out_dim=32,
        latent_trans_layers=2,
        depth=6,
        share_weights=True
    )
    x = torch.randn(16, 24 * 80, 32)
    print(sum(p.numel() for p in model.parameters()))
    print(model(x).shape)
    print(len(model.perceiver_blocks))