import torch
import torch.nn as nn

from d5rl.models.encoder.modules.encoder_decoder import Decoder, Encoder
from d5rl.models.encoder.modules.vq import vq, vq_st


class VQVAE(nn.Module):
    def __init__(
        self,
        chars_embedding_size,
        colors_embedding_size,
        cursor_embedding_size,
        hidden_size,
        num_codes,
        code_size,
    ):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(
            chars_embedding_size,
            colors_embedding_size,
            cursor_embedding_size,
            hidden_size,
            code_size,
        )
        self.codebook = VQEmbedding(num_codes, code_size)
        self.decoder = Decoder(hidden_size, code_size)

    def forward(self, input_state):
        z_e_x = self.encoder(input_state)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        reconstruction = self.decoder(z_q_x_st)
        return reconstruction, z_e_x, z_q_x

    def encode(self, input_state, discrete_embeddings=True):
        z_e_x = self.encoder(input_state)
        if discrete_embeddings:
            return self.codebook(z_e_x)
        else:
            return z_e_x


class VQEmbedding(nn.Module):
    def __init__(self, num_codes, code_size):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, code_size)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(
            self.embedding.weight, dim=0, index=indices
        )
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar
