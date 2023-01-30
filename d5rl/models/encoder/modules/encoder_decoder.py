import torch
import torch.nn as nn

from d5rl.utils.observations import num_chars, num_colors


class Encoder(nn.Module):
    def __init__(
        self,
        chars_embedding_size,
        colors_embedding_size,
        cursor_embedding_size,
        hidden_size,
        code_size,
    ):
        super(Encoder, self).__init__()

        self.chars_embeddings = nn.Embedding(num_chars(), chars_embedding_size)
        self.colors_embeddings = nn.Embedding(num_colors(), colors_embedding_size)
        self.cursor_embeddings = nn.Embedding(2, cursor_embedding_size)

        input_size = (
            chars_embedding_size + colors_embedding_size + cursor_embedding_size
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            ResNetBlock(hidden_size),
            ResNetBlock(hidden_size),
            ResNetBlock(hidden_size),
            nn.Conv2d(hidden_size, code_size, 1),
        )

    def forward(self, input_state):
        chars, colors, cursor = torch.split(input_state, 1, -1)

        chars = self.chars_embeddings(chars.squeeze(-1).int())
        colors = self.colors_embeddings(colors.squeeze(-1).int())
        cursor = self.cursor_embeddings(torch.sign(cursor.squeeze(-1)).int())

        input_state = torch.concat([chars, colors, cursor], -1).permute(0, -1, 1, 2)
        return self.encoder(input_state)


class Decoder(nn.Module):
    def __init__(self, hidden_size, code_size):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(code_size, hidden_size, 1),
            ResNetBlock(hidden_size),
            ResNetBlock(hidden_size),
            ResNetBlock(hidden_size),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, num_chars() + num_colors() + 1, 3, 1, 1),
        )

    def forward(self, z_q_x):
        return self.decoder(z_q_x).permute(0, 2, 3, 1)


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)
