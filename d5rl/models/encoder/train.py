import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pyrallis
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import wandb
from nle.nethack import tty_render
from torch.utils.data import DataLoader
from tqdm import tqdm

from d5rl.datasets import AutoAscendDatasetBuilder
from d5rl.datasets.state_autoascend import StateAutoAscendTTYDataset
from d5rl.models.encoder.modules.vqvae import VQVAE
from d5rl.utils.observations import num_chars, num_colors


@dataclass
class TrainConfig:
    data_path: str = "/app/data/nld-aa/nld-aa/nle_data"

    seed: int = 0
    log_freq: int = 50

    lr: float = 5e-4
    gamma: float = 5.0
    batch_size: int = 256
    yield_freq: int = 10
    n_train_steps: int = int(1e6)
    checkpoints_path: Optional[str] = None

    chars_embedding_size: int = 32
    colors_embedding_size: int = 8
    cursor_embedding_size: int = 8
    hidden_size: int = 512
    num_codes: int = 4096
    code_size: int = 64

    project: str = "NeuralNetHack-VQVAE"


class FocalLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, gamma, alpha=None, ignore_index=-100):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction="mean")
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 5,
    reduction: str = "none",
):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def train(config: TrainConfig, dataloader, model, optimizer, scaler):
    focal_loss = FocalLoss(config.gamma)
    for step, batch in tqdm(enumerate(dataloader)):
        batch = batch.cuda().squeeze(0).long()

        optimizer.zero_grad()
        with amp.autocast(dtype=torch.float16):
            reconstruction, z_e_x, z_q_x = model(batch)

            reconstruction_chars, reconstruction_colors, reconstruction_cursor = (
                reconstruction[:, :, :, : num_chars()],
                reconstruction[:, :, :, num_chars() : num_colors() + num_chars()],
                reconstruction[:, :, :, -1],
            )

            target_chars, target_colors, target_cursor = (
                batch[:, :, :, 0],
                batch[:, :, :, 1],
                torch.sign(batch[:, :, :, 2]).float(),
            )

            chars_loss = focal_loss(
                reconstruction_chars.flatten(0, 2),
                target_chars.flatten(),
            )
            colors_loss = focal_loss(
                reconstruction_colors.flatten(0, 2),
                target_colors.flatten(),
            )
            cursor_loss = sigmoid_focal_loss(
                reconstruction_cursor.flatten(),
                target_cursor.flatten(),
                0,
                config.gamma,
                "mean",
            )

            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

            loss = chars_loss + colors_loss + cursor_loss + loss_vq + loss_commit
        scaler.scale(loss).backward()

        if step % config.log_freq == 0:
            wandb.log(
                {
                    "losses/loss": loss.item(),
                    "losses/chars": chars_loss.item(),
                    "losses/colors": colors_loss.item(),
                    "losses/cursor": cursor_loss.item(),
                    "losses/vq": loss_vq.item(),
                    "losses/commit": loss_commit.item(),
                }
            )

        if step % (config.log_freq * 5) == 0:
            print()
            print("%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("INPUT")
            print()
            chars = batch[0, :, :, 0].cpu().numpy()
            colors = batch[0, :, :, 1].cpu().numpy()
            print(tty_render(chars, colors))
            print("PREDICTION")
            chars = reconstruction_chars[0, :, :].argmax(-1).cpu().numpy()
            colors = reconstruction_colors[0, :, :].argmax(-1).cpu().numpy()
            print(tty_render(chars, colors))

        scaler.step(optimizer)
        scaler.update()

        if step % 5000 == 0:
            torch.save(model.state_dict(), f"/app/data/model_{step}.pt")

        if step == config.n_train_steps:
            break


@pyrallis.wrap()
def main(config: TrainConfig):
    wandb.init(
        project=config.project,
        entity="tlab",
    )

    set_seed(config.seed, False)

    builder = AutoAscendDatasetBuilder(config.data_path)
    dataset = builder.build(
        config.batch_size, 1000, StateAutoAscendTTYDataset, yield_freq=config.yield_freq
    )
    loader = DataLoader(dataset, pin_memory=True, num_workers=32, prefetch_factor=4)

    scaler = amp.GradScaler()

    model = VQVAE(
        config.chars_embedding_size,
        config.colors_embedding_size,
        config.cursor_embedding_size,
        config.hidden_size,
        config.num_codes,
        config.code_size,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train(config, loader, model, optimizer, scaler)


if __name__ == "__main__":
    main()
