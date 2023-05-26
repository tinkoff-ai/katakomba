import pyrallis
from dataclasses import dataclass, asdict

import random
import wandb
import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.vector import AsyncVectorEnv
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm, trange
import numpy as np

from copy import deepcopy
from typing import Optional, Dict, Tuple, Any

from katakomba.env import NetHackChallenge, OfflineNetHackChallengeWrapper
from katakomba.nn.chaotic_dwarf import TopLineEncoder, BottomLinesEncoder, ScreenEncoder
from katakomba.utils.render import SCREEN_SHAPE, render_screen_image
from katakomba.utils.datasets import SequentialBuffer
from katakomba.utils.misc import Timeit, StatMean

LSTM_HIDDEN = Tuple[torch.Tensor, torch.Tensor]
UPDATE_INFO = Dict[str, Any]

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    character: str = "mon-hum-neu"
    data_mode: str = "compressed"
    # Wandb logging
    project: str = "NetHack"
    group: str = "small_scale_iql"
    name: str = "iql"
    version: int = 0
    # Model
    rnn_hidden_dim: int = 2048
    rnn_layers: int = 2
    use_prev_action: bool = True
    rnn_dropout: float = 0.0
    clip_range: float = 10.0
    tau: float = 0.005
    gamma: float = 0.999
    temperature: float = 1.0
    expectile_tau: float = 0.8
    # Training
    update_steps: int = 500_000
    batch_size: int = 64
    seq_len: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None
    checkpoints_path: Optional[str] = None
    eval_every: int = 10_000
    eval_episodes: int = 50
    eval_processes: int = 14
    eval_seed: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.group}-v{str(self.version)}"
        self.name = f"{self.name}-{self.character}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.group, self.name)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def filter_wd_params(model: nn.Module):
    no_decay, decay = [], []
    for name, param in model.named_parameters():
        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    assert len(no_decay) + len(decay) == len(list(model.parameters()))
    return no_decay, decay


def dict_to_tensor(data, device):
    return {k: torch.as_tensor(v, dtype=torch.float, device=device) for k, v in data.items()}


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Encoder(nn.Module):
    def __init__(self, rnn_hidden_dim=512, rnn_layers=1, rnn_dropout=0.0, out_dim=512, use_prev_action=True):
        super().__init__()
        self.out_dim = out_dim
        self.use_prev_action = use_prev_action
        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0

        # Encoders
        self.topline_encoder = torch.jit.script(TopLineEncoder())
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder())

        screen_shape = (SCREEN_SHAPE[1], SCREEN_SHAPE[2])
        self.screen_encoder = torch.jit.script(ScreenEncoder(screen_shape))

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )
        # Policy
        self.rnn = nn.LSTM(
            self.h_dim,
            rnn_hidden_dim,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True
        )
        self.head = nn.Linear(rnn_hidden_dim, out_dim)

    def forward(self, inputs, state=None):
        # [batch_size, seq_len, ...]
        B, T, C, H, W = inputs["screen_image"].shape
        topline = inputs["tty_chars"][..., 0, :]
        bottom_line = inputs["tty_chars"][..., -2:, :]

        encoded_state = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.bottomline_encoder(
                bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.screen_encoder(
                inputs["screen_image"]
                .float(memory_format=torch.contiguous_format)
                .view(T * B, C, H, W)
            ),
        ]
        if self.use_prev_action:
            encoded_state.append(
                F.one_hot(inputs["prev_actions"], self.num_actions).view(T * B, -1)
            )

        encoded_state = torch.cat(encoded_state, dim=1)
        core_output, new_state = self.rnn(encoded_state.view(B, T, -1), state)
        out = self.head(core_output).view(B, T, self.out_dim)

        return out, new_state


class Critic(nn.Module):
    def __init__(self, action_dim, rnn_hidden_dim, rnn_layers, rnn_dropout, out_dim, use_prev_action):
        super().__init__()
        self.encoder = Encoder(
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            out_dim=out_dim,
            use_prev_action=use_prev_action
        )
        self.q1 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, action_dim)
        )
        self.q2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, action_dim)
        )
        self.vf = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

    def forward(self, inputs, state=None):
        out, new_state = self.encoder(inputs, state=state)
        q1, q2, vf = self.q1(out), self.q2(out), self.vf(out)
        return q1, q2, vf


class Actor(nn.Module):
    def __init__(self, action_dim, rnn_hidden_dim, rnn_layers, rnn_dropout, out_dim, use_prev_action):
        super().__init__()
        self.encoder = Encoder(
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            out_dim=out_dim,
            use_prev_action=use_prev_action
        )
        self.policy_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, action_dim)
        )

    def forward(self, inputs, state=None):
        out, new_state = self.encoder(inputs, state=state)
        logits = self.policy_head(out)
        return logits, new_state


def iql_loss(
        actor: Actor,
        critic: Critic,
        target_critic: Critic,
        obs: Dict[str, torch.Tensor],
        next_obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rnn_states: LSTM_HIDDEN,
        target_rnn_states: LSTM_HIDDEN,
        gamma: float,
        temperature: float,
        expectile_tau: float,
):
    pass