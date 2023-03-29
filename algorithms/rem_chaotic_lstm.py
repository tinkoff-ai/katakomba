import pyrallis
from dataclasses import dataclass, asdict

import time
import random
import wandb
import sys
import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm, trange
from collections import defaultdict
import numpy as np

from copy import deepcopy
from typing import Optional
from katakomba.tasks import make_task_builder
from katakomba.utils.roles import Alignment, Race, Role, Sex
from katakomba.datasets import SARSChaoticAutoAscendTTYDataset
from katakomba.nn.chaotic_dwarf import TopLineEncoder, BottomLinesEncoder, ScreenEncoder
from katakomba.utils.render import SCREEN_SHAPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    env: str = "NetHackScore-v0-ttyimg-bot-v0"
    data_path: str = "data/nle_data"
    db_path: str = "ttyrecs.db"
    # Wandb logging
    project: str = "NeuralNetHack"
    group: str = "REM"
    name: str = "REM"
    version: str = "v0"
    # Model
    use_prev_action: bool = True
    rnn_layers: int = 1
    rnn_hidden_dim: int = 512
    tau: float = 5e-3
    gamma: float = 0.99
    num_heads: int = 50
    # Training
    update_steps: int = 180000
    batch_size: int = 256
    seq_len: int = 32
    n_workers: int = 8
    learning_rate: float = 3e-4
    clip_grad_norm: Optional[float] = 4.0
    checkpoints_path: Optional[str] = None
    eval_every: int = 10_000
    eval_episodes_per_seed: int = 1
    eval_seeds: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.group}-{self.env}-{self.version}"
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                self.checkpoints_path, self.group, self.name
            )


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def sample_convex_combination(size, device="cpu"):
    weights = torch.rand(size, device=device)
    weights = weights / weights.sum()
    assert torch.isclose(weights.sum(), torch.tensor([1.0], device=device))
    return weights


# def symlog(x):
#     return torch.sign(x) * torch.log(torch.abs(x) + 1)
#
#
# def symexp(x):
#     return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def rem_dqn_loss(
        critic,
        target_critic,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        rnn_states,
        target_rnn_states,
        convex_comb_weights,
        gamma,
):
    with torch.no_grad():
        next_q_values, next_target_rnn_states = target_critic(next_obs, state=target_rnn_states)
        next_q_values = (next_q_values * convex_comb_weights).sum(2)
        next_q_values = next_q_values.max(dim=-1).values

        assert next_q_values.shape == rewards.shape == dones.shape
        q_target = rewards + gamma * (1 - dones) * next_q_values
        # q_target = symlog(rewards + gamma * (1 - dones) * symexp(next_q_values))

    assert actions.dim() == 2
    q_pred, next_rnn_states = critic(obs, state=rnn_states)
    q_pred = (q_pred * convex_comb_weights.detach()).sum(2)
    q_pred = q_pred.gather(-1, actions.to(torch.long).unsqueeze(-1)).squeeze()
    assert q_pred.shape == q_target.shape

    loss = F.mse_loss(q_pred, q_target)
    loss_info = {
        "loss": loss.item(),
        "q_target": q_target.mean().item()
    }
    return loss, next_rnn_states, next_target_rnn_states, loss_info


class Critic(nn.Module):
    def __init__(self, action_dim, rnn_hidden_dim, rnn_layers, num_heads, use_prev_action=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_actions = action_dim
        self.use_prev_action = use_prev_action
        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0

        # Encoders
        self.topline_encoder = torch.jit.script(TopLineEncoder())
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder())

        screen_shape = (SCREEN_SHAPE[1], SCREEN_SHAPE[2])
        self.screen_encoder = torch.jit.script(ScreenEncoder(screen_shape))

        self.h_dim = sum([
            self.topline_encoder.hidden_dim,
            self.bottomline_encoder.hidden_dim,
            self.screen_encoder.hidden_dim,
            self.prev_actions_dim,
        ])
        # Policy
        self.rnn = nn.LSTM(self.h_dim, rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.head = nn.Linear(rnn_hidden_dim, self.num_actions * num_heads)

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
                torch.nn.functional.one_hot(
                    inputs["prev_actions"], self.num_actions
                ).view(T * B, -1)
            )
        encoded_state = torch.cat(encoded_state, dim=1)
        core_output, new_state = self.rnn(encoded_state.view(B, T, -1), state)
        q_values_ensemble = self.head(core_output).view(B, T, self.num_heads, self.num_actions)
        return q_values_ensemble, new_state

    @torch.no_grad()
    def act(self, obs, state=None, device="cpu"):
        inputs = {
            "screen_image": torch.tensor(obs["screen_image"], device=device)[None, None, ...],
            "tty_chars": torch.tensor(obs["tty_chars"], device=device)[None, None, ...],
            "prev_actions": torch.tensor(obs["prev_actions"], dtype=torch.long, device=device)[None, None, ...],
        }
        q_values_ensemble, new_state = self(inputs, state)
        # mean q value over all heads
        q_values = q_values_ensemble.mean(2)
        return torch.argmax(q_values).cpu().item(), new_state


@torch.no_grad()
def evaluate(env_builder, actor: Critic, episodes_per_seed: int, device="cpu"):
    actor.eval()
    eval_stats = defaultdict(dict)
    # TODO: we should not reset hidden state and prev_actions on evaluation, to mimic the training
    for (character, env, seed) in tqdm(env_builder.evaluate()):
        episodes_rewards = []
        for _ in trange(episodes_per_seed, desc="One seed evaluation", leave=False):
            env.seed(seed, reseed=False)

            obs, done, episode_reward = env.reset(), False, 0.0
            rnn_state = None
            obs["prev_actions"] = 0

            while not done:
                action, rnn_state = actor.act(obs, rnn_state, device=device)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                obs["prev_actions"] = action

            episodes_rewards.append(episode_reward)

        eval_stats[character][seed] = np.mean(episodes_rewards)

    # for each character also log mean across all seeds
    for character in eval_stats.keys():
        eval_stats[character]["mean_return"] = np.mean(list(eval_stats[character].values()))

    actor.train()
    return eval_stats


@pyrallis.wrap()
def train(config: TrainConfig):
    print(f"Device: {DEVICE}")
    saved_config = asdict(config)
    saved_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")
    wandb.init(
        config=saved_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    set_seed(config.train_seed)

    env_builder, dataset_builder = make_task_builder(
        config.env, data_path=config.data_path, db_path=config.db_path
    )
    env_builder = (
        env_builder.roles([Role.MONK])
        .races([Race.HUMAN])
        .alignments([Alignment.NEUTRAL])
        .sex([Sex.MALE])
        .eval_seeds(list(range(config.eval_seeds)))
    )

    dataset_builder = dataset_builder.roles([Role.MONK]).races([Race.HUMAN])
    dataset = dataset_builder.build(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_workers=config.n_workers,
        auto_ascend_cls=SARSChaoticAutoAscendTTYDataset,
    )

    critic = Critic(
        action_dim=env_builder.get_action_dim(),
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_layers=config.rnn_layers,
        num_heads=config.num_heads,
        use_prev_action=config.use_prev_action,
    ).to(DEVICE)
    with torch.no_grad():
        target_critic = deepcopy(critic)

    optim = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    print("Number of parameters:", sum(p.numel() for p in critic.parameters()))

    loader = DataLoader(
        dataset=dataset,
        # Disable automatic batching
        batch_sampler=None,
        batch_size=None,
        pin_memory=True,
    )
    scaler = torch.cuda.amp.GradScaler()

    prev_actions = torch.zeros((config.batch_size, 1), dtype=torch.long, device=DEVICE)
    rnn_state, target_rnn_state = None, None

    loader_iter = iter(loader)
    for step in trange(config.update_steps, desc="Training"):
        screen_image, tty_chars, actions, rewards, next_screen_image, next_tty_chars, dones = (
            [t.to(DEVICE) for t in next(loader_iter)]
        )
        actions = actions.long()

        obs = {
            "screen_image": screen_image,
            "tty_chars": tty_chars,
            "prev_actions": torch.cat([prev_actions, actions[:, :-1]], dim=1)
        }
        next_obs = {
            "screen_image": next_screen_image,
            "tty_chars": next_tty_chars,
            "prev_actions": actions
        }
        convex_comb_weights = sample_convex_combination(config.num_heads, device=DEVICE).view(1, 1, -1, 1)

        loss, rnn_state, target_rnn_state, loss_info = rem_dqn_loss(
            critic=critic,
            target_critic=target_critic,
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            rnn_states=rnn_state,
            target_rnn_states=target_rnn_state,
            convex_comb_weights=convex_comb_weights,
            gamma=config.gamma
        )
        rnn_state = [s.detach() for s in rnn_state]
        target_rnn_state = [s.detach() for s in target_rnn_state]

        scaler.scale(loss).backward()
        if config.clip_grad_norm is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.clip_grad_norm)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        soft_update(target_critic, critic, tau=config.tau)
        prev_actions = actions[:, -1].unsqueeze(-1)

        wandb.log(
            dict(loss_info, **{"transitions": config.batch_size * config.seq_len * step}),
            step=step,
        )

        if (step + 1) % config.eval_every == 0:
            eval_stats = evaluate(env_builder, critic, config.eval_episodes_per_seed, device=DEVICE)
            wandb.log(
                dict(eval_stats, **{"transitions": config.batch_size * config.seq_len * step}),
                step=step,
            )
            if config.checkpoints_path is not None:
                torch.save(
                    critic.state_dict(),
                    os.path.join(config.checkpoints_path, f"{step}.pt"),
                )


if __name__ == "__main__":
    train()
