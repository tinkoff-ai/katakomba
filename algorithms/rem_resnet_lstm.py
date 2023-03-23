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
from d5rl.tasks import make_task_builder
from d5rl.utils.roles import Alignment, Race, Role, Sex
from d5rl.nn.resnet import ResNet11, ResNet20, ResNet38, ResNet56, ResNet110

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    env: str = "NetHackScore-v0-tty-bot-v0"
    data_path: str = "data/nle_data"
    db_path: str = "ttyrecs.db"
    # Wandb logging
    project: str = "NeuralNetHack"
    group: str = "REM"
    name: str = "REM"
    version: str = "v0"
    # Model
    resnet_type: str = "ResNet20"
    lstm_layers: int = 1
    hidden_dim: int = 2048
    width_k: int = 1
    tau: float = 5e-3
    gamma: float = 0.99
    num_heads: int = 1
    # Training
    update_steps: int = 180000
    batch_size: int = 256
    seq_len: int = 32
    n_workers: int = 8
    learning_rate: float = 3e-4
    clip_grad_norm: Optional[float] = None
    checkpoints_path: Optional[str] = None
    eval_every: int = 10_000
    eval_episodes_per_seed: int = 1
    eval_seeds: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.env}-{self.name}-{self.version}"
        self.name = f"{self.group}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


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


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


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
    # TODO: should we use double Q? should we use temporal consistency loss from Ape-X DQfD?
    with torch.no_grad():
        next_q_values, next_target_rnn_states = target_critic(next_obs, state=target_rnn_states)
        next_q_values = (next_q_values * convex_comb_weights).sum(2)
        next_q_values = next_q_values.max(dim=-1).values

        assert next_q_values.shape == rewards.shape == dones.shape
        # q_target = rewards + gamma * (1 - dones) * next_q_values
        q_target = symlog(rewards + gamma * (1 - dones) * symexp(next_q_values))

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
    def __init__(self, action_dim, hidden_dim, lstm_layers, width_k, resnet_type, num_heads):
        super().__init__()
        resnet = getattr(sys.modules[__name__], resnet_type)
        self.state_encoder = resnet(img_channels=3, out_dim=hidden_dim, k=width_k)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, num_heads * action_dim)

        self.action_dim = action_dim
        self.num_heads = num_heads

    def forward(self, obs, state=None):
        # [batch_size, seq_len, ...]
        batch_size, seq_len, *_ = obs.shape

        out = self.state_encoder(obs.flatten(0, 1)).view(batch_size, seq_len, -1)
        #TODO: should we normalize penultimate layer?
        out, new_state = self.rnn(out, state)
        q_values_ensemble = self.head(out).view(batch_size, seq_len, self.num_heads, self.action_dim)
        return q_values_ensemble, new_state

    @torch.no_grad()
    def act(self, obs, state=None, device="cpu"):
        assert obs.ndim == 3, "act only for single obs"
        obs = torch.tensor(obs, device=device, dtype=torch.float32).permute(2, 0, 1)
        q_values_ensemble, new_state = self(obs[None, None, ...], state)
        # mean q value over all heads
        q_values = q_values_ensemble.mean(2)
        # return torch.argmax(q_values).cpu().item(), new_state
        return torch.argmax(symexp(q_values)).cpu().item(), new_state


@torch.no_grad()
def evaluate(env_builder, actor, episodes_per_seed, device="cpu"):
    actor.eval()
    eval_stats = defaultdict(dict)

    for (character, env, seed) in tqdm(env_builder.evaluate()):
        episodes_rewards = []
        for _ in trange(episodes_per_seed, desc="One seed evaluation", leave=False):
            env.seed(seed, reseed=False)

            obs, done, episode_reward = env.reset(), False, 0.0
            rnn_state = None

            while not done:
                action, rnn_state = actor.act(obs, rnn_state, device=device)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
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
        save_code=True
    )
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    set_seed(config.train_seed)
    env_builder, dataset_builder = make_task_builder(
        config.env,
        data_path=config.data_path,
        db_path=config.db_path
    )
    env_builder = (
        env_builder.roles([Role.MONK])
        .races([Race.HUMAN])
        .alignments([Alignment.NEUTRAL])
        .sex([Sex.MALE])
        .eval_seeds(list(range(config.eval_seeds)))
    )
    dataset = dataset_builder.build(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_workers=config.n_workers
    )

    critic = Critic(
        resnet_type=config.resnet_type,
        action_dim=env_builder.get_action_dim(),
        hidden_dim=config.hidden_dim,
        lstm_layers=config.lstm_layers,
        width_k=config.width_k,
        num_heads=config.num_heads
    ).to(DEVICE)
    with torch.no_grad():
        target_critic = deepcopy(critic)

    print("Number of parameters:",  sum(p.numel() for p in critic.parameters()))
    critic = torch.compile(critic, mode="reduce-overhead")

    optim = torch.optim.AdamW(critic.parameters(), lr=config.learning_rate)

    loader = DataLoader(
        dataset=dataset,
        # Disable automatic batching
        batch_sampler=None,
        batch_size=None,
        pin_memory=True
    )
    # scaler = torch.cuda.amp.GradScaler()

    rnn_state, target_rnn_state = None, None

    loader_iter = iter(loader)
    for step in trange(config.update_steps, desc="Training"):
        obs, actions, rewards, dones, next_obs = [t.to(DEVICE) for t in next(loader_iter)]

        # with torch.cuda.amp.autocast():
        convex_comb_weights = sample_convex_combination(config.num_heads, device=DEVICE).view(1, 1, -1, 1)

        loss, rnn_state, target_rnn_state, loss_info = rem_dqn_loss(
            critic=critic,
            target_critic=target_critic,
            obs=obs.permute(0, 1, 4, 2, 3).to(torch.float32),
            actions=actions,
            rewards=rewards,
            next_obs=next_obs.permute(0, 1, 4, 2, 3).to(torch.float32),
            dones=dones,
            rnn_states=rnn_state,
            target_rnn_states=target_rnn_state,
            convex_comb_weights=convex_comb_weights,
            gamma=config.gamma
        )
        rnn_state = [s.detach() for s in rnn_state]
        target_rnn_state = [s.detach() for s in target_rnn_state]

        # scaler.scale(loss).backward()
        loss.backward()
        if config.clip_grad_norm is not None:
            # scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.clip_grad_norm)
        # scaler.step(optim)
        # scaler.update()
        optim.step()
        optim.zero_grad(set_to_none=True)

        soft_update(target_critic, critic, tau=config.tau)

        wandb.log(loss_info, step=step)

        if (step + 1) % config.eval_every == 0:
            eval_stats = evaluate(env_builder, critic, config.eval_episodes_per_seed, device=DEVICE)
            wandb.log(eval_stats, step=step)

            if config.checkpoints_path is not None:
                torch.save(
                    critic.state_dict(),
                    os.path.join(config.checkpoints_path, f"{step}.pt"),
                )


if __name__ == "__main__":
    train()
