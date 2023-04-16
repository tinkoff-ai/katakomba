import pyrallis
from dataclasses import dataclass, asdict
import time
import gym
import minihack
import h5py
import random
import wandb
import sys
import os
import uuid
import torch
import torch.nn as nn

import torch.nn.functional as F
from collections import deque
from tqdm.auto import tqdm, trange
from torch.distributions import Categorical
import numpy as np

from katakomba.nn.chaotic_dwarf import TopLineEncoder, BottomLinesEncoder, ScreenEncoder
from katakomba.utils.render import SCREEN_SHAPE, render_screen_image
from typing import Optional


torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class timeit:
    def __enter__(self):
        self.start_gpu = torch.cuda.Event(enable_timing=True)
        self.end_gpu = torch.cuda.Event(enable_timing=True)
        self.start_cpu = time.time()
        self.start_gpu.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_gpu.record()
        torch.cuda.synchronize()
        self.elapsed_time_gpu = self.start_gpu.elapsed_time(self.end_gpu) / 1000
        self.elapsed_time_cpu = time.time() - self.start_cpu


@dataclass
class TrainConfig:
    env: str = "MiniHack-Room-Trap-15x15-v0"
    data_path: str = "data/MiniHack-Room-Trap-15x15-v0-dataset-v0.hdf5"
    # Wandb logging
    project: str = "MiniHack"
    group: str = "ChaoticBC"
    name: str = "ChaoticBC"
    version: str = "v0"
    # Model
    rnn_hidden_dim: int = 512
    rnn_layers: int = 1
    use_prev_action: bool = True
    # Training
    update_steps: int = 5000
    batch_size: int = 256
    seq_len: int = 8
    learning_rate: float = 3e-4
    clip_grad_norm: Optional[float] = None
    checkpoints_path: Optional[str] = None
    eval_every: int = 100
    eval_episodes: int = 25
    eval_seed: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.group}-{self.env}-{self.version}"
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.group, self.name)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def dict_slice(data, start, end):
    return {
        k: v[start:end] for k, v in data.items()
    }


def dict_concat(datas):
    return {
        k: torch.cat([d[k] for d in datas]) for k in datas[0].keys()
    }


def dict_stack(datas):
    return {
        k: torch.stack([d[k] for d in datas]) for k in datas[0].keys()
    }


def load_trajectories(hdf5_path):
    trajectories = []
    total_transitions = 0.0

    with h5py.File(hdf5_path, "r") as f:
        for key in tqdm(list(f["/"].keys())):  #[:500]
            # if f[key]["rewards"][()].sum() < 0.8:
            #     continue

            tty_chars = f[key]["observations/tty_chars"][()][:-1]
            tty_colors = f[key]["observations/tty_colors"][()][:-1]
            tty_cursor = f[key]["observations/tty_cursor"][()][:-1]

            screen_image = render_screen_image(tty_chars[None, :], tty_colors[None, :], tty_cursor[None, :])
            trajectory = {
                "screen_image": torch.tensor(screen_image[0]),
                "tty_chars": torch.tensor(tty_chars),
                "actions": torch.tensor(f[key]["actions"][()]),
            }
            trajectories.append(trajectory)
            total_transitions += f[key].attrs["total_steps"]

    print(f"Loaded total {len(trajectories)} trajectories! Transitions in total: {total_transitions}")
    return trajectories


class SequentialBuffer:
    def __init__(self, trajectories, batch_size, seq_len):
        assert batch_size < len(trajectories)

        self.traj = trajectories
        self.traj_idxs = list(range(len(self.traj)))

        self.batch_size = batch_size
        self.seq_len = seq_len

        random.shuffle(self.traj_idxs)
        self.free_traj = deque(self.traj_idxs)
        self.curr_traj = np.array([self.free_traj.popleft() for _ in range(batch_size)], dtype=int)
        self.curr_idx = np.zeros(batch_size, dtype=int)

    def sample(self):
        batch = []

        for i in range(self.batch_size):
            traj_idx = self.curr_traj[i]
            start_idx = self.curr_idx[i]

            data = dict_slice(self.traj[traj_idx], start_idx, start_idx + self.seq_len)

            if len(data["actions"]) < self.seq_len:
                # if next traj will have total_len < seq_len, then get next until data is seq_len
                while len(data["actions"]) < self.seq_len:
                    if len(self.free_traj) == 0:
                        # reinit buffer of free trajectories
                        random.shuffle(self.traj_idxs)
                        self.free_traj = deque(self.traj_idxs)

                    traj_idx = self.free_traj.popleft()
                    len_diff = self.seq_len - len(data["actions"])

                    data = dict_concat([
                        data,
                        dict_slice(self.traj[traj_idx], 0, len_diff),
                    ])
                    self.curr_traj[i] = traj_idx
                    self.curr_idx[i] = len_diff
            else:
                self.curr_idx[i] += self.seq_len

            batch.append(data)

        return dict_stack(batch)


class Actor(nn.Module):
    def __init__(self, action_dim, rnn_hidden_dim=512, rnn_layers=1, use_prev_action=True):
        super().__init__()
        # Action dimensions and prev actions
        self.num_actions = action_dim
        self.use_prev_action = use_prev_action
        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0

        # Encoders
        self.topline_encoder = TopLineEncoder()
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
        self.rnn = nn.LSTM(self.h_dim, rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.head = nn.Linear(rnn_hidden_dim, self.num_actions)

    def forward(self, inputs, state=None):
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
        logits = self.head(core_output)

        return logits, new_state

    @torch.no_grad()
    def act(self, obs, state=None, device="cpu"):
        inputs = {
            "tty_chars": torch.tensor(
                obs["tty_chars"][np.newaxis, np.newaxis, ...], device=device
            ),
            "screen_image": torch.tensor(
                obs["screen_image"][np.newaxis, np.newaxis, ...], device=device
            ),
            "prev_actions": torch.tensor(
                np.array([obs["prev_actions"]]).reshape(1, 1),
                dtype=torch.long,
                device=device,
            ),
        }
        logits, new_state = self(inputs, state)
        return torch.argmax(logits).cpu().item(), new_state


@torch.no_grad()
def evaluate(env, actor, num_episodes, seed=0, device="cpu"):
    actor.eval()
    returns = np.zeros(num_episodes)

    for i in trange(num_episodes, desc="Evaluation", leave=False):
        episode_reward = 0.0

        env.seed(seed + i, reseed=False)
        obs, done = env.reset(), False

        rnn_state = None
        obs["prev_actions"] = 0
        while not done:
            obs["screen_image"] = render_screen_image(
                tty_chars=obs["tty_chars"][np.newaxis, np.newaxis, ...],
                tty_colors=obs["tty_colors"][np.newaxis, np.newaxis, ...],
                tty_cursor=obs["tty_cursor"][np.newaxis, np.newaxis, ...],
            )
            obs["screen_image"] = np.squeeze(obs["screen_image"])

            action, rnn_state = actor.act(obs, rnn_state, device=device)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            obs["prev_actions"] = action

        returns[i] = episode_reward

    actor.train()
    return returns


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
    eval_env = gym.make(
        config.env,
        observation_keys=["tty_chars", "tty_colors", "tty_cursor"]
    )

    dataset = SequentialBuffer(
        trajectories=load_trajectories(config.data_path),
        batch_size=config.batch_size,
        seq_len=config.seq_len
    )
    actor = Actor(
        action_dim=eval_env.action_space.n,
        use_prev_action=config.use_prev_action,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_layers=config.rnn_layers,
    ).to(DEVICE)
    optim = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    print("Number of parameters:", sum(p.numel() for p in actor.parameters()))

    scaler = torch.cuda.amp.GradScaler()

    rnn_state = None
    prev_actions = torch.zeros((config.batch_size, 1), dtype=torch.long, device=DEVICE)
    for step in trange(config.update_steps, desc="Training"):
        # with timeit() as timer:
        batch = {k: v.to(DEVICE) for k, v in dataset.sample().items()}

        # wandb.log(
        #     {
        #         "times/batch_loading_cpu": timer.elapsed_time_cpu,
        #         "times/batch_loading_gpu": timer.elapsed_time_gpu,
        #     },
        #     step=step,
        # )

        # with timeit() as timer:
        with torch.cuda.amp.autocast():
            logits, rnn_state = actor(
                inputs={
                    "screen_image": batch["screen_image"],
                    "tty_chars": batch["tty_chars"],
                    "prev_actions": torch.cat(
                        [prev_actions.long(), batch["actions"][:, :-1].long()], dim=1
                    )
                },
                state=rnn_state,
            )
            rnn_state = [a.detach() for a in rnn_state]

            dist = Categorical(logits=logits)
            loss = -dist.log_prob(batch["actions"]).mean()
            # update prev_actions for next iteration
            prev_actions = batch["actions"][:, -1].unsqueeze(-1)

        # wandb.log({"times/forward_pass": timer.elapsed_time_gpu}, step=step)

        # with timeit() as timer:
        scaler.scale(loss).backward()
        # loss.backward()
        if config.clip_grad_norm is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.clip_grad_norm)
        # optim.step()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        # wandb.log({"times/backward_pass": timer.elapsed_time_gpu}, step=step)

        wandb.log({
                "loss": loss.detach().item(),
                "transitions": config.batch_size * config.seq_len * step,
        }, step=step)

        if (step + 1) % config.eval_every == 0:
            eval_returns = evaluate(
                eval_env, actor, config.eval_episodes, config.eval_seed, device=DEVICE
            )
            wandb.log({
                "return_mean": eval_returns.mean(),
                "return_std": eval_returns.std(),
                "transitions": config.batch_size * config.seq_len * step
            }, step=step)

            if config.checkpoints_path is not None:
                torch.save(
                    actor.state_dict(),
                    os.path.join(config.checkpoints_path, f"{step}.pt"),
                )


if __name__ == "__main__":
    train()
