import pyrallis
from dataclasses import dataclass, asdict
import time
import gym
import h5py
import random
import wandb
import sys
import os
import uuid
import torch
import torch.nn as nn

import haste_pytorch as haste

from gym.vector import AsyncVectorEnv
import nle
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from collections import deque
from tqdm.auto import tqdm, trange
from torch.distributions import Categorical
import numpy as np

from katakomba.envs import NetHackChallenge
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


# TODO: actions to embeddings, add prev reward
# TODO: multilayer rnn from haste + residual connections + LN + average over hiddens
@dataclass
class TrainConfig:
    data_path: str = "data/nle_medium.hdf5"
    # Wandb logging
    project: str = "NetHack"
    group: str = "nle_medium"
    name: str = "chaotic_bc"
    version: str = "v0"
    # Model
    rnn_hidden_dim: int = 512
    rnn_layers: int = 5
    use_prev_action: bool = True
    # Training
    update_steps: int = 500_000
    batch_size: int = 8 # 64
    seq_len: int = 512 # 16
    learning_rate: float = 3e-4
    clip_grad_norm: Optional[float] = None
    checkpoints_path: Optional[str] = None
    eval_every: int = 50_000
    eval_episodes: int = 100
    eval_processes: int = 14
    eval_seed: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.group}-NetHackChallenge-v0-{self.version}"
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.group, self.name)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def dict_slice(data, start, end):
    return {k: v[start:end] for k, v in data.items()}


def dict_concat(datas):
    return {k: np.concatenate([d[k] for d in datas]) for k in datas[0].keys()}


def dict_stack(datas):
    return {k: np.stack([d[k] for d in datas]) for k in datas[0].keys()}


def dict_to_tensor(data, device):
    return {k: torch.as_tensor(v, device=device) for k, v in data.items()}


def load_trajectories(hdf5_path):
    trajectories = []
    total_transitions = 0.0

    with h5py.File(hdf5_path, "r") as f:
        for key in tqdm(list(f["/"].keys())[:800]):
            trajectories.append({
                "tty_chars": f[key]["observations/tty_chars"][()],
                "tty_colors": f[key]["observations/tty_colors"][()],
                "tty_cursor": f[key]["observations/tty_cursor"][()],
                "actions": f[key]["actions"][()]
            })
            total_transitions += trajectories[-1]["actions"].shape[0]

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
        # needed for faster rendering of tty as images
        self.tp = ThreadPoolExecutor(max_workers=14)

    def sample(self, device="cpu"):
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
                    assert data["actions"].shape[0] == self.seq_len, f"seq_len is not full!, shape {data['actions'].shape[0]}"

                    self.curr_traj[i] = traj_idx
                    self.curr_idx[i] = len_diff
            else:
                self.curr_idx[i] += self.seq_len

            batch.append(data)

        batch = dict_stack(batch)
        screen_image = render_screen_image(
            tty_chars=batch["tty_chars"],
            tty_colors=batch["tty_colors"],
            tty_cursor=batch["tty_cursor"],
            threadpool=self.tp,
        )
        batch["screen_image"] = screen_image
        return dict_to_tensor(batch, device=device)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, zoneout=0.0, num_layers=1, batch_first=True):
        super().__init__()
        self.rnns = nn.ModuleList([
            # haste.LayerNormLSTM(
            #     input_size if i == 0 else hidden_size,
            #     hidden_size,
            #     zoneout=zoneout,
            #     batch_first=batch_first
            # )
            haste.IndRNN(
                input_size if i == 0 else hidden_size,
                hidden_size,
                zoneout=zoneout,
                batch_first=batch_first
            )
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(input_size if i == 0 else hidden_size) for i in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, state=None):
        outs, hs, cs = [], [], []

        x = self.norms[0](x)
        for i, (rnn, norm) in enumerate(zip(self.rnns, self.norms)):
            # l_state = None if state is None else (state[0][i].unsqueeze(0), state[1][i].unsqueeze(0))
            l_state = None if state is None else state[i].unsqueeze(0)

            # x, (h, c) = rnn(x, l_state)
            # x, (h, c) = rnn(norm(x), l_state)
            x, h = rnn(norm(x), l_state)
            outs.append(x)
            hs.append(h)
            # cs.append(c)

        return sum(outs), torch.cat(hs, dim=0)
        # return sum(outs), (torch.cat(hs, dim=0), torch.cat(cs, dim=0))
        # return outs[-1], (torch.cat(hs, dim=0), torch.cat(cs, dim=0))


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
        self.rnn = RNN(self.h_dim, rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
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
    def vec_act(self, obs, state=None, device="cpu"):
        inputs = {
            "tty_chars": torch.tensor(obs["tty_chars"][:, None], device=device),
            "screen_image": torch.tensor(obs["screen_image"][:, None], device=device),
            "prev_actions": torch.tensor(obs["prev_actions"][:, None], dtype=torch.long, device=device)
        }
        logits, new_state = self(inputs, state)
        actions = torch.argmax(logits.squeeze(1), dim=-1)
        return actions.cpu().numpy(), new_state


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

    result = {
        "reward_median": np.median(returns),
        "reward_mean": np.mean(returns),
        "reward_std": np.std(returns),
        "reward_min": np.min(returns),
        "reward_max": np.max(returns),
    }
    actor.train()
    return result


@torch.no_grad()
def vec_evaluate(vec_env, actor, num_episodes,  seed=0, device="cpu"):
    actor.eval()
    # set seed for reproducibility (reseed=False by default)
    vec_env.seed(seed)
    # all this work is needed to mitigate bias for shorter
    # episodes during vectorized evaluation, for more see:
    # https://github.com/DLR-RM/stable-baselines3/issues/402
    n_envs = vec_env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(num_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = vec_env.reset()
    observations["prev_actions"] = np.zeros(n_envs, dtype=float)

    rnn_states = None
    pbar = tqdm(total=num_episodes)
    while (episode_counts < episode_count_targets).any():
        observations["screen_image"] = render_screen_image(
            tty_chars=observations["tty_chars"][:, np.newaxis, ...],
            tty_colors=observations["tty_colors"][:, np.newaxis, ...],
            tty_cursor=observations["tty_cursor"][:, np.newaxis, ...],
        )
        observations["screen_image"] = np.squeeze(observations["screen_image"], 1)

        actions, rnn_states = actor.vec_act(observations, rnn_states, device=device)

        observations, rewards, dones, infos = vec_env.step(actions)
        observations["prev_actions"] = actions

        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    pbar.update(1)

                    current_rewards[i] = 0
                    current_lengths[i] = 0

    pbar.close()
    result = {
        "reward_median": np.median(episode_rewards),
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_min": np.min(episode_rewards),
        "reward_max": np.max(episode_rewards),
    }
    actor.train()
    return result


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
    # eval_env = NetHackChallenge(character="mon-hum", savedir=False)
    eval_env = AsyncVectorEnv(
         env_fns=[lambda: NetHackChallenge(character="mon-hum") for _ in range(config.eval_processes)],
         shared_memory=True,
         copy=False
     )

    dataset = SequentialBuffer(
        trajectories=load_trajectories(config.data_path),
        batch_size=config.batch_size,
        seq_len=config.seq_len
    )
    actor = Actor(
        action_dim=eval_env.single_action_space.n,
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
        with timeit() as timer:
            batch = dataset.sample(device=DEVICE)

        wandb.log(
            {
                "times/batch_loading_cpu": timer.elapsed_time_cpu,
                "times/batch_loading_gpu": timer.elapsed_time_gpu,
            },
            step=step,
        )

        with timeit() as timer:
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
                # rnn_state = [a.detach() for a in rnn_state]
                rnn_state = rnn_state.detach()

                dist = Categorical(logits=logits)
                loss = -dist.log_prob(batch["actions"]).mean()
                # update prev_actions for next iteration
                prev_actions = batch["actions"][:, -1].unsqueeze(-1)

        wandb.log({"times/forward_pass": timer.elapsed_time_gpu}, step=step)

        with timeit() as timer:
            scaler.scale(loss).backward()
            # loss.backward()
            if config.clip_grad_norm is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), config.clip_grad_norm)
            # optim.step()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        wandb.log({"times/backward_pass": timer.elapsed_time_gpu}, step=step)

        wandb.log({
                "loss": loss.detach().item(),
                "transitions": config.batch_size * config.seq_len * step,
        }, step=step)

        if (step + 1) % config.eval_every == 0:
            with timeit() as timer:
                eval_stats = vec_evaluate(
                    eval_env, actor, config.eval_episodes, config.eval_seed, device=DEVICE
                )

            wandb.log({
                "times/evaluation_gpu": timer.elapsed_time_gpu,
                "times/evaluation_cpu": timer.elapsed_time_cpu,
            }, step=step)

            wandb.log(dict(
                eval_stats,
                **{"transitions": config.batch_size * config.seq_len * step},
            ), step=step)

            if config.checkpoints_path is not None:
                torch.save(
                    actor.state_dict(),
                    os.path.join(config.checkpoints_path, f"{step}.pt"),
                )


if __name__ == "__main__":
    train()
