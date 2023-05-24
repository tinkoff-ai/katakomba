import pyrallis
from dataclasses import dataclass, asdict
import random
import wandb
import os
import uuid
import torch
import torch.nn as nn

from gym.vector import AsyncVectorEnv
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from torch.distributions import Categorical
import numpy as np

from katakomba.env import NetHackChallenge, OfflineNetHackChallengeWrapper
from katakomba.nn.chaotic_dwarf import TopLineEncoder, BottomLinesEncoder, ScreenEncoder
from katakomba.utils.render import SCREEN_SHAPE, render_screen_image
from katakomba.utils.datasets import SequentialBuffer
from katakomba.utils.misc import Timeit
from typing import Optional

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    character: str = "mon-hum-neu"
    data_mode: str = "compressed"
    # Wandb logging
    project: str = "NetHack"
    group: str = "small_scale_bc"
    name: str = "bc"
    version: int = 0
    # Model
    rnn_hidden_dim: int = 2048
    rnn_layers: int = 2
    use_prev_action: bool = True
    rnn_dropout: float = 0.0
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
    return {k: torch.as_tensor(v, device=device) for k, v in data.items()}


class Actor(nn.Module):
    def __init__(self, action_dim, rnn_hidden_dim=512, rnn_layers=1, rnn_dropout=0.0, use_prev_action=True):
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
        self.rnn = nn.LSTM(
            self.h_dim,
            rnn_hidden_dim,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True
        )
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
    episode_depths = []

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
        # faster to do this here for entire batch, than in wrappers for each env
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
                    episode_depths.append(infos[i]["current_depth"])
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
        "reward_raw": np.array(episode_rewards),
        # depth
        "depth_median": np.median(episode_depths),
        "depth_mean": np.mean(episode_depths),
        "depth_std": np.std(episode_depths),
        "depth_min": np.min(episode_depths),
        "depth_max": np.max(episode_depths),
        "depth_raw": np.array(episode_depths),
    }
    actor.train()
    return result


@pyrallis.wrap()
def train(config: TrainConfig):
    print(f"Device: {DEVICE}")
    wandb.init(
        config=asdict(config),
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

    def env_fn():
        env = NetHackChallenge(
            character=config.character,
            observation_keys=["tty_chars", "tty_colors", "tty_cursor"]
        )
        env = OfflineNetHackChallengeWrapper(env)
        return env

    tmp_env = env_fn()
    eval_env = AsyncVectorEnv(
        env_fns=[env_fn for _ in range(config.eval_processes)],
        shared_memory=True,
        copy=False
    )
    buffer = SequentialBuffer(
        dataset=tmp_env.get_dataset(mode=config.data_mode, scale="small"),
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        seed=config.train_seed,
        add_next_step=False
    )
    tp = ThreadPoolExecutor(max_workers=14)

    actor = Actor(
        action_dim=eval_env.single_action_space.n,
        use_prev_action=config.use_prev_action,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_layers=config.rnn_layers,
        rnn_dropout=config.rnn_dropout,
    ).to(DEVICE)

    no_decay_params, decay_params = filter_wd_params(actor)
    optim = torch.optim.AdamW([
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": config.weight_decay}
    ], lr=config.learning_rate)
    print("Number of parameters:", sum(p.numel() for p in actor.parameters()))

    scaler = torch.cuda.amp.GradScaler()

    rnn_state = None
    prev_actions = torch.zeros((config.batch_size, 1), dtype=torch.long, device=DEVICE)
    for step in trange(1, config.update_steps + 1, desc="Training"):
        with Timeit() as timer:
            batch = buffer.sample()
            screen_image = render_screen_image(
                tty_chars=batch["tty_chars"],
                tty_colors=batch["tty_colors"],
                tty_cursor=batch["tty_cursor"],
                threadpool=tp,
            )
            batch["screen_image"] = screen_image
            batch = dict_to_tensor(batch, device=DEVICE)

        wandb.log(
            {
                "times/batch_loading_cpu": timer.elapsed_time_cpu,
                "times/batch_loading_gpu": timer.elapsed_time_gpu,
            },
            step=step,
        )

        with Timeit() as timer:
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

        wandb.log({"times/forward_pass": timer.elapsed_time_gpu}, step=step)

        with Timeit() as timer:
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

        if step % config.eval_every == 0:
            with Timeit() as timer:
                eval_stats = vec_evaluate(
                    eval_env, actor, config.eval_episodes, config.eval_seed, device=DEVICE
                )
            raw_returns = eval_stats.pop("reward_raw")
            raw_depths = eval_stats.pop("depth_raw")
            normalized_scores = tmp_env.get_normalized_score(raw_returns)

            wandb.log({
                "times/evaluation_gpu": timer.elapsed_time_gpu,
                "times/evaluation_cpu": timer.elapsed_time_cpu,
            }, step=step)

            wandb.log(dict(
                eval_stats,
                **{"transitions": config.batch_size * config.seq_len * step},
            ), step=step)

            if config.checkpoints_path is not None:
                torch.save(actor.state_dict(), os.path.join(config.checkpoints_path, f"{step}.pt"))
                # saving raw logs
                np.save(os.path.join(config.checkpoints_path, f"{step}_returns.npy"), raw_returns)
                np.save(os.path.join(config.checkpoints_path, f"{step}_depths.npy"), raw_depths)
                np.save(os.path.join(config.checkpoints_path, f"{step}_normalized_scores.npy"), normalized_scores)

            # also saving to wandb files for easier use in the future
            np.save(os.path.join(wandb.run.dir, f"{step}_returns.npy"), raw_returns)
            np.save(os.path.join(wandb.run.dir, f"{step}_depths.npy"), raw_depths)
            np.save(os.path.join(wandb.run.dir, f"{step}_normalized_scores.npy"), normalized_scores)

    buffer.close()


if __name__ == "__main__":
    train()
