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
from typing import Optional, Dict, Tuple, Any, List

from multiprocessing import set_start_method
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
    group: str = "small_scale_awac"
    name: str = "awac"
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
    render_processes: int = 14
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
def filter_wd_params(model: nn.Module) -> Tuple[List[nn.parameter.Parameter], List[nn.parameter.Parameter]]:
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


def dict_to_tensor(data: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float, device=device) for k, v in data.items()}


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


class ActorCritic(nn.Module):
    def __init__(
            self,
            action_dim: int,
            rnn_hidden_dim: int = 512,
            rnn_layers: int = 1,
            rnn_dropout: float = 0.0,
            use_prev_action: bool = True
    ):
        super().__init__()
        self.num_actions = action_dim
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
        # networks
        self.rnn = nn.LSTM(
            self.h_dim,
            rnn_hidden_dim,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True
        )
        self.qf = nn.Linear(rnn_hidden_dim, self.num_actions)
        self.policy = nn.Linear(rnn_hidden_dim, self.num_actions)

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
        qf = self.qf(core_output)
        logits = self.policy(core_output)

        return (qf, logits), new_state

    @torch.no_grad()
    def vec_act(self, obs, state=None, device="cpu"):
        inputs = {
            "tty_chars": torch.tensor(obs["tty_chars"][:, None], device=device),
            "screen_image": torch.tensor(obs["screen_image"][:, None], device=device),
            "prev_actions": torch.tensor(obs["prev_actions"][:, None], dtype=torch.long, device=device)
        }
        (_, logits), new_state = self(inputs, state)
        actions = torch.argmax(logits.squeeze(1), dim=-1)
        return actions.cpu().numpy(), new_state


def awac_loss(
        model: ActorCritic,
        target_model: ActorCritic,
        obs: Dict[str, torch.Tensor],
        next_obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rnn_states: LSTM_HIDDEN,
        target_rnn_states: LSTM_HIDDEN,
        gamma: float,
        temperature: float
) -> Tuple[torch.Tensor, LSTM_HIDDEN, LSTM_HIDDEN, UPDATE_INFO]:
    # critic loss
    with torch.no_grad():
        (next_q, next_logits), new_target_rnn_states = target_model(next_obs, state=target_rnn_states)
        next_actions = torch.distributions.Categorical(logits=next_logits).sample()
        next_q_actions = next_q.gather(-1, next_actions.to(torch.long).unsqueeze(-1)).squeeze()

        assert rewards.shape == dones.shape == next_q_actions.shape
        target_q = rewards + (1 - dones) * gamma * next_q_actions

    assert actions.dim() == 2
    (q_pred, logits_pred), new_rnn_states = model(obs, state=rnn_states)
    q_pred_actions = q_pred.gather(-1, actions.to(torch.long).unsqueeze(-1)).squeeze()
    assert q_pred_actions.shape == target_q.shape
    td_loss = F.mse_loss(q_pred_actions, target_q)

    # actor loss
    with torch.no_grad():
        adv = q_pred_actions - (q_pred * F.softmax(logits_pred, dim=-1)).sum(-1)

    log_probs = torch.distributions.Categorical(logits=logits_pred).log_prob(actions)
    weights = torch.exp(temperature * adv).clamp(max=100.0)
    actor_loss = torch.mean(-log_probs * weights)

    loss = td_loss + actor_loss
    loss_info = {
        "td_loss": td_loss.item(),
        "actor_loss": actor_loss.item(),
        "loss": loss,
        "q_target":  next_q.mean().item()
    }
    return loss, new_rnn_states, new_target_rnn_states, loss_info


@torch.no_grad()
def vec_evaluate(
        vec_env: AsyncVectorEnv,
        actor: ActorCritic,
        num_episodes: int,
        seed: int = 0,
        device: str = "cpu"
) -> Dict[str, np.ndarray]:
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
        copy=False
    )
    buffer = SequentialBuffer(
        dataset=tmp_env.get_dataset(mode=config.data_mode, scale="small"),
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        seed=config.train_seed,
        add_next_step=True       # true as this is needed for next_obs
    )
    tp = ThreadPoolExecutor(max_workers=config.render_processes)

    model = ActorCritic(
        action_dim=eval_env.single_action_space.n,
        use_prev_action=config.use_prev_action,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_layers=config.rnn_layers,
        rnn_dropout=config.rnn_dropout,
    ).to(DEVICE)
    with torch.no_grad():
        target_model = deepcopy(model)

    no_decay_params, decay_params = filter_wd_params(model)
    optim = torch.optim.AdamW([
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": config.weight_decay}
    ], lr=config.learning_rate)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    scaler = torch.cuda.amp.GradScaler()
    rnn_state, target_rnn_state = None, None
    prev_actions = torch.zeros((config.batch_size, 1), dtype=torch.long, device=DEVICE)

    # For reward normalization
    reward_stats = StatMean(cumulative=True)
    running_rewards = 0.0

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

            # Update reward statistics (as in the original nle implementation)
            running_rewards *= config.gamma
            running_rewards += batch["rewards"]
            reward_stats += running_rewards ** 2
            running_rewards *= (~batch["dones"]).astype(float)
            # Normalize the reward
            reward_std = reward_stats.mean() ** 0.5
            batch["rewards"] = batch["rewards"] / max(0.01, reward_std)
            batch["rewards"] = np.clip(batch["rewards"], -config.clip_range, config.clip_range)

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
                obs = {
                    "screen_image": batch["screen_image"][:, :-1].contiguous(),
                    "tty_chars": batch["tty_chars"][:, :-1].contiguous(),
                    "prev_actions": torch.cat([prev_actions.long(), batch["actions"][:, :-2].long()], dim=1)
                }
                next_obs = {
                    "screen_image": batch["screen_image"][:, 1:].contiguous(),
                    "tty_chars": batch["tty_chars"][:, 1:].contiguous(),
                    "prev_actions": batch["actions"][:, :-1].long()
                }
                loss, rnn_state, target_rnn_state, loss_info = awac_loss(
                    model=model,
                    target_model=target_model,
                    obs=obs,
                    next_obs=next_obs,
                    actions=batch["actions"][:, :-1],
                    rewards=batch["rewards"][:, :-1],
                    dones=batch["dones"][:, :-1],
                    rnn_states=rnn_state,
                    target_rnn_states=target_rnn_state,
                    temperature=config.temperature,
                    gamma=config.gamma
                )
                # detaching rnn hidden states for the next iteration
                rnn_state = [a.detach() for a in rnn_state]
                target_rnn_state = [a.detach() for a in target_rnn_state]

                # update prev_actions for next iteration (-1 is seq_len + 1, so -2)
                prev_actions = batch["actions"][:, -2].unsqueeze(-1)

        wandb.log({"times/forward_pass": timer.elapsed_time_gpu}, step=step)

        with Timeit() as timer:
            scaler.scale(loss).backward()
            if config.clip_grad_norm is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            soft_update(target_model, model, tau=config.tau)

        wandb.log({"times/backward_pass": timer.elapsed_time_gpu}, step=step)
        wandb.log({"transitions": config.batch_size * config.seq_len * step, **loss_info}, step=step)

        if step % config.eval_every == 0:
            with Timeit() as timer:
                eval_stats = vec_evaluate(
                    eval_env, model, config.eval_episodes, config.eval_seed, device=DEVICE
                )
            raw_returns = eval_stats.pop("reward_raw")
            raw_depths = eval_stats.pop("depth_raw")
            normalized_scores = tmp_env.get_normalized_score(raw_returns)

            wandb.log({
                "times/evaluation_gpu": timer.elapsed_time_gpu,
                "times/evaluation_cpu": timer.elapsed_time_cpu,
            }, step=step)
            wandb.log({"transitions": config.batch_size * config.seq_len * step, **eval_stats}, step=step)

            if config.checkpoints_path is not None:
                torch.save(model.state_dict(), os.path.join(config.checkpoints_path, f"{step}.pt"))
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
    set_start_method("spawn")
    train()

