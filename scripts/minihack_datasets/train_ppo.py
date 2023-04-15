import os
import ray
import pyrallis

from glob import glob
from pyrallis import field
from typing import Optional
from dataclasses import dataclass

from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig

from env import RLLlibMinihackEnv
from models.model import RLLibMiniHackModel
from utils import combine_datasets


@dataclass
class ModelConfig:
    embedding_dim: int = 64  # use at least 32, 64 is stronger
    hidden_dim: int = 256  # use at least 128, 256 is stronger
    crop_model: str = "cnn"
    crop_dim: int = 9  # size of crop
    glyph_type: str = "all_cat"  # full, group_id, color_char, all, all_cat* (all_cat best, full fastest)
    use_index_select: bool = True  # use index select instead of normal embedding lookup
    layers: int = 5  # number of cnn layers for crop/glyph model
    msg_model: Optional[str] = None  # character model: none, lt_cnn*, cnn, gru, lstm
    msg_hidden_dim: int = 256  # recommend 256
    msg_embedding_dim: int = 64  # recommend 64
    equalize_input_dim: bool = False  # project inputs to same dim (*false unless doing dynamics)
    equalize_factor: int = 2  # multiplies hdim by this when equalize is enabled (2 > 1)


@dataclass
class TrainConfig:
    # wandb config
    project: str = "MiniHack"
    entity: str = "howuhh"
    # data set config
    version: int = 0
    data_path: str = "minihack_data"
    env_name: str = "MiniHack-Room-Trap-15x15-v0"
    # ray config
    num_gpus: int = 1
    num_cpus: int = 32
    num_actors: int = 128
    # algo config
    gamma: float = 0.999
    learning_rate: float = 2e-4
    grad_clip: float = 40.0
    total_steps: int = 2_000_000
    lstm_seq_len: int = 80
    # ppo config
    model_config: ModelConfig = field(default_factory=ModelConfig)
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 2
    rollout_fragment_length: int = 128
    entropy_coef: float = 0.0001
    vf_loss_coef: float = 0.5
    kl_coeff: float = 0.0
    clip_param: float = 0.2
    lambda_: float = 0.95
    # other config
    deterministic_eval: bool = False
    seed: int = 42

    def __post_init__(self):
        self.group = f"Datasets-v{self.version}"
        self.data_path = os.path.join(self.data_path, f"{self.env_name}", f"v{self.version}")


@pyrallis.wrap()
def train(config: TrainConfig):
    os.makedirs(config.data_path, exist_ok=True)
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"  # only log to wandb

    ray.init(num_gpus=config.num_gpus, num_cpus=config.num_cpus + 1)
    ModelCatalog.register_custom_model("rllib_minihack_model", RLLibMiniHackModel)

    algo_config = (
        PPOConfig()
        .environment(RLLlibMinihackEnv, env_config={"config": config})
        .framework("torch")
        .training(
            model={
                "custom_model": "rllib_minihack_model",
                "custom_model_config": {"config": config.model_config},
                "use_lstm": True,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
                "lstm_cell_size": config.model_config.hidden_dim,
                "max_seq_len": config.lstm_seq_len,
                "vf_share_layers": False
            },
            lr_schedule=[[0, config.learning_rate], [config.total_steps, 0.0]],
            gamma=config.gamma,
            grad_clip=config.grad_clip,
            shuffle_sequences=True,
            train_batch_size=config.num_actors * config.rollout_fragment_length,
            sgd_minibatch_size=config.sgd_minibatch_size,
            num_sgd_iter=config.num_sgd_iter,
            entropy_coeff=config.entropy_coef,
            vf_loss_coeff=config.vf_loss_coef,
            kl_coeff=config.kl_coeff,
            lambda_=config.lambda_,
            clip_param=config.clip_param,
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=50,
            evaluation_config={"explore": not config.deterministic_eval}
        )
        .rollouts(
            num_rollout_workers=config.num_cpus,
            num_envs_per_worker=int(config.num_actors / config.num_cpus),
            rollout_fragment_length=config.rollout_fragment_length,
        )
        .resources(num_gpus=config.num_gpus)
        .debugging(seed=config.seed)
        .environment(disable_env_checking=True)
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": config.total_steps},
            callbacks=[
                WandbLoggerCallback(
                    api_key=os.environ.get("WANDB_API_KEY"),
                    project=config.project,
                    entity=config.entity,
                    group=config.group,
                    name=f"{config.env_name}-dataset-v{config.version}"
                )
            ]
        ),
        param_space=algo_config
    )
    tuner.fit()

    # combining datasets from all workers then delete them
    datasets_paths = glob(f"{config.data_path}/{config.env_name}-*-v{config.version}.hdf5")
    combine_datasets(
        datasets_paths,
        new_path=os.path.join(config.data_path, f"{config.env_name}-dataset-v{config.version}.hdf5")
    )
    for path in datasets_paths:
        if "dataset" not in path:
            os.remove(path)


if __name__ == "__main__":
    train()
