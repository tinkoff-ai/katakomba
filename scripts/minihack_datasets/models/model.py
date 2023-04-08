import gym
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from .network import MiniHackNetwork


class RLLibMiniHackModel(TorchModelV2, nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: Optional[int],
            model_config: dict,
            name: str,
    ):
        TorchModelV2.__init__(
            self,
            observation_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)
        config = model_config["custom_model_config"]["config"]

        self.num_outputs = config.hidden_dim
        self.base = MiniHackNetwork(observation_space, config)

    @override(TorchModelV2)
    def forward(self, x: Dict[str, Any], *_: Any) -> Tuple[torch.Tensor, list]:
        return self.base(x["obs"]), []
