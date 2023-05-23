import gym
import numpy as np
from nle import nethack
from nle.env import base

from katakomba.env import OfflineNetHackChallengeWrapper


class TTYWrapper(OfflineNetHackChallengeWrapper):
    """
    An observation wrapper that converts tty_* to a numpy array.
    """

    def __init__(
        self,
        env: base.NLE,
    ):
        super().__init__(env)

        self.shape = (
            nethack.nethack.TERMINAL_SHAPE[0],
            nethack.nethack.TERMINAL_SHAPE[1],
            3,
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def _get_observation(self, observation):
        tty = np.zeros(self.shape, dtype=np.uint8)
        agent_pos = observation["tty_cursor"]

        tty[:, :, 0] = observation["tty_chars"]
        tty[:, :, 1] = observation["tty_colors"]
        tty[agent_pos[0], agent_pos[1], 2] = 255

        return tty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._get_observation(obs)
