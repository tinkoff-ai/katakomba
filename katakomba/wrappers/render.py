import gym
import numpy as np

from katakomba.utils.render import SCREEN_SHAPE, render_screen_image
from katakomba.env import OfflineNetHackChallengeWrapper


class CropRenderWrapper(OfflineNetHackChallengeWrapper):
    """
    Populates observation with:
        - screen_image: [3, crop_width, crop_height]. For specific values see d5rl/utils/render.py
        - tty_chars
        - tty_colors
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        obs_spaces = {
            "screen_image": gym.spaces.Box(
                low=0, high=255, shape=SCREEN_SHAPE, dtype=np.uint8
            )
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _populate_obs(self, obs):
        obs["screen_image"] = render_screen_image(
            tty_chars=obs["tty_chars"][np.newaxis, np.newaxis, ...],
            tty_colors=obs["tty_colors"][np.newaxis, np.newaxis, ...],
            tty_cursor=obs["tty_cursor"][np.newaxis, np.newaxis, ...],
        )
        obs["screen_image"] = np.squeeze(obs["screen_image"])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._populate_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._populate_obs(obs)
        return obs
