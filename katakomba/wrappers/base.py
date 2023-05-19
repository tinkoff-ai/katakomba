from typing import Optional, Tuple

import gym
from nle import nethack
from nle.env.base import NLE


class NetHackWrapper(gym.Wrapper):
    """
    NetHack needs a modified gym-wrapper due to its multiple-seeding strategy.
    """

    def __init__(self, env: NLE):
        super().__init__(env)

        self.env: NLE = env

    def seed(
        self,
        core: Optional[int] = None,
        disp: Optional[int] = None,
        reseed: bool = False,
    ) -> Tuple[int, int, bool]:
        """
        Sets the state of the NetHack RNGs after the next reset.

        NetHack 3.6 uses two RNGs, core and disp. This is to prevent
        RNG-manipulation by e.g. running into walls or other no-ops on the
        actual game state. This is a measure against "tool-assisted
        speedruns" (TAS). NLE can run in both NetHack's default mode and in
        TAS-friendly "no reseeding" if `reseed` is set to False, see below.

        Arguments:
            core [int or None]: Seed for the core RNG. If None, chose a random
                value.
            disp [int or None]: Seed for the disp (anti-TAS) RNG. If None, chose
                a random value.
            reseed [boolean]: As an Anti-TAS (automation) measure,
                NetHack 3.6 reseeds with true randomness every now and then. This
                flag enables or disables this behavior. If set to True, trajectories
                won't be reproducible.

        Returns:
            [tuple] The seeds supplied, in the form (core, disp, reseed).
        """
        return self.env.seed(core, disp, reseed)

    def get_current_depth(self) -> int:
        """
        This returns the depth your agent is at. Note that it's not the same as the dungeon level.
        https://nethackwiki.com/wiki/Dungeon_level

        Also note that this is not representative of how well your agent's doing after you descended to the amulet of yendor.
        But for current state-of-the-art this is good enough.
        We do not use dungeon's level as in some cases it can be biased by the agent's experience.
        """
        return int(
            self.env.last_observation[self.env._blstats_index][nethack.NLE_BL_DEPTH]
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["current_depth"] = self.get_current_depth()
        return obs, reward, done, info
