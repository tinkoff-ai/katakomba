from typing import Optional, Tuple

import gym
from nle.env.base import NLE


class OfflineNetHackWrapper(gym.Wrapper):
    """
    - NetHack needs a modified gym-wrapper due to its multiple-seeding strategy.
      This is needed for reproducible results. Reseeding is not an issue yet for SOTA ORL algorithms.
    - Normalized scores and levels
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

    def get_normalized_score(
        self, score: int, against: str = "autoascend-bot"
    ) -> float:
        raise NotImplementedError()

    def get_normalized_level(
        self, level: int, against: str = "autoascend-bot"
    ) -> float:
        raise NotImplementedError()
