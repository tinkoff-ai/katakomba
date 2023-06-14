"""
Adopted from here: https://github.com/facebookresearch/nle/blob/4f5b57ea0e18f80e40fbdc33c1dbf94bbd265e42/nle/env/tasks.py#L9

Changes:
1. Removed raise on the seed setting for NetHackChallenge.
2. Added get_normalized_score method
3. Added get_dataset method
4. Added get_depth method
"""
import gym

import nle
import numpy as np
from nle import nethack
from nle.env.tasks import NetHackScore

from katakomba.utils.scores import MEAN_SCORES_AUTOASCEND
from katakomba.utils.datasets.small_scale import NLDSmallDataset
from katakomba.utils.datasets.large_scale import load_nld_aa_large_dataset
from katakomba.utils.roles import Role, Race, Alignment

from typing import Optional, Tuple, Union


class NetHackChallenge(NetHackScore):
    """Environment for the NetHack Challenge.

    The task is an augmentation of the standard NLE task. This is the NLE Score Task
    but with some subtle differences:
        * the action space is fixed to include the full keyboard
        * menus and "<More>" tokens are not skipped
        * starting character is randomly assigned
    """

    def __init__(
        self,
        *args,
        character="@",
        allow_all_yn_questions=True,
        allow_all_modes=True,
        penalty_mode="constant",
        penalty_step: float = -0.00,
        penalty_time: float = -0.0,
        max_episode_steps: int = 1e6,
        observation_keys=(
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
            "misc",
        ),
        no_progress_timeout: int = 1000,
        **kwargs,
    ):
        actions = nethack.ACTIONS
        kwargs["wizard"] = False
        super().__init__(
            *args,
            actions=actions,
            character=character,
            allow_all_yn_questions=allow_all_yn_questions,
            allow_all_modes=allow_all_modes,
            penalty_mode=penalty_mode,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
            **kwargs,
        )
        # If the in-game turn count doesn't change for N steps, we abort
        self.no_progress_timeout = no_progress_timeout

    def reset(self, *args, **kwargs):
        self._turns = None
        self._no_progress_count = 0
        return super().reset(*args, **kwargs)

    def _check_abort(self, observation):
        """Check if time has stopped and no observations has changed long enough
        to trigger an abort."""

        turns = observation[self._blstats_index][nethack.NLE_BL_TIME]
        if self._turns == turns:
            self._no_progress_count += 1
        else:
            self._turns = turns
            self._no_progress_count = 0
        return (
            self._steps >= self._max_episode_steps
            or self._no_progress_count >= self.no_progress_timeout
        )


class OfflineNetHackChallengeWrapper(gym.Wrapper):
    """
    Offline NetHackChallenge wrappers. Adds normalized scores and dataset loading.
    """
    def __init__(self, env: nle.env.NLE):
        super().__init__(env)
        self.env = env

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

    def get_normalized_score(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
         Returns score normalized against AutoAscend bot scores achieved for this exact character.
        """
        if self.character.count("-") != 2:
            raise ValueError("Reference score is not provided for this character.")

        role, race, align = self.character.split("-")
        role, race, align = Role(role), Race(race), Alignment(align)

        ref_mean_score = MEAN_SCORES_AUTOASCEND[(role, race, align)]

        return score / ref_mean_score

    def get_current_depth(self):
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

    def get_dataset(self, scale: str = "small", **kwargs):
        if self.character.count("-") != 2:
            raise ValueError("Reference score is not provided for this character.")

        role, race, align = self.character.split("-")
        role, race, align = Role(role), Race(race), Alignment(align)

        if scale == "small":
            return NLDSmallDataset(role, race, align, **kwargs)
        elif scale == "big":
            return load_nld_aa_large_dataset(
                role=role,
                race=race,
                align=align,
                **kwargs
            )
        else:
            raise RuntimeError(
                "Unknown dataset scale. Please specify 'small' for small"
                " scale dataset and 'big' for NLD-AA full dataset."
            )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["current_depth"] = self.get_current_depth()
        return obs, reward, done, info
