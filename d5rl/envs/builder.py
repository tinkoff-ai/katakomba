from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import List, Optional, Tuple

from nle.env.base import NLE

from d5rl.envs import NetHackChallenge
from d5rl.utils.roles import ALLOWED_COMBOS, Alignment, Race, Role, Sex
from d5rl.wrappers import NetHackWrapper


class NetHackEnvBuilder:
    def __init__(self, nethack_env_fn: NLE, wrapper: Optional[NetHackWrapper]):
        """
        To keep in mind:
          - Not all combinations of character options are allowed by NetHack, we will filter it for you.
          - Default behavior is to build environments for each allowed combination, no seeds fixed (i.e. complete evaluation)
        """
        self._env_fn = nethack_env_fn
        self._env_wrapper = wrapper

        self._races = [race for race in Race]
        self._roles = [role for role in Role]
        self._sex = [sex for sex in Sex]
        self._alignments = [alignment for alignment in Alignment]
        self._eval_seeds = None
        self._train_seeds = None

    def races(self, races: List[Race]) -> NetHackEnvBuilder:
        self._races = races
        return self

    def roles(self, roles: List[Role]) -> NetHackEnvBuilder:
        self._roles = roles
        return self

    def sex(self, sex: List[Sex]) -> NetHackEnvBuilder:
        self._sex = sex
        return self

    def alignments(self, alignments: List[Alignment]) -> NetHackEnvBuilder:
        self._alignments = alignments
        return self

    def eval_seeds(self, seeds: List[int]) -> NetHackEnvBuilder:
        self._eval_seeds = seeds
        return self

    def train_seeds(self, seeds: List[int]) -> NetHackEnvBuilder:
        self._train_seeds = seeds

    def evaluate(self):
        """
        An iterator over the NLE settings to evaluate against.
        """

        all_valid_combinations = deepcopy(ALLOWED_COMBOS)
        valid_combinations = set()

        # Filter only allowed game settings
        for role, race, alignment, sex in product(
            self._roles, self._races, self._alignments, self._sex
        ):
            if (role, race, alignment) in all_valid_combinations:
                # Valkyries do not have sex
                if role is Role.VALKYRIE:
                    valid_combinations.add((role, race, alignment, None))
                else:
                    valid_combinations.add((role, race, alignment, sex))

        # Generate character descriptions for underlying NetHack engine
        eval_characters = []
        for (role, race, alignment, sex) in valid_combinations:
            if sex is not None:
                eval_characters.append(
                    f"{role.value}-{race.value}-{alignment.value}-{sex.value}"
                )
            else:
                eval_characters.append(f"{role.value}-{race.value}-{alignment.value}")

        # Environment and its wrapper are dataset-dependent (wrappers are needed for producing images of tty)
        if self._env_wrapper:
            env_fn = lambda char: self._env_wrapper(
                self._env_fn(character=char, savedir=False)
            )
        else:
            env_fn = lambda char: self._env_fn(character=char, savedir=False)

        # Generate nethack challenges
        for character in sorted(eval_characters):
            if self._eval_seeds is None:
                yield character, env_fn(character), None
            else:
                for seed in self._eval_seeds:
                    yield character, env_fn(character), seed

    def get_action_dim(self) -> int:
        if self._env_wrapper:
            env_fn = lambda char: self._env_wrapper(
                self._env_fn(character=char, savedir=False)
            )
        else:
            env_fn = self._env_fn

        # Environment with a random character (action space does not depent on the character)
        dummy_env = env_fn("@")

        return dummy_env.action_space.n
