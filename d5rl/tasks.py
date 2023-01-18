from __future__ import annotations
from copy import deepcopy
from typing import List, Tuple
from itertools import product
from d5rl.envs import NetHackChallenge
from d5rl.wrappers import TTYWrapper
from d5rl.utils.roles import Role, Race, Alignment, Sex, ALLOWED_COMBOS

DATASETS = [
    "NetHackChallenge-v0-tty-bot-v0",
    # TODO
    # "NetHackScore-v0-ttyimg-bot-v0",
    # "NetHackGold-v0-ttyimg-bot-v0",
]
DATASET_TO_ENV = {
    "NetHackChallenge-v0-tty-bot-v0": NetHackChallenge
}


class NetHackEnvBuilder:
    def __init__(self, dataset_name: str):
        """
        To keep in mind:
          - Not all combinations of character options are allowed by NetHack, we will filter it for you.
          - Default behavior is to build environments for each allowed combination, no seeds fixed (i.e. complete evaluation)
        """
        self.dataset_name = dataset_name

        self._races       = [race for race in Race]
        self._roles       = [role for role in Role]
        self._sex         = [sex for sex in Sex]
        self._alignments  = [alignment for alignment in Alignment]
        self._eval_seeds  = None
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

    def alignment(self, alignments: List[Alignment]) -> NetHackEnvBuilder:
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
        valid_combinations     = set()

        # Filter only allowed game settings
        for role, race, alignment, sex in product(self._roles, self._races, self._alignments, self._sex):
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
                eval_characters.append(f"{role.value}-{race.value}-{alignment.value}-{sex.value}")
            else:
                eval_characters.append(f"{role.value}-{race.value}-{alignment.value}")

        # Environment and its wrapper are dataset-dependent (wrappers are needed for producing images of tty)
        env_fn   = DATASET_TO_ENV[self.dataset_name]
        # TODO: Change it to different wrappers when needed
        env_wrap = TTYWrapper

        # Generate nethack challenges
        for character in sorted(eval_characters):
            if self._eval_seeds is None:
                yield character, env_wrap(env_fn(character=character)), None
            else:
                for seed in self._eval_seeds:
                    yield character, env_wrap(env_fn(character=character)), seed


def make_task_builder(dataset: str) -> Tuple[NetHackEnvBuilder, int]:
    """
    Creates an environment and dataset builders, which you can further configure for your needs.
    """
    if dataset not in DATASETS:
        raise Exception(f"There is no such dataset: {dataset}")
        
    return NetHackEnvBuilder(dataset), 10