"""
Utilities for better debugging experience connected to actions.
"""
from typing import Optional, Union

import numpy as np
from nle.nethack.actions import (
    ACTIONS,
    Command,
    CompassDirection,
    CompassDirectionLonger,
    MiscAction,
    MiscDirection,
    TextCharacters,
)

_ASCII_TO_GYM_ACTION = {action.value: ind for ind, action in enumerate(ACTIONS)}


def nle_action_to_gym_action(
    nle_action: Union[
        Command,
        CompassDirection,
        CompassDirectionLonger,
        MiscDirection,
        MiscAction,
        TextCharacters,
    ]
) -> Optional[int]:
    """
    Returns gym action index for a specific NLE action.
    """

    for ind, nl_action in enumerate(ACTIONS):
        if nl_action == nle_action:
            return ind

    return None


def ascii_action_to_gym_action(ascii_action: int) -> int:
    """
    Return gym action index for ascii action (useful for AA dataset).
    """
    return _ASCII_TO_GYM_ACTION[ascii_action]


def ascii_actions_to_gym_actions(ascii_actions: np.ndarray) -> np.ndarray:
    """
    Args
      ascii_actions: any size
    Returns
      gym_actions: any size
    """
    return np.vectorize(ascii_action_to_gym_action)(ascii_actions)


def yes_nle_action():
    """
    There is no enum value for yes/no actions :shrug:
    But they can be used by compass direction actions :kekw:
    """
    return CompassDirection.NW


def no_nle_action():
    """
    There is no enum value for yes/no actions :shrug:
    But they can be used by compass direction actions :kekw:
    """
    return CompassDirection.SE
