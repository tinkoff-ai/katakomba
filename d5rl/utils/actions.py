"""
Utilities for better debugging experience connected to actions.
"""

from nle.nethack.actions import ACTIONS, Command, CompassDirection, CompassDirectionLonger, MiscDirection, MiscAction, TextCharacters
from typing import Union, Optional


def nle_action_to_gym_action(
    nle_action: Union[Command, CompassDirection, CompassDirectionLonger, MiscDirection, MiscAction, TextCharacters]
) -> Optional[int]:
    """
    Returns gym action index for a specific NLE action.
    """
    
    for ind, nl_action in enumerate(ACTIONS):
        if nl_action == nle_action:
            return ind

    return None

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