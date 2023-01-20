import numpy as np 
from nle.nethack.nethack import TERMINAL_SHAPE


def tty_to_numpy(tty_chars, tty_colors, tty_cursor) -> np.ndarray:
    """
    Convers a tty into a numpy array of shape [24, 80, 3].
    24 is a terminal width, 80 -- a height, 3 -- for chars, colors, and cursor position.
    """
    obs    = np.zeros(shape=(TERMINAL_SHAPE[0], TERMINAL_SHAPE[1], 3), dtype=np.uint8)
    cursor = tty_cursor

    obs[:, :, 0] = tty_chars
    obs[:, :, 1] = tty_colors
    obs[cursor[0], cursor[1], 2] = 255

    return obs