import numpy as np 
from nle.nethack.nethack import TERMINAL_SHAPE


def tty_to_numpy(tty_chars, tty_colors, tty_cursor) -> np.ndarray:
    """
    Converts a tty into a numpy array of shape [terminal_width=24, terminal_height=80, depth=3].

    Args:
      tty_chars: [batch_size, 24, 80]
      tty_colors: [batch_size, 24, 80]
      tty_cursor: [batch_size, 2]
    Returns:
      tty_array: [batch_size, 24, 80, 3]
    """
    batch_size = tty_chars.shape[0]
    obs        = np.zeros(shape=(batch_size, TERMINAL_SHAPE[0], TERMINAL_SHAPE[1], 3), dtype=np.uint8)

    obs[:, :, :, 0] = tty_chars
    obs[:, :, :, 1] = tty_colors

    for b in range(batch_size):
        obs[b, tty_cursor[b, 0], tty_cursor[b, 1], 2] = 255    

    return obs