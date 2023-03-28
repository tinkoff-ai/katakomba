"""
This is taken from the original Dungeons&Data implementation.
"""

import numpy as np
import cv2
import os
import render_utils

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

PIXEL_SIZE = 6
CROP_SIZE = 18
FONT_SIZE = 9
RESCALE_FONT_SIZE = (6, 6)
SMALL_FONT_PATH = os.path.abspath("katakomba/utils/render_utils/Hack-Regular.ttf")

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right
# https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",  # - flipped these ones around
    "#C0C0C0",  # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


def _initialize_char_array(font_size, rescale_font_size):
    """
    Draw all characters in PIL and cache them in numpy arrays
    if rescale_font_size is given, assume it is (width, height)
    Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
    """
    print(SMALL_FONT_PATH)
    font = ImageFont.truetype(SMALL_FONT_PATH, font_size)
    dummy_text = "".join(
        [(chr(i) if chr(i).isprintable() else " ") for i in range(256)]
    )
    _, _, image_width, image_height = font.getbbox(dummy_text)
    # Above can not be trusted (or its siblings)....
    image_width = int(np.ceil(image_width / 256) * 256)

    char_width = rescale_font_size[0]
    char_height = rescale_font_size[1]

    char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
    image = Image.new("RGB", (image_width, image_height))
    image_draw = ImageDraw.Draw(image)
    for color_index in range(16):
        image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
        image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

        arr = np.array(image).copy()
        arrs = np.array_split(arr, 256, axis=1)
        for char_index in range(256):
            char = arrs[char_index]
            if rescale_font_size:
                char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
            char_array[char_index, color_index] = char
    return char_array


CHAR_ARRAY = _initialize_char_array(FONT_SIZE, RESCALE_FONT_SIZE)
CHAR_HEIGHT = CHAR_ARRAY.shape[2]
CHAR_WIDTH = CHAR_ARRAY.shape[3]
CHAR_ARRAY = np.ascontiguousarray(CHAR_ARRAY.transpose(0, 1, 4, 2, 3))
SCREEN_SHAPE = (3, CROP_SIZE * CHAR_WIDTH, CROP_SIZE * CHAR_HEIGHT)


def render_screen_image(
    tty_chars: np.ndarray,
    tty_colors: np.ndarray,
    tty_cursor: np.ndarray,
    threadpool: Optional[ThreadPoolExecutor] = None,
) -> np.ndarray:
    """
    tty_chars: [batch_size, seq_len, 24, 80]
    tty_colors: [batch_size, seq_len, 24, 80]
    tty_cursor: [batch_size, seq_len, 1]
    """
    batch_size = tty_chars.shape[0]
    seq_len = tty_chars.shape[1]
    screen_image = np.zeros((batch_size, seq_len) + SCREEN_SHAPE, dtype=np.uint8)

    cursor_uint8 = tty_cursor.astype(np.uint8)
    convert = lambda i: render_utils.render_crop(
        tty_chars[i],
        tty_colors[i],
        cursor_uint8[i],
        CHAR_ARRAY,
        screen_image[i],
        CROP_SIZE,
    )

    if threadpool is not None:
        list(threadpool.map(convert, range(batch_size)))
    else:
        list(map(convert, range(batch_size)))

    return screen_image
