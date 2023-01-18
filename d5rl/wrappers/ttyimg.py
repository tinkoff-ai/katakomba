"""
Taken & adapted from Chaos Dwarf in Nethack Challenge Starter Kit:
https://github.com/Miffyli/nle-sample-factory-baseline

UNDER CONSTRUCTION
"""

import os

import cv2
import gym
import numpy as np
import render_utils

from nle import nethack
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from nle.env import base
from typing import Tuple
from d5rl.wrappers.base import NetHackWrapper

SMALL_FONT_PATH = os.path.abspath("Hack-Regular.ttf")

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
    """Draw all characters in PIL and cache them in numpy arrays

    if rescale_font_size is given, assume it is (width, height)

    Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
    """
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


class TTYImageWrapper(NetHackWrapper):
    """
    An observation wrapper that converts tty_* to an image.
    This is an egocentric version! (not the whole terminal)
    """

    def __init__(
        self,
        env              : base.NLE,
        font_size        : int             = 9,
        crop_size        : int             = 12,
        rescale_font_size: Tuple[int, int] = (6, 6),
    ):
        super().__init__(env)
        self.char_array  = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width  = self.char_array.shape[3]

        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)
        self.char_array = np.ascontiguousarray(self.char_array)
        self.crop_size  = crop_size

        crop_rows = crop_size or nethack.nethack.TERMINAL_SHAPE[0]
        crop_cols = crop_size or nethack.nethack.TERMINAL_SHAPE[1]

        self.chw_image_shape = (
            3,
            crop_rows * self.char_height,
            crop_cols * self.char_width,
        )

        self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
        )

    def _get_observation(self, observation):
        screen = np.zeros(self.chw_image_shape, order="C", dtype=np.uint8)
        render_utils.render_crop(
            observation["tty_chars"],
            observation["tty_colors"],
            observation["tty_cursor"],
            self.char_array,
            screen,
            crop_size=self.crop_size,
        )
        return screen

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._get_observation(obs)