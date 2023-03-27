from __future__ import annotations

from typing import Tuple

from katakomba.datasets.builder import AutoAscendDatasetBuilder
from katakomba.envs import NetHackChallenge
from katakomba.envs.builder import NetHackEnvBuilder
from katakomba.wrappers import TTYWrapper, CropRenderWrapper

TASKS = {
    "NetHackScore-v0-tty-bot-v0": {
        "env_fn": NetHackChallenge,
        "wrapper_fn": TTYWrapper,
        "dataset_builder_fn": AutoAscendDatasetBuilder,
    },
    "NetHackScore-v0-ttyimg-bot-v0": {
        "env_fn": NetHackChallenge,
        "wrapper_fn": CropRenderWrapper,
        "dataset_builder_fn": AutoAscendDatasetBuilder,
    },
}


def make_task_builder(
    task: str, data_path: str = "data/nle_data", db_path: str = "ttyrecs.db"
) -> Tuple[NetHackEnvBuilder, AutoAscendDatasetBuilder]:
    """
    Creates environment and dataset builders for a task, which you can further configure for your needs.
    """
    if task not in TASKS:
        raise Exception(f"There is no such task: {task}")

    env_fn = TASKS[task]["env_fn"]
    wrapper_fn = TASKS[task]["wrapper_fn"]
    dataset_builder_fn = TASKS[task]["dataset_builder_fn"]

    return NetHackEnvBuilder(env_fn, wrapper_fn), dataset_builder_fn(
        path=data_path, db_path=db_path
    )
