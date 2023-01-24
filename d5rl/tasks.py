from __future__ import annotations

from typing import List, Tuple

from d5rl.datasets.builder import AutoAscendDatasetBuilder
from d5rl.envs import NetHackChallenge
from d5rl.envs.builder import NetHackEnvBuilder
from d5rl.wrappers import TTYWrapper

TASKS = {
    "NetHackScore-v0-tty-bot-v0": {
        "env_fn": NetHackChallenge,
        "wrapper_fn": TTYWrapper,
        "dataset_builder_fn": AutoAscendDatasetBuilder,
    }
}


def make_task_builder(task: str) -> Tuple[NetHackEnvBuilder, AutoAscendDatasetBuilder]:
    """
    Creates environment and dataset builders for a task, which you can further configure for your needs.
    """
    if task not in TASKS:
        raise Exception(f"There is no such task: {task}")

    env_fn = TASKS[task]["env_fn"]
    wrapper_fn = TASKS[task]["wrapper_fn"]
    dataset_builder_fn = TASKS[task]["dataset_builder_fn"]

    return NetHackEnvBuilder(env_fn, wrapper_fn), dataset_builder_fn()
