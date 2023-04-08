# WARN: this is tmp patch to the Minari combine utils, so that combine make explicit copy instead of the external link.
# Will be merged into Minari in the future as an optional argument.
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.envs.registration import EnvSpec

from minari import DataCollectorV0
from minari.dataset.minari_dataset import MinariDataset, clear_episode_buffer
from minari.storage.datasets_root_dir import get_dataset_path


def combine_datasets(datasets_to_combine, new_dataset_id):
    """Combine a group of MinariDataset in to a single dataset with its own name id.

    A new HDF5 metadata attribute will be added to the new dataset called `combined_datasets`. This will
    contain a list of strings with the dataset names that were combined to form this new Minari dataset.

    Args:
        datasets_to_combine (list[MinariDataset]): list of datasets to be combined
        new_dataset_id (str): name id for the newly created dataset
    """
    new_dataset_path = get_dataset_path(new_dataset_id)

    # Check if dataset already exists
    if not os.path.exists(new_dataset_path):
        new_dataset_path = os.path.join(new_dataset_path, "data")
        os.makedirs(new_dataset_path)
        new_data_path = os.path.join(new_dataset_path, "main_data.hdf5")
    else:
        raise ValueError(
            f"A Minari dataset with ID {new_dataset_id} already exists and it cannot be overridden. Please use a different dataset name or version."
        )

    with h5py.File(new_data_path, "a", track_order=True) as combined_data_file:
        combined_data_file.attrs["total_episodes"] = 0
        combined_data_file.attrs["total_steps"] = 0
        combined_data_file.attrs["dataset_id"] = new_dataset_id

        combined_data_file.attrs["combined_datasets"] = [
            dataset.spec.dataset_id for dataset in datasets_to_combine
        ]

        current_env_spec = None

        for dataset in datasets_to_combine:
            if not isinstance(dataset, MinariDataset):
                raise ValueError(f"The dataset {dataset} is not of type MinariDataset.")
            dataset_env_spec = dataset.spec.env_spec

            assert isinstance(dataset_env_spec, EnvSpec)
            # We have to check that all datasets can be merged by checking that they come from the same
            # environments. However, we override the time limit max_episode_steps with the max among all
            # the datasets to be combined. Then we check if the rest of the env_spec attributes are from
            # the same environment.
            if current_env_spec is None:
                current_env_spec = dataset_env_spec
            elif dataset_env_spec.max_episode_steps is not None:
                if current_env_spec.max_episode_steps is None:
                    current_env_spec.max_episode_steps = (
                        dataset_env_spec.max_episode_steps
                    )
                else:
                    if (
                        current_env_spec.max_episode_steps
                        < dataset_env_spec.max_episode_steps
                    ):
                        current_env_spec.max_episode_steps = (
                            dataset_env_spec.max_episode_steps
                        )
                    else:
                        dataset_env_spec.max_episode_steps = (
                            current_env_spec.max_episode_steps
                        )

            if current_env_spec != dataset_env_spec:
                raise ValueError(
                    "The datasets to be combined have different values for `env_spec` attribute."
                )

            if combined_data_file.attrs.get("flatten_action") is None:
                combined_data_file.attrs[
                    "flatten_action"
                ] = dataset.spec.flatten_actions
            else:
                if (
                    combined_data_file.attrs["flatten_action"]
                    != dataset.spec.flatten_actions
                ):
                    raise ValueError(
                        "The datasets to be combined have different values for `flatten_action` attribute."
                    )

            if combined_data_file.attrs.get("flatten_observation") is None:
                combined_data_file.attrs[
                    "flatten_observation"
                ] = dataset.spec.flatten_observations
            else:
                if (
                    combined_data_file.attrs["flatten_observation"]
                    != dataset.spec.flatten_observations
                ):
                    raise ValueError(
                        "The datasets to be combined have different values for `flatten_observation` attribute."
                    )

            last_episode_id = combined_data_file.attrs["total_episodes"]

            with h5py.File(dataset.spec.data_path, "r") as dataset_file:
                for id in range(dataset.total_episodes):
                    dataset_file.copy(dataset_file[f"episode_{id}"], combined_data_file, name=f"episode_{last_episode_id + id}")
                    combined_data_file[f"episode_{last_episode_id + id}"].attrs.modify("id", last_episode_id + id)
                # combined_data_file[f"episode_{last_episode_id + id}"] = h5py.ExternalLink(dataset.spec.data_path, f"/episode_{id}")
                # combined_data_file[f"episode_{last_episode_id + id}"].attrs.modify("id", last_episode_id + id)

            # Update metadata of minari dataset
            combined_data_file.attrs.modify(
                "total_episodes", last_episode_id + dataset.total_episodes
            )
            combined_data_file.attrs.modify(
                "total_steps",
                combined_data_file.attrs["total_steps"] + dataset.spec.total_steps,
            )

            with h5py.File(dataset.spec.data_path, "r") as dataset_file:
                combined_data_file.attrs.modify("author", dataset_file.attrs["author"])
                combined_data_file.attrs.modify("author_email", dataset_file.attrs["author_email"])

        assert current_env_spec is not None
        combined_data_file.attrs["env_spec"] = current_env_spec.to_json()

    return MinariDataset(new_data_path)