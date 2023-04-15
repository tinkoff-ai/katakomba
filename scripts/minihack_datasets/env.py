import os
import h5py
import numpy as np
import gym as old_gym
import gymnasium as new_gym
from minihack.envs import (
    corridor,
    room,
    skills_simple,
)
from uuid import uuid4
from collections import defaultdict

MINIHACK_ENVS = {
    # Room
    "MiniHack-Room-Random-15x15-v0": room.MiniHackRoom15x15Random,
    "MiniHack-Room-Dark-15x15-v0": room.MiniHackRoom15x15Dark,
    "MiniHack-Room-Monster-15x15-v0": room.MiniHackRoom15x15Monster,
    "MiniHack-Room-Trap-15x15-v0": room.MiniHackRoom15x15Trap,
    "MiniHack-Room-Ultimate-15x15-v0": room.MiniHackRoom15x15Ultimate,
    # Corridor
    "MiniHack-Corridor-R2-v0": corridor.MiniHackCorridor2,
    "MiniHack-Corridor-R3-v0": corridor.MiniHackCorridor3,
    "MiniHack-Corridor-R5-v0": corridor.MiniHackCorridor5,
    # Simple Skills
    "MiniHack-Eat-v0": skills_simple.MiniHackEat,
    "MiniHack-Pray-v0": skills_simple.MiniHackPray,
    "MiniHack-Wear-v0": skills_simple.MiniHackWear,
    "MiniHack-LockedDoor-v0": skills_simple.MiniHackLockedDoor,
}
MINIHACK_OBS = "glyphs,chars,colors,specials,blstats,message,tty_chars,tty_colors,tty_cursor"


def convert_to_gymnasium_space(space):
    if isinstance(space, old_gym.spaces.Dict):
        return new_gym.spaces.Dict({
            k: convert_to_gymnasium_space(v) for k, v in space.items()
        })
    elif isinstance(space, old_gym.spaces.Box):
        return new_gym.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype
        )
    elif isinstance(space, old_gym.spaces.Discrete):
        return new_gym.spaces.Discrete(n=space.n)
    else:
        raise RuntimeError("Unknown space!")


class MiniHackGymnasiumAdapter(new_gym.Env):
    def __init__(self, minihack_env_name, *args, **kwargs):
        minihack_env_cls = MINIHACK_ENVS[minihack_env_name]

        self.env = minihack_env_cls(*args, **kwargs)
        self.observation_space = convert_to_gymnasium_space(self.env.observation_space)
        self.action_space = convert_to_gymnasium_space(self.env.action_space)

    def reset(self, *, seed=None, options=None):
        self.env.seed(seed)
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, truncated, info = self.env.step(action)
        terminated = info["end_status"].name == "TASK_SUCCESSFUL"
        truncated = truncated and info["end_status"].name != "TASK_SUCCESSFUL"
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class MiniHackInMemoryDataCollector(new_gym.Wrapper):
    def __init__(self, env, data_path):
        super().__init__(env)
        self.data_path = data_path
        self.buffer = [defaultdict(list)]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        if len(self.buffer[-1]["tty_chars"]) != 0:
            # for some reason ray sometimes resetting same envs multiple times,
            # so we will keep only last reset obs as init (hopefully this is right obs for this episode)
            assert len(self.buffer[-1]["tty_chars"]) == 1
            self.buffer[-1]["tty_chars"][0] = obs["tty_chars"].copy()
            self.buffer[-1]["tty_colors"][0] = obs["tty_colors"].copy()
            self.buffer[-1]["tty_cursor"][0] = obs["tty_cursor"].copy()
        else:
            self.buffer[-1]["tty_chars"].append(obs["tty_chars"].copy())
            self.buffer[-1]["tty_colors"].append(obs["tty_colors"].copy())
            self.buffer[-1]["tty_cursor"].append(obs["tty_cursor"].copy())

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.buffer[-1]["actions"].append(action)
        self.buffer[-1]["tty_chars"].append(obs["tty_chars"].copy())
        self.buffer[-1]["tty_colors"].append(obs["tty_colors"].copy())
        self.buffer[-1]["tty_cursor"].append(obs["tty_cursor"].copy())
        self.buffer[-1]["rewards"].append(reward)
        self.buffer[-1]["terminations"].append(terminated)
        self.buffer[-1]["truncations"].append(truncated)

        if terminated or truncated:
            self.buffer.append(defaultdict(list))

        return obs, reward, terminated, truncated, info

    def close(self):
        with h5py.File(self.data_path, "w", track_order=True) as df:
            for i, episode in enumerate(self.buffer):
                # skip empty episode if any
                if len(episode["actions"]) == 0:
                    continue
                # skip episode if it is not ended
                if not (episode["truncations"][-1] or episode["terminations"][-1]):
                    continue

                assert len(episode["tty_chars"]) == len(episode["actions"]) + 1
                gp = df.create_group(f"episode_{i}")

                gp.attrs["total_steps"] = len(episode["actions"])
                gp.create_dataset("observations/tty_chars", data=np.stack(episode["tty_chars"]), compression="gzip")
                gp.create_dataset("observations/tty_colors", data=np.stack(episode["tty_colors"]), compression="gzip")
                gp.create_dataset("observations/tty_cursor", data=np.stack(episode["tty_cursor"]), compression="gzip")

                gp.create_dataset("actions", data=np.array(episode["actions"]), dtype="uint8", compression="gzip")
                gp.create_dataset("rewards", data=np.array(episode["rewards"]), dtype="float16", compression="gzip")
                gp.create_dataset("terminations", data=np.array(episode["terminations"]), compression="gzip")
                gp.create_dataset("truncations", data=np.array(episode["truncations"]), compression="gzip")

        return self.env.close()


class RLLlibMinihackEnv(new_gym.Env):
    def __init__(self, env_config):
        self.env_config = env_config["config"]
        dataset_name = f"{self.env_config.env_name}-{str(uuid4())}-v{self.env_config.version}.hdf5"

        new_gym.register(
            id=self.env_config.env_name,
            entry_point=MiniHackGymnasiumAdapter,
            kwargs={
                "minihack_env_name": self.env_config.env_name,
                "observation_keys": MINIHACK_OBS.split(",")
            }
        )
        env = new_gym.make(self.env_config.env_name)
        env = MiniHackInMemoryDataCollector(
            env,
            data_path=os.path.join(self.env_config.data_path, dataset_name)
        )
        env.spec.max_episode_steps = env.unwrapped.env._max_episode_steps

        self.env = env
        self.spec = env.spec
        self.observation_space = new_gym.spaces.Dict(self.__filter_tty(env.observation_space))
        self.action_space = env.action_space

    def __filter_tty(self, obs_dict):
        return {
            k: v for k, v in obs_dict.items()
            if k not in ["tty_chars", "tty_colors", "tty_cursor"]
        }

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.__filter_tty(obs), info

    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        return self.__filter_tty(obs), reward, truncated, terminated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()