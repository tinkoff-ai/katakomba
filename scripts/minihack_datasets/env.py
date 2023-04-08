import minari
import gym as old_gym
import gymnasium as new_gym
from minihack.envs import (
    corridor,
    room,
    skills_simple,
)
from uuid import uuid4

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
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class RLLlibMinihackEnv(new_gym.Env):
    def __init__(self, env_config):
        self.env_config = env_config["config"]
        env = new_gym.make(
            self.env_config.env_name,
            observation_keys="glyphs,chars,colors,specials,blstats,message,tty_chars,tty_colors,tty_cursor",
            penalty_mode="constant",
            penalty_time=0.0,
            penalty_step=-0.001,
            reward_lose=0,
            reward_win=1,
        )
        self.env = minari.DataCollectorV0(env, max_buffer_episodes=100)

        self.observation_space = new_gym.spaces.Dict(
            self.__filter_tty(env.observation_space)
        )
        self.action_space = env.action_space

    def __filter_tty(self, obs_dict):
        return {
            k: v for k, v in obs_dict.items()
            if k not in ["tty_chars", "tty_colors", "tty_cursor"]
        }

    def reset(self,  *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.__filter_tty(obs), info

    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        return self.__filter_tty(obs), reward, truncated, terminated, info

    def render(self):
        return self.env.render()

    def close(self):
        minari.create_dataset_from_collector_env(
            collector_env=self.env,
            dataset_id=f"{self.env_config.env_name}-{str(uuid4())}-v{self.env_config.version}",
            algorithm_name="RLlibPPO",
            author="howuhh",
            author_email="a.p.nikulin@tinkoff.ai",
            code_permalink="LOL"
        )
        return self.env.close()