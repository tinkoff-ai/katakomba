from gym.vector.utils import batch_space
from gym.vector.utils.numpy_utils import concatenate, create_empty_array

from copy import deepcopy
import numpy as np


class NetHackEvalVectorEnv:
    def __init__(self, env_fn, seeds, num_episodes_per_seed, reseed=False):
        self._envs = [env_fn() for _ in range(len(seeds))]

        self.single_observation_space = self._envs[0].observation_space
        self.single_action_space = self._envs[0].action_space

        self.observation_space = batch_space(self.single_observation_space, n=len(seeds))
        self.action_space = batch_space(self.single_action_space, n=len(seeds))

        self.num_episodes_per_seed = num_episodes_per_seed
        self.seeds = seeds
        self.reseed = reseed
        self.num_envs = len(seeds)
        self.__reset_buffers()

    def __reset_buffers(self):
        # logging buffers
        self._total_reward = np.zeros(len(self.seeds), dtype=float)
        self._total_len = np.zeros(len(self.seeds), dtype=float)
        self._num_episodes = np.zeros(len(self.seeds), dtype=int)

        # states buffers
        self._obs = create_empty_array(
            self.single_observation_space, n=len(self.seeds), fn=np.zeros
        )
        self._rewards = np.zeros(len(self.seeds), dtype=np.float64)
        self._dones = np.zeros(len(self.seeds), dtype=bool)

    def reset(self):
        self.__reset_buffers()

        obs = []
        for i in range(len(self._envs)):
            self._envs[i].seed(self.seeds[i], reseed=self.reseed)
            env_obs = self._envs[i].reset()
            obs.append(env_obs)

        self.__fake_obs = obs[0]
        self._obs = concatenate(obs, self._obs, self.single_observation_space)

        return self._obs

    def step(self, actions):
        if self.evaluation_done():
            raise RuntimeError("Can not step further. Evaluation env is exhausted!")

        obs, infos = [], []
        for i in range(len(self._envs)):
            if self._num_episodes[i] >= self.num_episodes_per_seed:
                # TODO: for now, after the limit of episodes for this seed,
                #  we will add fake obs, this is not compute optimal!
                obs.append(self.__fake_obs)
                self._rewards[i] = 0.0
                self._dones[i] = True
                infos.append({})
                continue

            env_obs, self._rewards[i], self._dones[i], env_info = self._envs[i].step(actions[i])

            self._total_reward[i] += self._rewards[i]
            self._total_len[i] += 1

            if self._dones[i]:
                env_info.update(
                    total_return=self._total_reward[i],
                    total_length=self._total_len[i],
                    seed=self.seeds[i],
                    within_seed_episode_idx=self._num_episodes[i]
                )
                # reset counters
                self._num_episodes[i] += 1
                self._total_reward[i] = 0.0
                self._total_len[i] = 0.0

                # TODO: this might be problematic for RNN, as we skip last obs after done
                self._envs[i].seed(self.seeds[i], reseed=self.reseed)
                env_obs = self._envs[i].reset()

            obs.append(env_obs)
            infos.append(env_info)

        self._obs = concatenate(obs, self._obs, self.single_observation_space)
        # TODO: ideally we should copy here, but we will transfer them to tensors anyway
        return (
            self._obs,
            self._rewards,
            self._dones,
            infos
        )

    def evaluation_done(self):
        return np.all(self._num_episodes == self.num_episodes_per_seed)