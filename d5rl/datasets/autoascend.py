import numpy as np
import sys

from torch.utils.data import IterableDataset
from nle.dataset.dataset import TtyrecDataset
from d5rl.utils.observations import tty_to_numpy

from typing import List, Any
from torch import tensor
from copy import deepcopy
from dataclasses import dataclass


class _PrefetchedGame:
    states : List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones  : List[bool]
    indices: List[int]

    def __init__(self):
        self.cur_ind = 0

    def is_over(self) -> bool:
        if self.cur_ind >= self.states.shape[0]:
            return True
        else:
            return False

    def pop_tuple(self):
        index = self.indices[self.cur_ind]

        # Set next state to be the same if it's done
        n_index = index + 1
        if n_index >= self.states.shape[0]:
            n_index = index

        self.cur_ind += 1

        return self.states[index], self.actions[index], self.rewards[index], self.states[n_index], self.dones[index]

    def set_data(self, states, actions, rewards, dones):
        self.states  = states
        self.actions = actions
        self.rewards = rewards
        self.dones   = dones

        num_states   = self.states.shape[0]
        self.indices = np.random.choice(range(0, num_states), size=num_states, replace=False)

class _AutoAscendIterator:
    def __init__(
        self, 
        ttyrecdata      : TtyrecDataset,
        batch_size      : int,
        n_prefetch_games: int = 8
    ):
        if batch_size % n_prefetch_games != 0:
            raise Exception(f"Batch size must be a multiple of n_prefetch_games. In your case a multiple of {n_prefetch_games}")

        self._ttyrecdata       = ttyrecdata
        self._iterator         = iter(ttyrecdata)
        self._batch_size       = batch_size
        self._n_prefetch_games = n_prefetch_games

    def __iter__(self):
        # Prefetch games for faster/random access
        self._prev_batch = None
        self._prefetched_games: List[_PrefetchedGame] = []
        for _ in range(self._n_prefetch_games):
            self._prefetched_games.append(self._prefetch_game())

        return self

    def __next__(self):
        """
        Returns a usual (s, a, s', r, done), where
            - s is a tty-screen [batch_size, 80, 24, ?] (uint8)
            - a is an action [batch_size, 1] (uint8)
            - s' is a tty-screen [batch_size, 80, 24, ?] (uint8)
            - r is the change in the game score
            - whether the episode ended (game-over usually) (bool)

        [batch_size, seq_len, *]
        """
        # For your mental health, here are the keys
        # dict_keys(['tty_chars', 'tty_colors', 'tty_cursor', 'timestamps', 'done', 'gameids', 'keypresses', 'scores'])

        # Forming a batch
        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []
        for ind in range(self._batch_size):
            game_ind = ind % self._n_prefetch_games
            game = self._prefetched_games[game_ind]
            if game.is_over():
                game = self._prefetch_game()
                self._prefetched_games[game_ind] = game

            s, a, r, next_s, d = game.pop_tuple()

            states.append(np.expand_dims(s, 0))
            actions.append(np.expand_dims(a, 0))
            rewards.append(np.expand_dims(r, 0))
            next_states.append(np.expand_dims(next_s, 0))
            dones.append(np.expand_dims(d, 0))

        return np.concatenate(states), np.concatenate(actions), np.concatenate(rewards), np.concatenate(next_states), np.concatenate(dones) 

    def _prefetch_game(self) -> _PrefetchedGame:
        # If we've just started iterating
        if self._prev_batch is None:
            self._prev_batch = deepcopy(next(self._iterator))

        game    = _PrefetchedGame()
        states  = []
        actions = []
        rewards = []
        dones   = []

        # Previous batch is pointing to the start of the game
        cur_batch = next(self._iterator)
        while True:
            # Todo: need to check for an episode end (where scores are for some reason set to zero for several steps)

            state = tty_to_numpy(
                tty_chars  = self._prev_batch["tty_chars"][0, 0],
                tty_colors = self._prev_batch["tty_colors"][0, 0],
                tty_cursor = self._prev_batch["tty_cursor"][0, 0]
            )
            action = 0 # TODO
            reward = cur_batch["scores"][0, 0] - self._prev_batch["scores"][0, 0] # potentials are better
            done   = cur_batch["done"][0, 0]

            # Make dims = [1, *] to concat further
            states.append(np.expand_dims(state, 0))
            actions.append(np.array(action, dtype=np.uint8).reshape(1, 1))
            rewards.append(np.array(reward, dtype=np.float32).reshape(1, 1))
            dones.append(np.array(done, dtype=bool).reshape(1, 1))

            self._prev_batch = deepcopy(cur_batch)

            # This is where the new episode starts
            if done:
                break
            else:
                cur_batch = next(self._iterator)

        game.set_data(
            np.concatenate(states),
            np.concatenate(actions),
            np.concatenate(rewards),
            np.concatenate(dones)
        )

        self._prev_batch = deepcopy(cur_batch)

        print(f"Size of the game: {(game.states.nbytes+game.actions.nbytes+game.rewards.nbytes+game.dones.nbytes+game.indices.nbytes)/1024/1024}mb")
        print(f"Num tuples: {game.states.shape[0]}")

        return game


class AutoAscendDataset(IterableDataset):
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int):
        self._ttyrecdata = ttyrecdata
        self._batch_size = batch_size

    def __iter__(self):
        return iter(_AutoAscendIterator(self._ttyrecdata, self._batch_size))