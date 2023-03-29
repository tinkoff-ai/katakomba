"""
Memes
 - Screen may not contain the map on the screen (there can be just a menu, or inventory)
 - Default TTYRec dataset fetches datapoints sequentially (i.e. each sample goes one after another within a game)

What to keep in mind:
 - Alignment between dataset actions and environment actions
 - Match terminal sizes (seems that the dataset uses 80x24, but original NLE 79x21)
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import nle.dataset as nld

from katakomba.datasets.base import BaseAutoAscend
from katakomba.datasets.sars_autoascend import SARSAutoAscendTTYDataset
from katakomba.datasets.sa_chaotic_autoascend import SAChaoticAutoAscendTTYDataset
from katakomba.utils.roles import Alignment, Race, Role, Sex


class AutoAscendDatasetBuilder:
    """
    This is the most basic wrapper.
    It obeys the original logic of the TTYRec dataset that samples data within-game-sequentially.
    """

    def __init__(self, path: str = "data/nle_data", db_path: str = "ttyrecs.db"):
        # Create a sql-lite database for keeping trajectories
        if not nld.db.exists(db_path):
            nld.db.create(db_path)
            nld.add_nledata_directory(path, "autoascend", db_path)

        # Create a connection to specify the database to use
        db_conn = nld.db.connect(filename=db_path)
        logging.info(
            f"AutoAscend Dataset has {nld.db.count_games('autoascend', conn=db_conn)} games."
        )

        self.db_path = db_path
        # Pre-init filters
        # Note that all strings are further converted to be first-letter-capitalized
        # This is how it's stored in dungeons data :shrug:
        self._races: Optional[List[str]] = None
        self._game_ids: Optional[List[int]] = None
        self._alignments: Optional[List[str]] = None
        self._sex: Optional[List[str]] = None
        self._roles: Optional[List[str]] = None
        self._game_versions: List[str] = ["3.6.6"]

    def races(self, races: List[Race]) -> AutoAscendDatasetBuilder:
        self._races = [str(race.value).title() for race in races]
        return self

    def roles(self, roles: List[Role]) -> AutoAscendDatasetBuilder:
        self._roles = [str(role.value).title() for role in roles]
        return self

    def sex(self, sex: List[Sex]) -> AutoAscendDatasetBuilder:
        self._sex = [str(s.value).title() for s in sex]
        return self

    def alignments(self, alignments: List[Alignment]) -> AutoAscendDatasetBuilder:
        self._alignments = [str(alignment.value).title() for alignment in alignments]
        return self

    def game_ids(self, game_ids: List[int]) -> AutoAscendDatasetBuilder:
        self._game_ids = game_ids
        return self

    def game_versions(self, versions: List[str]) -> AutoAscendDatasetBuilder:
        self._game_versions = versions
        return self

    def build(
        self,
        batch_size: int,
        seq_len: int = 1,
        n_workers: int = 32,
        auto_ascend_cls=SARSAutoAscendTTYDataset,
        **kwargs,
    ) -> BaseAutoAscend:
        """
        Args:
            batch_size: well
            n_prefetch_states: how many states will be preloaded into the device memory (CPU for now)
        """
        # Build a sql query to select only filtered ones
        query, query_args = self._build_sql_query()

        tp = ThreadPoolExecutor(max_workers=n_workers)
        self._dataset = nld.TtyrecDataset(
            dataset_name="autoascend",
            dbfilename=self.db_path,
            batch_size=batch_size,
            seq_length=seq_len,
            shuffle=True,
            loop_forever=True,
            subselect_sql=query,
            subselect_sql_args=query_args,
            threadpool=tp,
        )
        print(f"Total games in the filtered dataset: {len(self._dataset._gameids)}")

        return auto_ascend_cls(
            self._dataset, batch_size=batch_size, threadpool=tp, **kwargs
        )

    def _build_sql_query(self) -> Tuple[str, Tuple]:
        subselect_sql = "SELECT gameid FROM games WHERE "

        # Game version (there can be potentially recordings from various NetHack versions)
        subselect_sql += "version in ({seq}) AND ".format(
            seq=",".join(["?"] * len(self._game_versions))
        )
        subselect_sql_args = tuple(self._game_versions)

        # If specific game ids were specified
        if self._game_ids is not None:
            subselect_sql += "gameid in ({seq}) AND ".format(
                seq=",".join(["?"] * len(self._game_ids))
            )
            subselect_sql_args += tuple(self._game_ids)
        if self._roles:
            subselect_sql += "role in ({seq}) AND ".format(
                seq=",".join(["?"] * len(self._roles))
            )
            subselect_sql_args += tuple(self._roles)
        if self._races:
            subselect_sql += "race in ({seq}) AND ".format(
                seq=",".join(["?"] * len(self._races))
            )
            subselect_sql_args += tuple(self._races)
        if self._alignments:
            # align can change during the game, so need to use 0
            subselect_sql += "align0 in ({seq}) AND ".format(
                seq=",".join(["?"] * len(self._alignments))
            )
            subselect_sql_args += tuple(self._alignments)
        if self._sex:
            # gender can change during the game, so need to use 0
            subselect_sql += "gender0 in ({seq}) AND ".format(
                seq=",".join(["?"] * len(self._sex))
            )
            subselect_sql_args += tuple(self._sex)

        # There will always be an AND at the end
        subselect_sql = subselect_sql[:-5]

        return subselect_sql, subselect_sql_args
