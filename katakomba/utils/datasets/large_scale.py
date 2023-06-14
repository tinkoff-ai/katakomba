import shutil
import nle.dataset as nld

from katakomba.utils.roles import Alignment, Race, Role, Sex
from typing import Tuple, Sequence, Optional
from concurrent.futures import ThreadPoolExecutor


def load_nld_aa_large_dataset(
        data_path: str,
        db_path: str,
        seq_len: int,
        batch_size: int,
        num_workers: int = 8,
        role: Optional[Role] = None,
        race: Optional[Race] = None,
        align: Optional[Alignment] = None,
        **kwargs
) -> nld.TtyrecDataset:
    if nld.db.exists(db_path):
        # if the db was not properly initialized previously for some reason
        # (i.e., a wrong path and then fixed) we need to delete it and recreate from scratch
        shutil.rmtree(db_path)

    nld.db.create(db_path)
    nld.add_nledata_directory(data_path, "autoascend", db_path)

    # how to write it more clearly?
    query, query_args = build_dataset_sql_query(
        roles=[str(role.value).title()] if role is not None else None,
        races=[str(race.value).title()] if race is not None else None,
        alignments=[str(align.value).title()] if align is not None else None,
        **kwargs
    )
    tp = ThreadPoolExecutor(max_workers=num_workers)

    dataset = nld.TtyrecDataset(
        dataset_name="autoascend",
        dbfilename=db_path,
        batch_size=batch_size,
        seq_length=seq_len,
        shuffle=True,
        loop_forever=True,
        subselect_sql=query,
        subselect_sql_args=query_args,
        threadpool=tp,
    )
    print(f"Total games in the filtered dataset: {len(dataset._gameids)}")

    return dataset


def build_dataset_sql_query(
        roles: Optional[Sequence[str]] = None,
        races: Optional[Sequence[str]] = None,
        alignments: Optional[Sequence[str]] = None,
        genders: Optional[Sequence[str]] = None,
        game_versions: Optional[Sequence[str]] = ("3.6.6",),
        game_ids: Optional[Tuple[int]] = None
) -> Tuple[str, Tuple]:
    subselect_sql = "SELECT gameid FROM games WHERE "

    # Game version (there can be potentially recordings from various NetHack versions)
    subselect_sql += "version in ({seq}) AND ".format(
        seq=",".join(["?"] * len(game_versions))
    )
    subselect_sql_args = tuple(game_versions)

    # If specific game ids were specified
    if game_ids is not None:
        subselect_sql += "gameid in ({seq}) AND ".format(
            seq=",".join(["?"] * len(game_ids))
        )
        subselect_sql_args += tuple(game_ids)
    if roles is not None:
        subselect_sql += "role in ({seq}) AND ".format(
            seq=",".join(["?"] * len(roles))
        )
        subselect_sql_args += tuple(roles)
    if races is not None:
        subselect_sql += "race in ({seq}) AND ".format(
            seq=",".join(["?"] * len(races))
        )
        subselect_sql_args += tuple(races)
    if alignments is not None:
        subselect_sql += "align in ({seq}) AND ".format(
            seq=",".join(["?"] * len(alignments))
        )
        subselect_sql_args += tuple(alignments)
    if genders is not None:
        subselect_sql += "gender in ({seq}) AND ".format(
            seq=",".join(["?"] * len(genders))
        )
        subselect_sql_args += tuple(genders)

    # There will always be an AND at the end
    subselect_sql = subselect_sql[:-5]

    return subselect_sql, subselect_sql_args