"""
This is a script for extracting the scores (used for populating values in utils/scores.py)
"""
import nle.dataset as nld
import numpy as np

from nle.dataset.db import db as nld_database
from katakomba.utils.roles import ALLOWED_COMBOS, Role, Race, Alignment, Sex


db_path = "ttyrecs.db"
db_connection = nld.db.connect(filename=db_path)

with nld_database(conn=db_connection) as connection:
    c = connection.execute(
        "SELECT games.role, games.race, games.align0, games.gender0, games.maxlvl "
        "FROM games "
        "JOIN datasets ON games.gameid=datasets.gameid "
        "WHERE datasets.dataset_name='autoascend' "
    )

    all_levels = {}
    for row in c:
        role, race, alignment, gender, maxlvl = row
        key = (role, race, alignment, gender)
        if key not in all_levels:
            all_levels[key] = [maxlvl]
        else:
            all_levels[key].append(maxlvl)

for key in all_levels:
    print(key, np.median(all_levels[key]), np.mean(all_levels[key]))
