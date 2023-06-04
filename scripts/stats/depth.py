"""
This is a script for extracting the scores (used for populating values in utils/scores.py)
"""
import nle.dataset as nld
import numpy as np

from nle.dataset.db import db as nld_database

data_path = "data/nle_data"
db_path = "ttyrecs.db"

if not nld.db.exists(db_path):
    nld.db.create(db_path)
    nld.add_nledata_directory(data_path, "autoascend", db_path)

db_connection = nld.db.connect(filename=db_path)

with nld_database(conn=db_connection) as connection:
    c = connection.execute(
        "SELECT games.role, games.race, games.align0, games.deathlev "
        "FROM games "
        "JOIN datasets ON games.gameid=datasets.gameid "
        "WHERE datasets.dataset_name='autoascend' "
    )

all_levels = {}
global_levels = []
for row in c:
    role, race, alignment, deathlev = row
    key = (role, race, alignment)
    if key not in all_levels:
        all_levels[key] = [deathlev]
    else:
        all_levels[key].append(deathlev)
    global_levels.append(deathlev)

print("All dataset: ", np.median(global_levels), np.mean(global_levels))
for key in all_levels:
    print(key, np.median(all_levels[key]), np.mean(all_levels[key]))
