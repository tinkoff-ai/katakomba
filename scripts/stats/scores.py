"""
This is a script for extracting the scores (used for populating values in utils/scores.py)
"""
import nle.dataset as nld

from nle.dataset.db import db as nld_database
from katakomba.utils.roles import Role, Race, Alignment

data_path = "data/nle_data"
db_path = "ttyrecs.db"

if not nld.db.exists(db_path):
    nld.db.create(db_path)
    nld.add_nledata_directory(data_path, "autoascend", db_path)

db_connection = nld.db.connect(filename=db_path)

with nld_database(conn=db_connection) as connection:
    c = connection.execute(
        "SELECT games.role, games.race, games.align0, AVG(games.points) "
        "FROM games "
        "JOIN datasets ON games.gameid=datasets.gameid "
        "WHERE datasets.dataset_name='autoascend' "
        "GROUP BY games.role, games.race, games.align0;",
    )

    for row in c:
        role, race, alignment, avg_score = row
        copypaste_string = f"({Role._value2member_map_[str.lower(role)]}, "
        copypaste_string += f"{Race._value2member_map_[str.lower(race)]}, "
        copypaste_string += f"{Alignment._value2member_map_[str.lower(alignment)]})"
        copypaste_string += f": {avg_score:.2f},"
        print(copypaste_string)