"""
This is a script for extracting the scores (used for populating values in utils/scores.py)
"""
import nle.dataset as nld

from nle.dataset.db import db as nld_database
from katakomba.utils.roles import ALLOWED_COMBOS, Role, Race, Alignment, Sex


db_path = "ttyrecs.db"
db_connection = nld.db.connect(filename=db_path)

with nld_database(conn=db_connection) as connection:
    c = connection.execute(
        "SELECT games.role, games.race, games.align0, games.gender0, AVG(games.points) "
        "FROM games "
        "JOIN datasets ON games.gameid=datasets.gameid "
        "WHERE datasets.dataset_name='autoascend' "
        "GROUP BY games.role, games.race, games.gender0, games.align0;",
    )

    for row in c:
        role, race, alignment, gender, avg_score = row
        copypaste_string = f"({Role._value2member_map_[str.lower(role)]}, "
        copypaste_string += f"{Race._value2member_map_[str.lower(race)]}, "
        copypaste_string += f"{Alignment._value2member_map_[str.lower(alignment)]}, "
        copypaste_string += f"{Sex._value2member_map_[str.lower(gender)]})"
        copypaste_string += f": {avg_score:.2f},"
        print(copypaste_string)
