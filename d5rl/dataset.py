"""
Memes
 - Screen may not contain the map on the screen (there can be just a menu, or inventory)
 - Default TTYRec dataset fetches datapoints sequentially (i.e. each sample goes one after another within a game)

What to keep in mind:
 - Alignment between dataset actions and environment actions
 - Match terminal sizes (seems that the dataset uses 80x24, but original NLE 79x21)
"""
import nle.dataset as nld
import nle
import logging
import render_utils

# Filters
# - Same as for environment
# - Game version (to make sure we match 3.6.6)

class AutoAscendNetHackDataset:
    """
    This is the most basic wrapper. 
    It obeys the original logic of the TTYRec dataset that samples data within-game-sequentially.
    """
    def __init__(
        self, 
        batch_size: int,
        seq_len   : int = 1,
        path      : str = "../data/nle_data",
        db_path   : str = "ttyrecs.db"
    ):
        # Create a sql-lite database for keeping trajectories
        if not nld.db.exists(db_path):
            nld.db.create(db_path)
            nld.add_nledata_directory(path, "autoascend", db_path)

        # Create a connection to specify the database to use
        db_conn = nld.db.connect(filename=db_path)
        logging.info(f"AutoAscend Dataset has {nld.db.count_games('autoascend', conn=db_conn)} games.")

        self._dataset = nld.TtyrecDataset(
            dataset_name = "autoascend",
            batch_size   = batch_size,
            seq_length   = seq_len,
            shuffle      = True,         # Note that this shuffles gameids only (not frames within games as you usually expect)
            loop_forever = True
        )
        self._iterator = iter(self._dataset)

    def sample(self):
        """
        Returns a usual (s, a, s', r, done), where
        - s is a tty-screen [batch_size, 80, 24, ?] (uint8)
        - a is an action [batch_size, 1] (uint8)
        - s' is a tty-screen [batch_size, 80, 24, ?] (uint8)
        - r is the change in the game score
        - whether the episode ended (game-over usually) (bool)
        """
        # For your mental health, here are the keys
        # dict_keys(['tty_chars', 'tty_colors', 'tty_cursor', 'timestamps', 'done', 'gameids', 'keypresses', 'scores'])
        batch = next(self._iterator)

        # Transform to a picture
        image = render_utils.render_crop(
            batch["tty_chars"][0],
            batch["tty_colors"][0],
            batch["tty_cursor"].astype(np.uint8)[0],
            self.char_array,
            screen_image,
            self.crop_dim
        )
        # All the normalization should happen further

def load_dataset(path: str = "./data/nle_data", db_path: str = "ttyrecs.db") -> nld.TtyrecDataset:
    # 
    if not nld.db.exists(db_path):
        nld.db.create(db_path)
        nld.add_nledata_directory(path, "taster-dataset", db_path)

    # Create a connection to specify the database to use
    db_conn = nld.db.connect(filename=db_path)

    # Then you can inspect the number of games in each dataset:
    print(f"NLD AA \"Taster\" Dataset has {nld.db.count_games('taster-dataset', conn=db_conn)} games.")

    dataset = nld.TtyrecDataset(
        "taster-dataset",
        batch_size=32,
        seq_length=32,
        dbfilename=dbfilename,
    )

minibatch = next(iter(dataset))
print(minibatch.keys())

from nle.nethack import tty_render

batch_idx = 0
time_idx = 0
chars = minibatch['tty_chars'][batch_idx, time_idx]
colors = minibatch['tty_colors'][batch_idx, time_idx]
cursor = minibatch['tty_cursor'][batch_idx, time_idx]

print(tty_render(chars, colors, cursor))