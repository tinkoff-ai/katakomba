import h5py


def combine_datasets(datasets_paths, new_path):
    with h5py.File(new_path, "a", track_order=True) as new_df:
        total_episodes = 0

        for dataset_path in datasets_paths:
            with h5py.File(dataset_path, "r") as df:
                for idx in range(len(df.keys())):
                    df.copy(f"episode_{idx}", new_df, name=f"episode_{total_episodes}")
                    total_episodes += 1