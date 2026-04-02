from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np


class SensorPlotter:
    def __init__(self, datasets, episode_idx):
        """
        datasets: list of (dataset_root, repo_id) tuples
        episode_idx: which episode index to plot (0-based within each dataset)
        """
        if len(datasets) == 0 or len(datasets) > 3:
            raise ValueError("You must provide between 1 and 3 datasets.")

    
        self.datasets_info = datasets
        self.datasets = [
            LeRobotDataset(repo_id=repo_id, root=dataset_root)
            for dataset_root, repo_id in datasets
        ]
        self.episode_idx = episode_idx


        episodes_meta = self.datasets[0].meta.episodes  # TODO fix for multiple datasets
        self.episode_indices_list = [range(episodes_meta["dataset_from_index"][idx], episodes_meta["dataset_to_index"][idx]) for idx in range(len(episodes_meta["dataset_from_index"]))]
        for idx, episode_indices in enumerate(self.episode_indices_list):
            logger.debug(f"Dataset {idx}: episode_indices = {episode_indices}")

        # Matplotlib handles
        self.fig = None
        self.ax_list = []

    def init_plot(self):
        """Initialize figure and plot sensor values."""
        n_datasets = len(self.datasets)
        self.fig, self.ax_list = plt.subplots(n_datasets, 1, figsize=(12, 4 * n_datasets), sharex=True)

        # Ensure ax_list is iterable even when n_datasets = 1
        if n_datasets == 1:
            self.ax_list = [self.ax_list]

        episode_indices = self.episode_indices_list[self.episode_idx]  # TODO fix for multiple datasets
        print(episode_indices)
        sensor_obs_list = []
        for frame_idx in episode_indices:
            item = dict(self.datasets[0][frame_idx])
            #sensor_obs_list.append(item["observation.state"][7:])
            sensor_obs_list.append(item["clothes_hanger"])

        self.ax_list[0].plot(sensor_obs_list)
        self.ax_list[0].set_title(self.datasets_info[0][0], fontsize=12)
        self.ax_list[0].set_ylabel("Value")

        self.ax_list[-1].set_xlabel("Episode Index")

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Provide up to three datasets here
    datasets = [
        ("datasets/b-n150-INSTR0-250k-EVAL", "repo-id"),
        #("datasets/clothes-hanger-v3p5-2cam-EVAL", "repo-id"),
        #("datasets/clothes-hanger-v3p7-2cam-n50-EVAL", "yet-another-repo-id"),
    ]

    viewer = SensorPlotter(datasets, episode_idx=1)
    viewer.init_plot()
    viewer.show()
