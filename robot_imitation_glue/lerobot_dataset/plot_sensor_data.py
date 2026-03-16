from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np


class SensorPlotter:
    def __init__(self, datasets):
        """
        datasets: list of (dataset_root, repo_id) tuples
        """
        if len(datasets) == 0 or len(datasets) > 3:
            raise ValueError("You must provide between 1 and 3 datasets.")

        self.datasets_info = datasets
        self.datasets = [
            LeRobotDataset(repo_id=repo_id, root=dataset_root)
            for dataset_root, repo_id in datasets
        ]

        self.episode_indices_list = [ds.episode_data_index for ds in self.datasets]
        for idx, episode_indices in enumerate(self.episode_indices_list):
            logger.debug(f"Dataset {idx}: episode_indices = {episode_indices}")

        self.observation_index = 0

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

        for (dataset_root, _), dataset, episode_indices, ax in zip(
            self.datasets_info, self.datasets, self.episode_indices_list, self.ax_list
        ):
            sensor_obs_list = []
            for start_idx in episode_indices["from"]:
                item = dict(dataset[int(start_idx) + self.observation_index])
                sensor_obs_list.append(item["clothes_hanger"])
            episodes_to_exclude_from_mean = [83, 156, 173, 174]
            sensor_obs_list_for_mean = [
                obs for idx, obs in enumerate(sensor_obs_list) if idx not in episodes_to_exclude_from_mean
            ]
            mean_sensor_obs = np.mean(sensor_obs_list_for_mean, axis=0)
            logger.info(f"{dataset_root} | mean_sensor_obs = {mean_sensor_obs}")

            ax.plot(sensor_obs_list)
            ax.set_title(dataset_root, fontsize=12)
            ax.set_ylabel("Value")

        self.ax_list[-1].set_xlabel("Episode Index")

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Provide up to three datasets here
    datasets = [
        ("datasets/clothes-hanger-v3p5-w500h720-2cam", "repo-id"),
        ("datasets/clothes-hanger-v3p5-2cam-EVAL", "repo-id"),
        ("datasets/clothes-hanger-v3p7-2cam-n50-EVAL", "yet-another-repo-id"),
    ]

    viewer = SensorPlotter(datasets)
    viewer.init_plot()
    viewer.show()
