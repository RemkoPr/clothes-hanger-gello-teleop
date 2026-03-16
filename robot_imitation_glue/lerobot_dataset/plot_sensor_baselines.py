import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset = LeRobotDataset(repo_id="whatevs", root="datasets/b")

values = []  # list of (N,) arrays, one per episode
for ep_idx in range(dataset.num_episodes):
    first_frame_idx = dataset.meta.episodes[ep_idx]["dataset_from_index"]
    frame = dataset[first_frame_idx]
    values.append(np.array(frame["clothes_hanger"]).flatten())

values = np.array(values)  # (num_episodes, N)
n_channels = values.shape[1]
episode_indices = np.arange(len(values))

fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)
if n_channels == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.plot(episode_indices, values[:, i])
    ax.set_ylabel(f"clothes_hanger[{i}]")
    ax.grid(True)

axes[-1].set_xlabel("Episode Index")
fig.suptitle("clothes_hanger — first frame of each episode")
plt.tight_layout()
plt.show()