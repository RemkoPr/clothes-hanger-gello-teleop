# made changes to /home/rproesma/miniconda3/envs/robilglue/lib/python3.10/json/decoder.py

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from loguru import logger
import numpy as np
import torch


ACTION_IDX = 0


def preprocessor(obs_dict):
    # In this script, we load the preprocessed dataset, so no additional preprocessing is needed.
    return obs_dict


# Load train data
root_dir="datasets/clothes-hanger-v3p3-w426h480"
#root_dir="datasets/clothes-hanger-v3-raw"
repo_id="tmp"#"clothes-hanger-repo-v3p3",
dataset = LeRobotDataset(repo_id=repo_id, root=root_dir)

episode_indices = dataset.episode_data_index
episode_idx = 0
episode_start_idx = episode_indices["from"][episode_idx].item()
episode_to_idx = episode_indices["to"][episode_idx].item()
logger.debug(f"episode_indices = {episode_indices}")
logger.debug(f"episode_idx = {episode_idx}")
logger.debug(f"episode_start_idx = {episode_start_idx}")
logger.debug(f"episode_to_idx = {episode_to_idx}")
logger.debug(f'dataset keys: {list(dataset[episode_start_idx].keys())}')

# Load policy
checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-08/18-23-04_clothes-hanger-v3.3/checkpoints/100000/pretrained_model"
train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p3-w426h480"  # preprocessed train dataset    
policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
lerobot_agent = LerobotAgent(policy, "cuda", preprocessor)

# Predict actions from observations
predicted_actions = []
for i in range(episode_start_idx, episode_to_idx):
    obs = dataset[i]
    # drop "task" key
    obs.pop("task", None)
    action = lerobot_agent.get_action(obs)  # preprocessing happens in the agent
    predicted_actions.append(action[ACTION_IDX])

# Extract all actions for the action plot
actions = [dataset[idx]["action"][ACTION_IDX] for idx in range(episode_start_idx, episode_to_idx)]
n = len(actions)
assert(len(actions) == len(predicted_actions))


# --- Create the figure layout ---
fig = plt.figure(figsize=(12, 8))

# Image subplots
ax_scene = fig.add_subplot(2, 3, 1)
ax_wilson = fig.add_subplot(2, 3, 2)
ax_sophie = fig.add_subplot(2, 3, 3)

# Action plot
ax_action = fig.add_subplot(2, 1, 2)

# Display initial images
im_scene = ax_scene.imshow(dataset[episode_start_idx]["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
ax_scene.set_title("Scene Image")
ax_scene.axis("off")

im_wilson = ax_wilson.imshow(dataset[episode_start_idx]["observation.images.wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
ax_wilson.set_title("Wrist Wilson Image")
ax_wilson.axis("off")

im_sophie = ax_sophie.imshow(dataset[episode_start_idx]["observation.images.wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
ax_sophie.set_title("Wrist Sophie Image")
ax_sophie.axis("off")

# Display action plot
ax_action.plot(range(n), actions, label="Action")
ax_action.plot(range(n), predicted_actions, label="Predicted action")
vline = ax_action.axvline(episode_start_idx, color="r", linestyle="--", label="Current index")
ax_action.set_xlabel("Index i")
ax_action.set_ylabel("Action value")
ax_action.legend()

# Adjust layout for slider
plt.subplots_adjust(bottom=0.15)

# Slider axis: [left, bottom, width, height]
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Index', episode_start_idx, episode_to_idx, valinit=episode_start_idx, valstep=1)

# --- Update function ---
def update(val):
    i = int(slider.val)
    im_scene.set_data(dataset[i]["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
    im_wilson.set_data(dataset[i]["observation.images.wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
    im_sophie.set_data(dataset[i]["observation.images.wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
    vline.set_xdata([i, i])
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
