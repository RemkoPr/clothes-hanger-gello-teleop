# made changes to /home/rproesma/miniconda3/envs/robilglue/lib/python3.10/json/decoder.py

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from loguru import logger
import numpy as np
import torch
import cv2


ACTION_IDX = 0


def preprocessor(obs_dict):
    crop_width_range = (320+0, 960-0)
    crop_width = crop_width_range[1] - crop_width_range[0]
    crop_height_range = (0, 720)
    crop_height = crop_height_range[1] - crop_height_range[0]
    crop_width = crop_width_range[1] - crop_width_range[0]
    resize = (2*crop_width//3, 2*crop_height//3)
    train_crop_cutoff = (0, 0)  # 10 left and right, 0 top and bottom
    scene_img = obs_dict["scene_image"]
    print("=======================")
    print(scene_img)
    wrist_wilson_img = obs_dict["wrist_wilson_image"]
    wrist_sophie_img = obs_dict["wrist_sophie_image"]
    state = obs_dict["state"]

    current_joints = obs_dict["joints"]
    current_gripper = np.array([obs_dict["gripper_states"][1]])
    clothes_hanger = obs_dict["clothes_hanger"]
    state = np.concatenate([current_joints, current_gripper, clothes_hanger]).astype(np.float32)
    state = torch.tensor(state).float().unsqueeze(0)

    scene_image = torch.tensor(cv2.resize(np.array(scene_img)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_LINEAR)).float() / 255.0
    wrist_wilson_image = cv2.resize(np.array(wrist_wilson_img)[:, crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR)
    wrist_wilson_image = np.array(wrist_wilson_image)[train_crop_cutoff[0]:-train_crop_cutoff[0] if train_crop_cutoff[0] > 0 else None, train_crop_cutoff[1]:-train_crop_cutoff[1] if train_crop_cutoff[1] > 0 else None]
    wrist_wilson_image = torch.tensor(wrist_wilson_image).float() / 255.0
    wrist_sophie_image = cv2.resize(np.array(wrist_sophie_img)[:, crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR)
    wrist_sophie_image = np.array(wrist_sophie_image)[train_crop_cutoff[0]:-train_crop_cutoff[0] if train_crop_cutoff[0] > 0 else None, train_crop_cutoff[1]:-train_crop_cutoff[1] if train_crop_cutoff[1] > 0 else None]
    wrist_sophie_image = torch.tensor(wrist_sophie_image).float() / 255.0
    scene_image = scene_image.permute(2, 0, 1).unsqueeze(0)
    wrist_wilson_image = wrist_wilson_image.permute(2, 0, 1).unsqueeze(0)
    wrist_sophie_image = wrist_sophie_image.permute(2, 0, 1).unsqueeze(0)

    return {
        "observation.images.scene_image": scene_image,
        "observation.images.wrist_wilson_image": wrist_wilson_image,
        "observation.images.wrist_sophie_image": wrist_sophie_image,
        "observation.state": state,
    }

# Load train data
root_dir="datasets/clothes-hanger-v3-raw"
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
    preprocessed_obs = preprocessor(obs)
    action = lerobot_agent.get_action(preprocessed_obs)
    predicted_actions.append(action[ACTION_IDX])

# Extract all actions for the action plot
actions = [dataset[idx]["action"][ACTION_IDX] for idx in range(episode_start_idx, episode_to_idx)]
n = len(actions)

# --- Create the figure layout ---
fig = plt.figure(figsize=(12, 8))

# Image subplots
ax_scene = fig.add_subplot(2, 3, 1)
ax_wilson = fig.add_subplot(2, 3, 2)
ax_sophie = fig.add_subplot(2, 3, 3)

# Action plot
ax_action = fig.add_subplot(2, 1, 2)

# Display initial images
im_scene = ax_scene.imshow(dataset[episode_start_idx]["scene_image"].cpu().numpy().transpose(1, 2, 0))
ax_scene.set_title("Scene Image")
ax_scene.axis("off")

im_wilson = ax_wilson.imshow(dataset[episode_start_idx]["wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
ax_wilson.set_title("Wrist Wilson Image")
ax_wilson.axis("off")

im_sophie = ax_sophie.imshow(dataset[episode_start_idx]["wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
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
    im_scene.set_data(dataset[i]["scene_image"].cpu().numpy().transpose(1, 2, 0))
    im_wilson.set_data(dataset[i]["wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
    im_sophie.set_data(dataset[i]["wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
    vline.set_xdata([i, i])
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
