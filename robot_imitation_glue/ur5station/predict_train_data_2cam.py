from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from loguru import logger
import numpy as np
import torch

# ------------------------
# Load dataset & policy
# ------------------------
root_dir = "datasets/clothes-hanger-prepr-cropped-w500"
repo_id = "tmp"
dataset = LeRobotDataset(repo_id=repo_id, root=root_dir)

episode_indices = dataset.episode_data_index
logger.debug(f"episode_indices = {episode_indices}")

checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-06-17/18-16-06_clothes-hanger-v1/checkpoints/100000/pretrained_model"
train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-prepr-cropped-w500"
policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
lerobot_agent = LerobotAgent(policy, "cuda", lambda obs: obs)

# ------------------------
# Globals (will be changed by UI)
# ------------------------
episode_idx = 0   # which episode to show (editable via TextBox)
ACTION_IDX = 0    # which action dim to plot (0..7)
slider_idx = None # will hold the index Slider object
ax_slider_idx = None

# ------------------------
# Load one episode into memory (true_actions/pred_actions arrays)
# ------------------------
def load_episode(idx):
    """Loads episode idx and computes true_actions & pred_actions arrays for that episode."""
    global episode_idx, episode_start_idx, episode_to_idx
    global true_actions, pred_actions, n

    episode_idx = idx
    episode_start_idx = episode_indices["from"][episode_idx].item()
    episode_to_idx = episode_indices["to"][episode_idx].item()

    true_actions_list = []
    pred_actions_list = []

    logger.warning(f"Loading episode {episode_idx} from {episode_start_idx} to {episode_to_idx}")
    for i in range(episode_start_idx, episode_to_idx):
        item = dataset[i]
        # copy or shallow-cast to dict to avoid changing the dataset
        obs = dict(item)
        obs.pop("task", None)
        pred = lerobot_agent.get_action(obs)  # expected shape (8,)
        pred_actions_list.append(np.asarray(pred))            # ensure numpy
        true_actions_list.append(np.asarray(item["action"]))  # ensure numpy

    true_actions = np.stack(true_actions_list)   # shape (n, 8)
    pred_actions = np.stack(pred_actions_list)   # shape (n, 8)
    n = true_actions.shape[0]

# initial load
load_episode(episode_idx)

# ------------------------
# Build figure & initial plots
# ------------------------
fig = plt.figure(figsize=(12, 8))

ax_scene = fig.add_subplot(2, 3, 1)
ax_wilson = fig.add_subplot(2, 3, 2)
ax_sophie = fig.add_subplot(2, 3, 3)
ax_action = fig.add_subplot(2, 1, 2)

# show first frame images
im_scene = ax_scene.imshow(dataset[episode_start_idx]["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
ax_scene.set_title("Scene Image")
ax_scene.axis("off")

im_sophie = ax_sophie.imshow(dataset[episode_start_idx]["observation.images.wrist_image"].cpu().numpy().transpose(1, 2, 0))
ax_sophie.set_title("Wrist (Sophie) Image")
ax_sophie.axis("off")

# initial action plot: x is relative index 0..n-1
x = np.arange(n)
line_real, = ax_action.plot(x, true_actions[:, ACTION_IDX], label="Action")
line_pred, = ax_action.plot(x, pred_actions[:, ACTION_IDX], label="Predicted action")
vline = ax_action.axvline(0, color="r", linestyle="--", label="Current index (relative)")
ax_action.set_xlabel("Relative index (0 .. n-1)")
ax_action.set_ylabel("Action value")
ax_action.legend()
ax_action.set_xlim(0, max(1, n - 1))

# ------------------------
# Sliders + TextBox layout
# ------------------------
plt.subplots_adjust(bottom=0.35)

ax_slider_idx = plt.axes([0.2, 0.25, 0.6, 0.03])
slider_idx = Slider(ax_slider_idx, 'Index', 0, max(0, n - 1), valinit=0, valstep=1)

ax_slider_act = plt.axes([0.2, 0.20, 0.6, 0.03])
slider_act = Slider(ax_slider_act, 'Action Idx', 0, 6, valinit=ACTION_IDX, valstep=1)

ax_text_ep = plt.axes([0.15, 0.05, 0.12, 0.05])
textbox_ep = TextBox(ax_text_ep, "Episode idx", initial=str(episode_idx))

# ------------------------
# Update callbacks
# ------------------------
def update_frame(val):
    """Update images and vertical line according to the relative index slider."""
    i_rel = int(slider_idx.val)           # relative index into current episode (0..n-1)
    i_abs = episode_start_idx + i_rel     # absolute dataset index
    # update images
    im_scene.set_data(dataset[i_abs]["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
    im_sophie.set_data(dataset[i_abs]["observation.images.wrist_image"].cpu().numpy().transpose(1, 2, 0))
    # vertical line should be on the same relative x-axis as the action plot
    vline.set_xdata([i_rel, i_rel])
    fig.canvas.draw_idle()

def update_action_idx(val):
    """When Action-Idx slider moves, slice the precomputed arrays and update y-data.
       Note: x-data must already match the current episode's length (we ensure that when episodes change)."""
    idx = int(slider_act.val)
    # update ydata (xdata remains 0..n-1)
    line_real.set_ydata(true_actions[:, idx])
    line_pred.set_ydata(pred_actions[:, idx])
    # recompute axis limits for y (x limits remain 0..n-1)
    ax_action.relim()
    ax_action.autoscale_view()
    fig.canvas.draw_idle()

def set_episode(text):
    """Handler for episode TextBox. Re-load episode and update all plot/widget state safely."""
    global slider_idx, ax_slider_idx, episode_idx, n

    try:
        new_ep = int(text)
    except ValueError:
        logger.warning("Invalid episode index entered (not an int).")
        return

    if not (0 <= new_ep < len(episode_indices["from"])):
        logger.warning("Episode index out of range.")
        return

    # Load new episode (updates true_actions, pred_actions, episode_start_idx, episode_to_idx, n)
    load_episode(new_ep)

    # Update images to first frame of the new episode
    im_scene.set_data(dataset[episode_start_idx]["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
    im_sophie.set_data(dataset[episode_start_idx]["observation.images.wrist_image"].cpu().numpy().transpose(1, 2, 0))

    # Rebuild x-data for action lines to match new n
    x = np.arange(n)
    # choose current action-dim (keep slider_act value if possible, else 0)
    curr_action_idx = int(slider_act.val) if hasattr(slider_act, "val") else 0
    curr_action_idx = max(0, min(6, curr_action_idx))

    line_real.set_xdata(x)
    line_pred.set_xdata(x)
    # set new ydata sliced from newly loaded arrays
    line_real.set_ydata(true_actions[:, curr_action_idx])
    line_pred.set_ydata(pred_actions[:, curr_action_idx])

    # Update action-axis limits to match new x-range and recompute y-limits
    ax_action.set_xlim(0, max(1, n - 1))
    ax_action.relim()
    ax_action.autoscale_view()

    # Reset and recreate the Index slider so its range matches the new n
    # Clear the old axis and make a new Slider widget on the same axes
    ax_slider_idx.clear()
    # Re-create the Slider (and reconnect the callback)
    slider_idx = Slider(ax_slider_idx, 'Index', 0, max(0, n - 1), valinit=0, valstep=1)
    slider_idx.on_changed(update_frame)

    # Reset vline to start of episode
    vline.set_xdata([0, 0])

    # redraw
    fig.canvas.draw_idle()

# wire callbacks
slider_idx.on_changed(update_frame)
slider_act.on_changed(update_action_idx)
textbox_ep.on_submit(set_episode)

plt.show()
