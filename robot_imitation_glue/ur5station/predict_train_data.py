from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from loguru import logger
import numpy as np
import torch



def preprocessor(obs):

    state = obs["observation.state"]
    state = torch.tensor(state).float().unsqueeze(0)
    scene_image = obs["observation.images.scene_image"]
    wrist_wilson_image = obs["observation.images.wrist_wilson_image"]
    wrist_sophie_image = obs["observation.images.wrist_sophie_image"]
    #scene_image = scene_image.permute(2, 0, 1).unsqueeze(0)
    #wrist_wilson_image = wrist_wilson_image.permute(2, 0, 1).unsqueeze(0)
    #wrist_sophie_image = wrist_sophie_image.permute(2, 0, 1).unsqueeze(0)
    scene_image = scene_image.unsqueeze(0)  # TODO: why no permute needed?
    wrist_wilson_image = wrist_wilson_image.unsqueeze(0)
    wrist_sophie_image = wrist_sophie_image.unsqueeze(0)
    return {
            "observation.images.scene_image": scene_image,
            "observation.images.wrist_wilson_image": wrist_wilson_image,
            "observation.images.wrist_sophie_image": wrist_sophie_image,
            "observation.state": state,
        }


class EpisodeViewer:
    def __init__(self, dataset_root, repo_id, checkpoint_path, train_dataset_path):
        # Load dataset to inference
        self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        self.episode_indices = self.dataset.episode_data_index
        logger.debug(f"episode_indices = {self.episode_indices}")

        # Load policy
        policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
        self.lerobot_agent = LerobotAgent(policy, "cuda", preprocessor)

        # State
        self.episode_idx = 0
        self.action_idx = 0
        self.n = 0
        self.episode_start_idx = None
        self.episode_to_idx = None
        self.true_actions = None
        self.pred_actions = None

        # Matplotlib handles
        self.fig = None
        self.ax_scene = None
        self.ax_wilson = None
        self.ax_sophie = None
        self.ax_action = None
        self.im_scene = None
        self.im_wilson = None
        self.im_sophie = None
        self.line_real = None
        self.line_pred = None
        self.vline = None
        self.slider_idx = None
        self.slider_act = None
        self.textbox_ep = None

    def load_episode(self, idx):
        """Load episode idx and compute true_actions & pred_actions arrays."""
        self.episode_idx = idx
        self.episode_start_idx = self.episode_indices["from"][idx].item()
        self.episode_to_idx = self.episode_indices["to"][idx].item()

        true_actions_list = []
        pred_actions_list = []

        logger.warning(f"Loading episode {idx} from {self.episode_start_idx} to {self.episode_to_idx}")
        for i in range(self.episode_start_idx, self.episode_to_idx):
            item = dict(self.dataset[i])  # copy to avoid changing the dataset
            item.pop("task", None)
            pred = self.lerobot_agent.get_action(item)
            pred_actions_list.append(np.asarray(pred))
            true_actions_list.append(np.asarray(item["action"]))
        self.true_actions = np.stack(true_actions_list)
        self.pred_actions = np.stack(pred_actions_list)
        self.n = self.true_actions.shape[0]

    def init_plot(self):
        """Initialize figure and widgets."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_scene = self.fig.add_subplot(2, 3, 1)
        self.ax_wilson = self.fig.add_subplot(2, 3, 2)
        self.ax_sophie = self.fig.add_subplot(2, 3, 3)
        self.ax_action = self.fig.add_subplot(2, 1, 2)

        # Show initial images
        obs0 = self.dataset[self.episode_start_idx]
        self.im_scene = self.ax_scene.imshow(obs0["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
        self.ax_scene.set_title("Scene Image")
        self.ax_scene.axis("off")

        self.im_wilson = self.ax_wilson.imshow(obs0["observation.images.wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
        self.ax_wilson.set_title("Wrist Wilson Image")
        self.ax_wilson.axis("off")

        self.im_sophie = self.ax_sophie.imshow(obs0["observation.images.wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
        self.ax_sophie.set_title("Wrist Sophie Image")
        self.ax_sophie.axis("off")

        # Action plot
        x = np.arange(self.n)
        self.line_real, = self.ax_action.plot(x, self.true_actions[:, self.action_idx], label="Action", linewidth=5)
        self.line_pred, = self.ax_action.plot(x, self.pred_actions[:, self.action_idx], label="Predicted action")
        self.vline = self.ax_action.axvline(0, color="r", linestyle="--", label="Current index (relative)")
        self.ax_action.set_xlabel("Relative index (0 .. n-1)")
        self.ax_action.set_ylabel("Action value")
        self.ax_action.legend()
        self.ax_action.set_xlim(0, max(1, self.n - 1))

        plt.subplots_adjust(bottom=0.35)

        # Sliders
        ax_slider_idx = plt.axes([0.2, 0.25, 0.6, 0.03])
        self.slider_idx = Slider(ax_slider_idx, 'Index', 0, max(0, self.n - 1), valinit=0, valstep=1)

        ax_slider_act = plt.axes([0.2, 0.20, 0.6, 0.03])
        self.slider_act = Slider(ax_slider_act, 'Action Idx', 0, 6, valinit=self.action_idx, valstep=1)

        # TextBox
        ax_text_ep = plt.axes([0.15, 0.05, 0.12, 0.05])
        self.textbox_ep = TextBox(ax_text_ep, "Episode idx", initial=str(self.episode_idx))

        # Wire callbacks
        self.slider_idx.on_changed(self.update_frame)
        self.slider_act.on_changed(self.update_action_idx)
        self.textbox_ep.on_submit(self.set_episode)

    def update_frame(self, val):
        """Update images and vertical line based on slider."""
        i_rel = int(self.slider_idx.val)
        i_abs = self.episode_start_idx + i_rel
        obs = self.dataset[i_abs]
        self.im_scene.set_data(obs["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
        self.im_wilson.set_data(obs["observation.images.wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
        self.im_sophie.set_data(obs["observation.images.wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
        self.vline.set_xdata([i_rel, i_rel])
        self.fig.canvas.draw_idle()

    def update_action_idx(self, val):
        """When Action-Idx slider moves, update y-data."""
        idx = int(self.slider_act.val)
        self.line_real.set_ydata(self.true_actions[:, idx])
        self.line_pred.set_ydata(self.pred_actions[:, idx])
        self.ax_action.relim()
        self.ax_action.autoscale_view()
        self.fig.canvas.draw_idle()

    def set_episode(self, text):
        """Handler for episode TextBox."""
        try:
            new_ep = int(text)
        except ValueError:
            logger.warning("Invalid episode index entered (not an int).")
            return

        if not (0 <= new_ep < len(self.episode_indices["from"])):
            logger.warning("Episode index out of range.")
            return

        self.load_episode(new_ep)
        self.update_all_plots()

    def update_all_plots(self):
        """Refresh everything after loading a new episode."""
        obs0 = self.dataset[self.episode_start_idx]
        self.im_scene.set_data(obs0["observation.images.scene_image"].cpu().numpy().transpose(1, 2, 0))
        self.im_wilson.set_data(obs0["observation.images.wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
        self.im_sophie.set_data(obs0["observation.images.wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))

        x = np.arange(self.n)
        curr_action_idx = max(0, min(6, int(self.slider_act.val)))
        self.line_real.set_xdata(x)
        self.line_pred.set_xdata(x)
        self.line_real.set_ydata(self.true_actions[:, curr_action_idx])
        self.line_pred.set_ydata(self.pred_actions[:, curr_action_idx])

        self.ax_action.set_xlim(0, max(1, self.n - 1))
        self.ax_action.relim()
        self.ax_action.autoscale_view()

        # Update slider range
        self.slider_idx.valmax = max(0, self.n - 1)
        self.slider_idx.ax.clear()
        self.slider_idx = Slider(self.slider_idx.ax, 'Index', 0, max(0, self.n - 1), valinit=0, valstep=1)
        self.slider_idx.on_changed(self.update_frame)

        self.vline.set_xdata([0, 0])
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    # Dataset to inference
    root_dir = "datasets/clothes-hanger-v3p3-w426h480"
    #root_dir = "datasets/clothes-hanger-v3-test-w426h480"
    repo_id = "clothes-hanger-repo-v3-test"
    # Model
    checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-08/18-23-04_clothes-hanger-v3.3/checkpoints/100000/pretrained_model"
    train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p3-w426h480"

    viewer = EpisodeViewer(root_dir, repo_id, checkpoint_path, train_dataset_path)
    viewer.load_episode(89)
    viewer.init_plot()
    viewer.show()
