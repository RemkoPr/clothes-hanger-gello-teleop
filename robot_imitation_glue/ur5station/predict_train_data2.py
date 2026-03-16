from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from loguru import logger
import numpy as np
import torch
from eval_diffusion_lerobot import obs_preprocessor


class EpisodeViewer:
    def __init__(self, repo_id, checkpoint_path, raw_train_dataset_path, prepr_train_dataset_path, sophie_cam=True):
        self.sophie_cam = sophie_cam
        # Load dataset to inference
        self.prepr_dataset = LeRobotDataset(repo_id=repo_id, root=prepr_train_dataset_path)
        self.raw_dataset = LeRobotDataset(repo_id=repo_id, root=raw_train_dataset_path)
        self.checkpoint_path = checkpoint_path
        self.raw_train_dataset_path = raw_train_dataset_path
        self.prepr_train_dataset_path = prepr_train_dataset_path
        # for dataset root, take raw_train_dataset_path and remove everryithin in the string before "datasets"
        self.dataset_root = "datasets" + self.prepr_train_dataset_path.split("datasets")[1]
        print(f"Dataset root: {self.dataset_root}")
        self.include_instr = True if checkpoint_path.split("INSTR")[1][0] == "1" else False
        logger.info(f"Include_instr: {self.include_instr}")
        self.image_key_prefix = "observation.images."
        self.state_key_prefix = "observation."
        self.prepr_episode_indices = self.prepr_dataset.meta.episodes
        self.raw_episode_indices = self.raw_dataset.meta.episodes
        logger.debug(f"prepr_episode_indices = {self.prepr_episode_indices}")
        logger.debug(f"raw_episode_indices = {self.raw_episode_indices}")

        # Load independent policy pipelines.
        # Sharing a single policy object across both agents couples their internal temporal state.
        prepr_policy, prepr_preprocessor, prepr_postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
        raw_policy, raw_preprocessor, raw_postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
        self.prepr_lerobot_agent = LerobotAgent(
            prepr_policy,
            prepr_preprocessor,
            prepr_postprocessor,
            "cuda",
            lambda x: x,
        )
        self.raw_lerobot_agent = LerobotAgent(
            raw_policy,
            raw_preprocessor,
            raw_postprocessor,
            "cuda",
            self._wrapped_obs_preprocessor,
        )
 
        # State
        self.episode_idx = 0
        self.action_idx = 0
        self.n = 0
        self.episode_start_idx = None
        self.episode_to_idx = None
        self.true_actions = None
        self.pred1_actions = None
        self.pred2_actions = None
        self._raw_init_clothes_hanger = None  # home-pose reading used for relative ch normalisation

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
        self.line_pred1 = None
        self.vline = None
        self.slider_idx = None
        self.slider_act = None
        self.textbox_ep = None

    def _wrapped_obs_preprocessor(self, obs_dict):
        # Need to convert camera images from Tensors to opencv
        for key in obs_dict:
            if "image" in key:
                # images loaded from dataset have C,H,W torch format, the obs_preprocessor expects
                # the image to come directly from camera getter in opencv format H,W,C
                # we must first conver from torch to opencv, so that obs_preprocessor 
                # can go from opencv to torch
                obs_dict[key] = obs_dict[key].numpy().transpose(1, 2, 0)*255
        # prepare_datasets.py stores clothes_hanger as max(init_ch - current_ch, 0)
        # (the init frame is the home-pose frame, which is then dropped from the dataset).
        # Replicate that here so the state matches what the model was trained on.
        if self._raw_init_clothes_hanger is not None and "clothes_hanger" in obs_dict:
            init_ch = self._raw_init_clothes_hanger.numpy()
            curr_ch = obs_dict["clothes_hanger"].numpy() if hasattr(obs_dict["clothes_hanger"], "numpy") else obs_dict["clothes_hanger"]
            obs_dict["clothes_hanger"] = np.maximum(init_ch - curr_ch, 0).astype(np.float32)
        return obs_preprocessor(obs_dict, include_instr=self.include_instr)

    def load_episode(self, idx):
        """Load episode idx and compute true_actions & pred_actions arrays."""
        self.prepr_lerobot_agent.reset()
        self.raw_lerobot_agent.reset()
        self.episode_idx = idx
        self.episode_start_idx = self.prepr_episode_indices["dataset_from_index"][idx]
        self.episode_to_idx = self.prepr_episode_indices["dataset_to_index"][idx]
        raw_episode_start_idx = self.raw_episode_indices["dataset_from_index"][idx]
        raw_episode_to_idx = self.raw_episode_indices["dataset_to_index"][idx]

        prepr_len = int(self.episode_to_idx - self.episode_start_idx)
        raw_len = int(raw_episode_to_idx - raw_episode_start_idx)
        # In prepare_datasets.py we drop the first frame per episode (frames_to_drop=[0]).
        # Align raw episode frames with preprocessed frames by skipping the raw-only prefix.
        raw_prefix_skip = max(0, raw_len - prepr_len)

        # Store the home-pose clothes_hanger reading for relative normalisation
        # (mirrors prepare_datasets.py: clothes_hanger = max(init_ch - current_ch, 0))
        self._raw_init_clothes_hanger = self.raw_dataset[raw_episode_start_idx]["clothes_hanger"]

        true_actions_list = []
        pred1_actions_list = []
        pred2_actions_list = []
        inference_times = []
        
        # Inference first frame to get an observation step loaded in the model's memory
        for _ in range(self.prepr_lerobot_agent.policy.config.n_action_steps):
            item_prepr = dict(self.prepr_dataset[self.episode_start_idx]) 
            item_prepr.pop("action", None)
            item_prepr.pop("task", None)
            item_raw = dict(self.raw_dataset[raw_episode_start_idx + raw_prefix_skip])
            item_raw.pop("action", None)
            item_raw.pop("task", None)
            pred1 = self.prepr_lerobot_agent.get_action(item_prepr)
            pred2 = self.raw_lerobot_agent.get_action(item_raw)
        
        logger.warning(f"Loading episode {idx} from {self.episode_start_idx} to {self.episode_to_idx}")
        for i_rel, i in enumerate(range(self.episode_start_idx, self.episode_to_idx)):
            raw_i = raw_episode_start_idx + raw_prefix_skip + i_rel
            item_prepr = dict(self.prepr_dataset[i])  # copy to avoid changing the dataset
            true_actions_list.append(np.asarray(item_prepr["action"]))
            item_prepr.pop("action", None)  # technically not necessary since preprocessor removes it, but for clarity (if action in obesrvation, get_action will not inference)
            item_prepr.pop("task", None)  # remove task from observation for inference, since this is not provided at test time and we want to check if policy can still do something reasonable based on the other observations
            item_raw = dict(self.raw_dataset[raw_i])
            item_raw.pop("action", None)
            item_raw.pop("task", None)
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            t_start.record()
            pred1 = self.prepr_lerobot_agent.get_action(item_prepr)
            pred2 = self.raw_lerobot_agent.get_action(item_raw)
            t_end.record()
            torch.cuda.synchronize()
            inference_times.append(t_start.elapsed_time(t_end))
            pred1_actions_list.append(np.asarray(pred1))
            pred2_actions_list.append(np.asarray(pred2))
        print(f'Average inference time over episode {idx}: {np.mean(inference_times[5:]):.2f} ms +/- {np.std(inference_times[5:]):.2f} ms')
        self.true_actions = np.stack(true_actions_list)
        self.pred1_actions = np.stack(pred1_actions_list)
        self.pred2_actions = np.stack(pred2_actions_list)
        self.n = self.true_actions.shape[0]

        # Loss metrics
        mse_per_joint = np.mean((self.true_actions - self.pred1_actions) ** 2, axis=0)
        mae_per_joint = np.mean(np.abs(self.true_actions - self.pred1_actions), axis=0)
        mse_total = float(np.mean(mse_per_joint))
        mae_total = float(np.mean(mae_per_joint))
        logger.info(f"Episode {idx} loss  |  MSE: {mse_total:.6f}  MAE: {mae_total:.6f}")
        logger.info(f"  per-joint MSE: {np.array2string(mse_per_joint, precision=6, separator=', ')}")
        logger.info(f"  per-joint MAE: {np.array2string(mae_per_joint, precision=6, separator=', ')}")

    def init_plot(self):
        """Initialize figure and widgets."""
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle(f"Inference dataset: {self.dataset_root.split('/')[-1]}\
                          \nModel: {self.checkpoint_path.split('/')[-1]}\
                          \nTrain dataset: {self.prepr_train_dataset_path.split('/')[-1]}", fontsize=16)
        self.ax_scene = self.fig.add_subplot(2, 3, 1)
        self.ax_wilson = self.fig.add_subplot(2, 3, 2)
        self.ax_sophie = self.fig.add_subplot(2, 3, 3)
        self.ax_action = self.fig.add_subplot(2, 1, 2)

        # Show initial images
        obs0 = self.prepr_dataset[self.episode_start_idx]
        self.im_scene = self.ax_scene.imshow(obs0[self.image_key_prefix + "scene_image"].cpu().numpy().transpose(1, 2, 0))
        self.ax_scene.set_title("Scene Image")
        self.ax_scene.axis("off")

        self.im_wilson = self.ax_wilson.imshow(obs0[self.image_key_prefix + "wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
        self.ax_wilson.set_title("Wrist Wilson Image")
        self.ax_wilson.axis("off")

        if self.sophie_cam:
            self.im_sophie = self.ax_sophie.imshow(obs0[self.image_key_prefix + "wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
            self.ax_sophie.set_title("Wrist Sophie Image")
            self.ax_sophie.axis("off")

        # Action plot
        x = np.arange(self.n)
        self.line_real, = self.ax_action.plot(x, self.true_actions[:, self.action_idx], label="Action", linewidth=5)
        self.line_pred1, = self.ax_action.plot(x, self.pred1_actions[:, self.action_idx], label="Predicted action (prepr dataset)", linewidth=3)
        self.line_pred2, = self.ax_action.plot(x, self.pred2_actions[:, self.action_idx], label="Predicted action (raw dataset)", linewidth=3)
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
        obs = self.prepr_dataset[i_abs]
        self.im_scene.set_data(obs[self.image_key_prefix + "scene_image"].cpu().numpy().transpose(1, 2, 0))
        self.im_wilson.set_data(obs[self.image_key_prefix + "wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
        if self.sophie_cam:
            self.im_sophie.set_data(obs[self.image_key_prefix + "wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))
        self.vline.set_xdata([i_rel, i_rel])
        self.fig.canvas.draw_idle()

    def update_action_idx(self, val):
        """When Action-Idx slider moves, update y-data."""
        idx = int(self.slider_act.val)
        self.line_real.set_ydata(self.true_actions[:, idx])
        self.line_pred1.set_ydata(self.pred1_actions[:, idx])
        self.line_pred2.set_ydata(self.pred2_actions[:, idx])
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

        if not (0 <= new_ep < len(self.prepr_episode_indices["dataset_from_index"])):
            logger.warning("Episode index out of range.")
            return

        self.load_episode(new_ep)
        self.update_all_plots()

    def update_all_plots(self):
        """Refresh everything after loading a new episode."""
        obs0 = self.prepr_dataset[self.episode_start_idx]
        self.im_scene.set_data(obs0[self.image_key_prefix + "scene_image"].cpu().numpy().transpose(1, 2, 0))
        self.im_wilson.set_data(obs0[self.image_key_prefix + "wrist_wilson_image"].cpu().numpy().transpose(1, 2, 0))
        if self.sophie_cam:
            self.im_sophie.set_data(obs0[self.image_key_prefix + "wrist_sophie_image"].cpu().numpy().transpose(1, 2, 0))

        x = np.arange(self.n)
        curr_action_idx = max(0, min(6, int(self.slider_act.val)))
        self.line_real.set_xdata(x)
        self.line_pred1.set_xdata(x)
        self.line_real.set_ydata(self.true_actions[:, curr_action_idx])
        self.line_pred1.set_ydata(self.pred1_actions[:, curr_action_idx])
        self.line_pred2.set_ydata(self.pred2_actions[:, curr_action_idx])
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
    repo_id = "repo_id"
    root = "/storage/rproesma/clothes-hanger/"
    prepr_train_dataset_path = root + "datasets/b-n200-PREPR-INSTR1"
    raw_train_dataset_path = root + "datasets/b"
    model_checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/b-n200-INSTR1-300k-1enc"
    
    viewer = EpisodeViewer(repo_id, model_checkpoint_path, raw_train_dataset_path, prepr_train_dataset_path, sophie_cam=False)
    viewer.load_episode(5)
    viewer.init_plot()
    viewer.show()
