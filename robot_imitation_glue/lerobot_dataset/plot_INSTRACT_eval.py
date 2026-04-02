from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numpy as np


def _to_numpy(x):
	if hasattr(x, "cpu"):
		return x.cpu().numpy()
	if hasattr(x, "numpy"):
		return x.numpy()
	return np.asarray(x)


class EpisodeViewer:
	def __init__(self, dataset_root, repo_id):
		self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
		self.dataset_root = dataset_root
		self.episode_indices = self.dataset.meta.episodes

		self.episode_idx = 0
		self.action_idx = 7
		self.n = 0
		self.episode_start_idx = None
		self.episode_to_idx = None

		self.actions = None
		self.clothes_hanger_obs = None
		self.max_action_idx = 0

		self.fig = None
		self.ax_scene = None
		self.ax_wilson = None
		self.ax_sophie = None
		self.ax_action = None
		self.im_scene = None
		self.im_wilson = None
		self.im_sophie = None
		self.line_action = None
		self.line_ref = None
		self.vline = None
		self.slider_idx = None
		self.slider_act = None
		self.textbox_ep = None

		# EVAL datasets can be 2-cam or 3-cam.
		sample0 = self.dataset[0]
		self.sophie_cam = "wrist_sophie_image" in sample0

	def _get_clothes_hanger(self, frame):
		if "observation.clothes_hanger" in frame:
			return _to_numpy(frame["observation.clothes_hanger"]).reshape(-1)
		if "clothes_hanger" in frame:
			return _to_numpy(frame["clothes_hanger"]).reshape(-1)
		raise KeyError("Could not find clothes_hanger observation in frame.")

	def _get_image(self, frame, key):
		return _to_numpy(frame[key]).transpose(1, 2, 0)

	def get_reference_series(self, idx):
		if 7 <= idx <= 10 and self.clothes_hanger_obs is not None and self.clothes_hanger_obs.shape[1] > (idx - 7):
			return self.clothes_hanger_obs[:, idx - 7]
		if self.actions is not None and self.actions.shape[1] > idx:
			return self.actions[:, idx]
		if self.actions is not None:
			return np.zeros(self.actions.shape[0], dtype=float)
		return np.array([], dtype=float)

	def load_episode(self, idx):
		self.episode_idx = idx
		self.episode_start_idx = self.episode_indices["dataset_from_index"][idx]
		self.episode_to_idx = self.episode_indices["dataset_to_index"][idx]

		actions = []
		clothes_hanger_obs = []

		for i in range(self.episode_start_idx, self.episode_to_idx):
			frame = self.dataset[i]
			actions.append(_to_numpy(frame["action"]).reshape(-1))
			clothes_hanger_obs.append(self._get_clothes_hanger(frame))

		self.actions = np.stack(actions)
		self.clothes_hanger_obs = np.stack(clothes_hanger_obs)
		self.n = self.actions.shape[0]
		self.max_action_idx = max(0, self.actions.shape[1] - 1)

	def init_plot(self):
		self.fig = plt.figure(figsize=(12, 8))
		self.fig.suptitle(f"Dataset: {self.dataset_root.split('/')[-1]}", fontsize=16)
		self.ax_scene = self.fig.add_subplot(2, 3, 1)
		self.ax_wilson = self.fig.add_subplot(2, 3, 2)
		self.ax_sophie = self.fig.add_subplot(2, 3, 3)
		self.ax_action = self.fig.add_subplot(2, 1, 2)

		obs0 = self.dataset[self.episode_start_idx]
		self.im_scene = self.ax_scene.imshow(self._get_image(obs0, "scene_image"))
		self.ax_scene.set_title("Scene Image")
		self.ax_scene.axis("off")

		self.im_wilson = self.ax_wilson.imshow(self._get_image(obs0, "wrist_wilson_image"))
		self.ax_wilson.set_title("Wrist Wilson Image")
		self.ax_wilson.axis("off")

		if self.sophie_cam:
			self.im_sophie = self.ax_sophie.imshow(self._get_image(obs0, "wrist_sophie_image"))
			self.ax_sophie.set_title("Wrist Sophie Image")
			self.ax_sophie.axis("off")
		else:
			self.ax_sophie.set_title("No wrist_sophie_image")
			self.ax_sophie.axis("off")

		x = np.arange(self.n)
		self.line_action, = self.ax_action.plot(x, self.actions[:, self.action_idx], label="Action", linewidth=4)
		self.line_ref, = self.ax_action.plot(x, self.get_reference_series(self.action_idx), label="Reference", linewidth=2)
		self.vline = self.ax_action.axvline(0, color="r", linestyle="--", label="Current index")
		self.ax_action.set_xlabel("Relative index (0 .. n-1)")
		self.ax_action.set_ylabel("Value")
		self.ax_action.legend()
		self.ax_action.set_xlim(0, max(1, self.n - 1))

		plt.subplots_adjust(bottom=0.35)

		ax_slider_idx = plt.axes([0.2, 0.25, 0.6, 0.03])
		self.slider_idx = Slider(ax_slider_idx, "Index", 0, max(0, self.n - 1), valinit=0, valstep=1)

		ax_slider_act = plt.axes([0.2, 0.20, 0.6, 0.03])
		init_action_idx = max(0, min(self.max_action_idx, self.action_idx))
		self.slider_act = Slider(ax_slider_act, "Action Idx", 0, self.max_action_idx, valinit=init_action_idx, valstep=1)

		ax_text_ep = plt.axes([0.15, 0.05, 0.12, 0.05])
		self.textbox_ep = TextBox(ax_text_ep, "Episode idx", initial=str(self.episode_idx))

		self.slider_idx.on_changed(self.update_frame)
		self.slider_act.on_changed(self.update_action_idx)
		self.textbox_ep.on_submit(self.set_episode)

	def update_frame(self, _):
		i_rel = int(self.slider_idx.val)
		i_abs = self.episode_start_idx + i_rel
		obs = self.dataset[i_abs]

		self.im_scene.set_data(self._get_image(obs, "scene_image"))
		self.im_wilson.set_data(self._get_image(obs, "wrist_wilson_image"))
		if self.sophie_cam:
			self.im_sophie.set_data(self._get_image(obs, "wrist_sophie_image"))

		self.vline.set_xdata([i_rel, i_rel])
		self.fig.canvas.draw_idle()

	def update_action_idx(self, _):
		idx = int(self.slider_act.val)
		self.line_action.set_ydata(self.actions[:, idx])
		self.line_ref.set_ydata(self.get_reference_series(idx))
		self.ax_action.relim()
		self.ax_action.autoscale_view()
		self.fig.canvas.draw_idle()

	def set_episode(self, text):
		try:
			new_ep = int(text)
		except ValueError:
			return

		if not (0 <= new_ep < len(self.episode_indices["dataset_from_index"])):
			return

		self.load_episode(new_ep)
		self.update_all_plots()

	def update_all_plots(self):
		obs0 = self.dataset[self.episode_start_idx]
		self.im_scene.set_data(self._get_image(obs0, "scene_image"))
		self.im_wilson.set_data(self._get_image(obs0, "wrist_wilson_image"))
		if self.sophie_cam:
			self.im_sophie.set_data(self._get_image(obs0, "wrist_sophie_image"))

		x = np.arange(self.n)
		curr_action_idx = max(0, min(self.max_action_idx, int(self.slider_act.val)))
		self.line_action.set_xdata(x)
		self.line_ref.set_xdata(x)
		self.line_action.set_ydata(self.actions[:, curr_action_idx])
		self.line_ref.set_ydata(self.get_reference_series(curr_action_idx))

		self.ax_action.set_xlim(0, max(1, self.n - 1))
		self.ax_action.relim()
		self.ax_action.autoscale_view()

		self.slider_idx.ax.clear()
		self.slider_idx = Slider(self.slider_idx.ax, "Index", 0, max(0, self.n - 1), valinit=0, valstep=1)
		self.slider_idx.on_changed(self.update_frame)

		self.slider_act.ax.clear()
		self.slider_act = Slider(self.slider_act.ax, "Action Idx", 0, self.max_action_idx, valinit=curr_action_idx, valstep=1)
		self.slider_act.on_changed(self.update_action_idx)

		self.vline.set_xdata([0, 0])
		self.fig.canvas.draw_idle()

	def show(self):
		plt.show()


if __name__ == "__main__":
	dataset_root = "datasets/b-n200-INSTR0ACT-300k-1enc-EVAL3"
	repo_id = "whatevs"

	viewer = EpisodeViewer(dataset_root=dataset_root, repo_id=repo_id)
	viewer.load_episode(0)
	viewer.init_plot()
	viewer.show()

