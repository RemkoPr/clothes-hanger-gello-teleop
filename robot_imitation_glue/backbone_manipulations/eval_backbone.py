from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import _replace_submodules
from lerobot.policies.diffusion.configuration_diffusion import PreTrainedConfig
from lerobot.processor.core import TransitionKey
from lerobot.policies.factory import make_pre_post_processors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from loguru import logger
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from robot_imitation_glue.ur5station.eval_diffusion_lerobot import obs_preprocessor


# ----- Config -----
DATASET_ROOT = "datasets/b-n200-INSTR1-300k-1enc-EVAL2"
REPO_ID = "whatevs"
CHECKPOINT_PATH = "outputs/vision_backbones/bb_AUG_b-n200-PREPR-INSTR1-fold1-epoch89.pth"
NORMALIZER_CHECKPOINT_PATH = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/b-n200-INSTR1-300k-1enc"

# Which camera stream to feed to the backbone for prediction.
PRED_IMAGE_KEYS = ["observation.images.wrist_wilson_image", "observation.images.scene_image"]
MEAN_PRED_KEY = "mean(wrist, scene)"


# Use pre/postprocessors from a Diffusion model trained on the same preprocessed dataset.
config = PreTrainedConfig.from_pretrained(NORMALIZER_CHECKPOINT_PATH)
preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=NORMALIZER_CHECKPOINT_PATH)

_preprocessor_normalizer_step = next(
	step for step in preprocessor.steps if step.__class__.__name__ == "NormalizerProcessorStep"
)
_postprocessor_unnormalizer_step = next(
	step for step in postprocessor.steps if step.__class__.__name__ == "UnnormalizerProcessorStep"
)


def _wrapped_obs_preprocessor(obs_dict, include_instr=True):
	# Need to convert camera images from Tensors to opencv
	for key in obs_dict:
		if "image" in key:
			obs_dict[key] = obs_dict[key].numpy().transpose(1, 2, 0)*255
	return obs_preprocessor(obs_dict, include_instr=include_instr)


def _normalize_image_with_lerobot_preprocessor(image_key: str, image: torch.Tensor) -> torch.Tensor:
	transition = {TransitionKey.OBSERVATION.value: {image_key: image}}
	out = _preprocessor_normalizer_step(transition)
	return out[TransitionKey.OBSERVATION.value][image_key]


def _unnormalize_state_slice_with_lerobot_postprocessor(
	state_slice: torch.Tensor,
	target_key: str,
	target_slice: slice,
) -> torch.Tensor:
	if target_slice.stop is None or target_slice.start is None:
		raise ValueError("target_slice must define both start and stop.")

	expected_size = target_slice.stop - target_slice.start
	if state_slice.shape[-1] != expected_size:
		raise ValueError(f"Expected final dim {expected_size}, got {state_slice.shape[-1]}.")

	post_stats = _postprocessor_unnormalizer_step.stats
	if target_key not in post_stats or "min" not in post_stats[target_key]:
		raise KeyError(f"Missing MIN_MAX stats for {target_key} in loaded postprocessor.")

	min_vals = torch.as_tensor(post_stats[target_key]["min"], device=state_slice.device, dtype=state_slice.dtype)[target_slice]
	max_vals = torch.as_tensor(post_stats[target_key]["max"], device=state_slice.device, dtype=state_slice.dtype)[target_slice]
	denom = max_vals - min_vals
	denom = torch.where(denom == 0, torch.full_like(denom, 1e-8), denom)
	# Inverse of MIN_MAX normalization from [-1, 1] back to [min, max].
	return (state_slice + 1.0) * denom / 2.0 + min_vals


class BackboneEpisodeViewer:
	def __init__(self, dataset_root: str, repo_id: str, checkpoint_path: str, pred_image_keys: list[str]):
		self.dataset_root = dataset_root
		self.repo_id = repo_id
		self.checkpoint_path = checkpoint_path
		self.pred_image_keys = pred_image_keys
		self.plot_pred_keys = pred_image_keys + [MEAN_PRED_KEY]

		self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
		self.episode_indices = self.dataset.meta.episodes

		self.model, self.target_key, self.target_slice = self._load_model_and_target_spec(checkpoint_path)

		self.episode_idx = 0
		self.target_idx = 0
		self.n = 0
		self.episode_start_idx = None
		self.episode_to_idx = None
		self.true_targets = None
		self.pred_targets = None

		sample0 = self.dataset[0]
		self.has_sophie = "observation.images.wrist_sophie_image" in sample0

		self.fig = None
		self.ax_scene = None
		self.ax_wilson = None
		self.ax_sophie = None
		self.ax_target = None
		self.im_scene = None
		self.im_wilson = None
		self.im_sophie = None
		self.line_real = None
		self.line_preds = {}
		self.vline = None
		self.slider_idx = None
		self.slider_target = None
		self.textbox_ep = None

	def _load_model_and_target_spec(self, checkpoint_path: str):
		ckpt = torch.load(checkpoint_path, map_location="cpu")
		model = resnet18(weights=None)
		model = _replace_submodules(
			root_module=model,
			predicate=lambda x: isinstance(x, nn.BatchNorm2d),
			func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
		)
		model.fc = nn.Linear(model.fc.in_features, 4)
		model.load_state_dict(ckpt["state_dict"])
		model.eval().to("cuda")

		target_key = ckpt.get("target_key", "observation.state")
		target_slice_list = ckpt.get("target_slice", [7, 11])
		target_slice = slice(int(target_slice_list[0]), int(target_slice_list[1]))
		return model, target_key, target_slice

	def _to_display_image(self, tensor_img: torch.Tensor) -> np.ndarray:
		return tensor_img.cpu().numpy().squeeze().transpose(1, 2, 0)

	def _prepare_images_for_model(self, item: dict) -> list[torch.Tensor]:
		processed_obs = _wrapped_obs_preprocessor(item)
		images = []
		for key in self.pred_image_keys:
			image = processed_obs[key].float()
			image = _normalize_image_with_lerobot_preprocessor(key, image)
			image = image.to("cuda")
			images.append(image)
		return images

	def load_episode(self, idx: int):
		self.episode_idx = idx
		self.episode_start_idx = int(self.episode_indices["dataset_from_index"][idx])
		self.episode_to_idx = int(self.episode_indices["dataset_to_index"][idx])

		pred = {key: [] for key in self.pred_image_keys}
		true_list = []

		with torch.no_grad():
			for i in range(self.episode_start_idx, self.episode_to_idx):
				item = self.dataset[i]

				images = self._prepare_images_for_model(item)
				for i, image in enumerate(images):
					pred_norm = self.model(image).squeeze(0).cpu()
					pred_raw = _unnormalize_state_slice_with_lerobot_postprocessor(
						pred_norm,
						target_key=self.target_key,
						target_slice=self.target_slice,
					).numpy()
					pred[self.pred_image_keys[i]].append(pred_raw)

				state = item.get("observation.state", None)
				if state is not None and state.shape[0] >= 11:
					true_raw = state.float()[7:11].cpu().numpy()
				elif "clothes_hanger" in item:
					true_raw = item["clothes_hanger"].float()[:4].cpu().numpy()
				elif "observation.clothes_hanger" in item:
					true_raw = item["observation.clothes_hanger"].float()[:4].cpu().numpy()
				else:
					raise KeyError("Could not find ground-truth target (state[7:11] or clothes_hanger) in dataset item.")

				true_list.append(true_raw)

		self.pred_targets = {key: np.stack(pred[key]) for key in self.pred_image_keys}
		self.pred_targets[MEAN_PRED_KEY] = np.mean(
			np.stack([self.pred_targets[key] for key in self.pred_image_keys], axis=0),
			axis=0,
		)
		self.true_targets = np.stack(true_list)
		self.n = self.true_targets.shape[0]

		mses = {key: np.mean((self.true_targets - self.pred_targets[key]) ** 2, axis=0) for key in self.plot_pred_keys}
		maes = {key: np.mean(np.abs(self.true_targets - self.pred_targets[key]), axis=0) for key in self.plot_pred_keys}
		logger.info(f"Episode {idx} | MSE: {', '.join(f'{key}: {np.mean(mse):.6f}' for key, mse in mses.items())} | MAE={', '.join(f'{key}: {np.mean(mae):.6f}' for key, mae in maes.items())}")

	def init_plot(self):
		self.fig = plt.figure(figsize=(12, 8))
		self.fig.suptitle(
			f"Backbone Eval: {self.checkpoint_path.split('/')[-1]}\n"
			f"Dataset: {self.dataset_root.split('/')[-1]} | Inputs: {', '.join(self.pred_image_keys)} + {MEAN_PRED_KEY}",
			fontsize=14,
		)

		self.ax_scene = self.fig.add_subplot(2, 3, 1)
		self.ax_wilson = self.fig.add_subplot(2, 3, 2)
		self.ax_sophie = self.fig.add_subplot(2, 3, 3)
		self.ax_target = self.fig.add_subplot(2, 1, 2)

		obs0 = _wrapped_obs_preprocessor(self.dataset[self.episode_start_idx])
		self.im_scene = self.ax_scene.imshow(self._to_display_image(obs0["observation.images.scene_image"]))
		self.ax_scene.set_title("Scene Image")
		self.ax_scene.axis("off")

		self.im_wilson = self.ax_wilson.imshow(self._to_display_image(obs0["observation.images.wrist_wilson_image"]))
		self.ax_wilson.set_title("Wrist Wilson Image")
		self.ax_wilson.axis("off")

		if self.has_sophie:
			self.im_sophie = self.ax_sophie.imshow(
				self._to_display_image(obs0["observation.images.wrist_sophie_image"])
			)
			self.ax_sophie.set_title("Wrist Sophie Image")
		else:
			self.ax_sophie.set_title("No Wrist Sophie Image")
		self.ax_sophie.axis("off")

		x = np.arange(self.n)
		self.line_real, = self.ax_target.plot(x, self.true_targets[:, self.target_idx], label="Ground truth", linewidth=5)
		self.line_preds = {}
		for image_key in self.plot_pred_keys:
			pred_targets = self.pred_targets[image_key]
			if image_key == MEAN_PRED_KEY:
				label = "Backbone pred (mean wrist/scene)"
			else:
				label = f"Backbone pred ({image_key.split('.')[-1]})"
			line_pred, = self.ax_target.plot(x, pred_targets[:, self.target_idx], label=label)
			self.line_preds[image_key] = line_pred
		self.vline = self.ax_target.axvline(0, color="r", linestyle="--", label="Current index")
		self.ax_target.set_xlabel("Relative index (0 .. n-1)")
		self.ax_target.set_ylabel("Target value")
		self.ax_target.legend()
		self.ax_target.set_xlim(0, max(1, self.n - 1))

		plt.subplots_adjust(bottom=0.35)

		ax_slider_idx = plt.axes([0.2, 0.25, 0.6, 0.03])
		self.slider_idx = Slider(ax_slider_idx, "Index", 0, max(0, self.n - 1), valinit=0, valstep=1)

		ax_slider_target = plt.axes([0.2, 0.20, 0.6, 0.03])
		self.slider_target = Slider(ax_slider_target, "Target Idx", 0, 3, valinit=self.target_idx, valstep=1)

		ax_text_ep = plt.axes([0.15, 0.05, 0.12, 0.05])
		self.textbox_ep = TextBox(ax_text_ep, "Episode idx", initial=str(self.episode_idx))

		self.slider_idx.on_changed(self.update_frame)
		self.slider_target.on_changed(self.update_target_idx)
		self.textbox_ep.on_submit(self.set_episode)

	def update_frame(self, _):
		i_rel = int(self.slider_idx.val)
		i_abs = self.episode_start_idx + i_rel
		obs = _wrapped_obs_preprocessor(self.dataset[i_abs])

		self.im_scene.set_data(self._to_display_image(obs["observation.images.scene_image"]))
		self.im_wilson.set_data(self._to_display_image(obs["observation.images.wrist_wilson_image"]))
		if self.has_sophie:
			self.im_sophie.set_data(self._to_display_image(obs["observation.images.wrist_sophie_image"]))

		self.vline.set_xdata([i_rel, i_rel])
		self.fig.canvas.draw_idle()

	def update_target_idx(self, _):
		idx = int(self.slider_target.val)
		self.line_real.set_ydata(self.true_targets[:, idx])
		for image_key, line_pred in self.line_preds.items():
			line_pred.set_ydata(self.pred_targets[image_key][:, idx])
		self.ax_target.relim()
		self.ax_target.autoscale_view()
		self.fig.canvas.draw_idle()

	def set_episode(self, text: str):
		try:
			new_ep = int(text)
		except ValueError:
			logger.warning("Invalid episode index (not int).")
			return

		if not (0 <= new_ep < len(self.episode_indices["dataset_from_index"])):
			logger.warning("Episode index out of range.")
			return

		self.load_episode(new_ep)
		self.update_all_plots()

	def update_all_plots(self):
		obs0 = _wrapped_obs_preprocessor(self.dataset[self.episode_start_idx])
		self.im_scene.set_data(self._to_display_image(obs0["observation.images.scene_image"]))
		self.im_wilson.set_data(self._to_display_image(obs0["observation.images.wrist_wilson_image"]))
		if self.has_sophie:
			self.im_sophie.set_data(self._to_display_image(obs0["observation.images.wrist_sophie_image"]))

		x = np.arange(self.n)
		curr_target_idx = max(0, min(3, int(self.slider_target.val)))
		self.line_real.set_xdata(x)
		self.line_real.set_ydata(self.true_targets[:, curr_target_idx])
		for image_key, line_pred in self.line_preds.items():
			line_pred.set_xdata(x)
			line_pred.set_ydata(self.pred_targets[image_key][:, curr_target_idx])

		self.ax_target.set_xlim(0, max(1, self.n - 1))
		self.ax_target.relim()
		self.ax_target.autoscale_view()

		self.slider_idx.ax.clear()
		self.slider_idx = Slider(self.slider_idx.ax, "Index", 0, max(0, self.n - 1), valinit=0, valstep=1)
		self.slider_idx.on_changed(self.update_frame)

		self.vline.set_xdata([0, 0])
		self.fig.canvas.draw_idle()

	def show(self):
		plt.show()


if __name__ == "__main__":
	viewer = BackboneEpisodeViewer(
		dataset_root=DATASET_ROOT,
		repo_id=REPO_ID,
		checkpoint_path=CHECKPOINT_PATH,
		pred_image_keys=PRED_IMAGE_KEYS,
	)
	viewer.load_episode(2)
	viewer.init_plot()
	viewer.show()