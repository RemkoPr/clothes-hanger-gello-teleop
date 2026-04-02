from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import _replace_submodules
from lerobot.processor.core import TransitionKey
from loguru import logger
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from robot_imitation_glue.ur5station.eval_diffusion_lerobot import obs_preprocessor
from robot_imitation_glue.agents.lerobot_agent import make_lerobot_policy_for_inference
from tqdm import tqdm


# ----- Config -----
CHECKPOINT_PATH = "outputs/vision_backbones/bb_b-n200-PREPR-INSTR1-all-data-epoch15.pth"
NORMALIZER_CHECKPOINT_PATH = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/b-n200-INSTR1-300k-1enc"

DATASETS = [
	#"datasets/b-n200-INSTR0-300k-1enc-EVAL",
	#"datasets/b-n200-INSTR0-300k-1enc-EVAL2",
	#"datasets/b-n200-INSTR0-300k-1enc-EVAL3",
	"datasets/b-n200-INSTR1-300k-1enc-EVAL",
	"datasets/b-n200-INSTR1-300k-1enc-EVAL2",
	"datasets/b-n200-INSTR1-300k-1enc-EVAL3",
]

PRED_IMAGE_KEYS = [
	"observation.images.wrist_wilson_image",
	"observation.images.scene_image",
]
MEAN_PRED_KEY = "mean(wrist, scene)"
MIN_PRED_KEY = "min(wrist, scene)"


# Use pre/postprocessors from a Diffusion model trained on the same preprocessed dataset.
policy, preprocessor, postprocessor = make_lerobot_policy_for_inference(NORMALIZER_CHECKPOINT_PATH)
del policy

_preprocessor_normalizer_step = next(
	step for step in preprocessor.steps if step.__class__.__name__ == "NormalizerProcessorStep"
)
_postprocessor_unnormalizer_step = next(
	step for step in postprocessor.steps if step.__class__.__name__ == "UnnormalizerProcessorStep"
)


def _wrapped_obs_preprocessor(obs_dict, include_instr=True):
	# Need to convert camera images from Tensors to opencv
	processed_obs = dict(obs_dict)
	for key, value in processed_obs.items():
		if "image" not in key:
			continue

		if isinstance(value, torch.Tensor):
			arr = value.detach().cpu().numpy()
		elif isinstance(value, np.ndarray):
			arr = value
		else:
			continue

		# Convert CHW -> HWC when needed.
		if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
			arr = arr.transpose(1, 2, 0)

		# Scale normalized floats only; avoid re-scaling arrays already in [0, 255].
		if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and arr.max() <= 1.0:
			arr = arr * 255.0

		processed_obs[key] = arr

	return obs_preprocessor(processed_obs, include_instr=include_instr)


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


def load_model_and_target_spec(checkpoint_path: str):
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


def evaluate_dataset(dataset_root: str, repo_id: str, model: nn.Module, target_key: str, target_slice: slice):
	"""Evaluate model on a single dataset. Returns (pred_targets_by_key, true_targets)."""
	dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
	episode_indices = dataset.meta.episodes

	pred_lists = {image_key: [] for image_key in PRED_IMAGE_KEYS}
	true_list = []

	total_frames = int(episode_indices["dataset_to_index"][-1])
	pbar = tqdm(total=total_frames, desc=f"  {dataset_root.split('/')[-1]}", leave=False)

	with torch.no_grad():
		for ep_idx in range(len(episode_indices["dataset_from_index"])):
			ep_start = int(episode_indices["dataset_from_index"][ep_idx])
			ep_end = int(episode_indices["dataset_to_index"][ep_idx])

			for i in range(ep_start, ep_end):
				item = dataset[i]

				# Prepare normalized observations once, then run inference for each image key.
				processed_obs = _wrapped_obs_preprocessor(item)

				for image_key in PRED_IMAGE_KEYS:
					image = processed_obs[image_key].float()
					image = _normalize_image_with_lerobot_preprocessor(image_key, image)
					image = image.to("cuda")

					pred_norm = model(image).squeeze(0).cpu()
					pred_raw = _unnormalize_state_slice_with_lerobot_postprocessor(
						pred_norm,
						target_key=target_key,
						target_slice=target_slice,
					).numpy()
					pred_lists[image_key].append(pred_raw)

				# Get ground truth
				state = item.get("observation.state", None)
				if state is not None and state.shape[0] >= 11:
					true_raw = state.float()[7:11].cpu().numpy()
				elif "clothes_hanger" in item:
					true_raw = item["clothes_hanger"].float()[:4].cpu().numpy()
				elif "observation.clothes_hanger" in item:
					true_raw = item["observation.clothes_hanger"].float()[:4].cpu().numpy()
				else:
					raise KeyError("Could not find ground-truth target.")

				true_list.append(true_raw)
				pbar.update(1)

	pbar.close()

	pred_targets = {image_key: np.stack(preds) for image_key, preds in pred_lists.items()}
	pred_targets[MEAN_PRED_KEY] = np.mean(np.stack([pred_targets[k] for k in PRED_IMAGE_KEYS], axis=0), axis=0)
	pred_targets[MIN_PRED_KEY] = np.min(np.stack([pred_targets[k] for k in PRED_IMAGE_KEYS], axis=0), axis=0)
	true_targets = np.stack(true_list)

	return pred_targets, true_targets


def compute_metrics(pred_targets: np.ndarray, true_targets: np.ndarray):
	mse_per_channel = np.mean((true_targets - pred_targets) ** 2, axis=0)
	mae_per_channel = np.mean(np.abs(true_targets - pred_targets), axis=0)
	mse_overall = np.mean(mse_per_channel)
	mae_overall = np.mean(mae_per_channel)
	return mse_per_channel, mae_per_channel, mse_overall, mae_overall


def print_results(dataset_root: str, image_key: str, pred_targets: np.ndarray, true_targets: np.ndarray):
	"""Print per-channel and overall MSE/MAE."""
	mse_per_channel, mae_per_channel, mse_overall, mae_overall = compute_metrics(pred_targets, true_targets)

	print(f"\n{dataset_root} | {image_key}")
	print(f"  Frames: {len(true_targets)}")
	print(f"  Per-channel MSE: {', '.join(f'{mse:.6f}' for mse in mse_per_channel)}")
	print(f"  Per-channel MAE: {', '.join(f'{mae:.6f}' for mae in mae_per_channel)}")
	print(f"  Overall MSE: {mse_overall:.6f}")
	print(f"  Overall MAE: {mae_overall:.6f}")


def print_key_comparison(dataset_root: str, pred_targets_by_key: dict[str, np.ndarray], true_targets: np.ndarray):
	key_a = PRED_IMAGE_KEYS[0]
	key_b = PRED_IMAGE_KEYS[1]
	key_m = MEAN_PRED_KEY
	key_i = MIN_PRED_KEY
	_, _, mse_a, mae_a = compute_metrics(pred_targets_by_key[key_a], true_targets)
	_, _, mse_b, mae_b = compute_metrics(pred_targets_by_key[key_b], true_targets)
	_, _, mse_m, mae_m = compute_metrics(pred_targets_by_key[key_m], true_targets)
	_, _, mse_i, mae_i = compute_metrics(pred_targets_by_key[key_i], true_targets)
	print(f"  Delta overall ({key_b} - {key_a}): MSE={mse_b - mse_a:+.6f}, MAE={mae_b - mae_a:+.6f}")
	print(f"  Delta overall ({key_m} - {key_a}): MSE={mse_m - mse_a:+.6f}, MAE={mae_m - mae_a:+.6f}")
	print(f"  Delta overall ({key_m} - {key_b}): MSE={mse_m - mse_b:+.6f}, MAE={mae_m - mae_b:+.6f}")
	print(f"  Delta overall ({key_i} - {key_a}): MSE={mse_i - mse_a:+.6f}, MAE={mae_i - mae_a:+.6f}")
	print(f"  Delta overall ({key_i} - {key_b}): MSE={mse_i - mse_b:+.6f}, MAE={mae_i - mae_b:+.6f}")
	print(f"  Delta overall ({key_i} - {key_m}): MSE={mse_i - mse_m:+.6f}, MAE={mae_i - mae_m:+.6f}")


if __name__ == "__main__":
	print("Loading model...")
	model, target_key, target_slice = load_model_and_target_spec(CHECKPOINT_PATH)
	print(f"Model checkpoint: {CHECKPOINT_PATH}")
	print(f"Target key: {target_key}, slice: {target_slice}")
	print(f"Input image keys: {', '.join(PRED_IMAGE_KEYS)}")
	print(f"Also reporting: {MEAN_PRED_KEY}, {MIN_PRED_KEY}")
	print()

	all_results = {image_key: [] for image_key in PRED_IMAGE_KEYS + [MEAN_PRED_KEY, MIN_PRED_KEY]}
	all_truth = []

	for dataset_root in DATASETS:
		try:
			pred_targets_by_key, true_targets = evaluate_dataset(
				dataset_root=dataset_root,
				repo_id="whatevs",
				model=model,
				target_key=target_key,
				target_slice=target_slice,
			)
			all_truth.append(true_targets)
			for image_key in PRED_IMAGE_KEYS + [MEAN_PRED_KEY, MIN_PRED_KEY]:
				all_results[image_key].append(pred_targets_by_key[image_key])
				print_results(dataset_root, image_key, pred_targets_by_key[image_key], true_targets)
			print_key_comparison(dataset_root, pred_targets_by_key, true_targets)
		except Exception as e:
			print(f"\n{dataset_root}")
			print(f"  ERROR: {e}")

	# Overall summary
	print("\n" + "=" * 80)
	print("OVERALL SUMMARY")
	print("=" * 80)
	if not all_truth:
		print("No successful dataset evaluations.")
		raise SystemExit(1)

	all_true = np.concatenate(all_truth)
	for image_key in PRED_IMAGE_KEYS + [MEAN_PRED_KEY, MIN_PRED_KEY]:
		all_pred = np.concatenate(all_results[image_key])
		print_results("All datasets combined", image_key, all_pred, all_true)

	all_pred_by_key = {image_key: np.concatenate(all_results[image_key]) for image_key in PRED_IMAGE_KEYS + [MEAN_PRED_KEY, MIN_PRED_KEY]}
	print_key_comparison("All datasets combined", all_pred_by_key, all_true)
