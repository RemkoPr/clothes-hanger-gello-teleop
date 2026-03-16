import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DATASET_ROOT = "datasets/b"
REPO_ID = "whatevs"
IMAGE_KEY = "scene_image"
MAX_EPISODES = None  # e.g. 100, or None for all episodes
DIFF_CLIP = 0.35  # absolute diff that maps to full colormap saturation
ALPHA_SCALE = 0.9  # max alpha used for each diff layer


def _to_hwc_float01(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert image to float32 (H, W, C) in [0, 1]."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _episode_start_index(dataset: LeRobotDataset, episode_idx: int) -> int:
    # Support both metadata layouts seen across LeRobot versions.
    episodes_meta = dataset.meta.episodes
    if isinstance(episodes_meta, dict) and episode_idx in episodes_meta:
        return int(episodes_meta[episode_idx]["dataset_from_index"])
    return int(episodes_meta["dataset_from_index"][episode_idx])


def _difference_to_rgba(delta: np.ndarray, clip_value: float, alpha_scale: float) -> np.ndarray:
    """
    Convert signed RGB difference (H, W, C) into an RGBA layer.

    RGB comes from a diverging colormap of signed luminance delta.
    Alpha comes from absolute per-pixel delta magnitude.
    """
    signed_delta = np.mean(delta, axis=-1)  # luminance-like signed difference
    norm = colors.TwoSlopeNorm(vmin=-clip_value, vcenter=0.0, vmax=clip_value)
    rgb = plt.get_cmap("seismic")(norm(signed_delta))[..., :3]

    magnitude = np.max(np.abs(delta), axis=-1)
    alpha = np.clip(magnitude / clip_value, 0.0, 1.0) * alpha_scale
    return np.dstack([rgb, alpha])


def _composite_rgba_layers(base_rgb: np.ndarray, rgba_layers: list[np.ndarray]) -> np.ndarray:
    """Alpha-composite a list of RGBA layers on top of base RGB."""
    out = base_rgb.copy()
    for layer in rgba_layers:
        alpha = layer[..., 3:4]
        out = layer[..., :3] * alpha + out * (1.0 - alpha)
    return np.clip(out, 0.0, 1.0)


def main() -> None:
	dataset = LeRobotDataset(repo_id=REPO_ID, root=DATASET_ROOT)

	num_episodes = dataset.num_episodes if MAX_EPISODES is None else min(dataset.num_episodes, MAX_EPISODES)
	if num_episodes < 1:
		raise ValueError("Dataset has no episodes to visualize.")

	first_frames: list[np.ndarray] = []
	for ep_idx in range(num_episodes):
		first_frame_idx = _episode_start_index(dataset, ep_idx)
		sample = dataset[first_frame_idx]
		first_frames.append(_to_hwc_float01(sample[IMAGE_KEY]))

	frames = np.stack(first_frames, axis=0)
	base_frame = frames[0]

	rgba_diffs: list[np.ndarray] = []
	max_abs_delta = np.zeros(base_frame.shape[:2], dtype=np.float32)
	for ep_idx in range(1, num_episodes):
		delta = frames[ep_idx] - base_frame
		max_abs_delta = np.maximum(max_abs_delta, np.max(np.abs(delta), axis=-1))
		rgba_diffs.append(_difference_to_rgba(delta, clip_value=DIFF_CLIP, alpha_scale=ALPHA_SCALE))

	overlay = _composite_rgba_layers(base_frame, rgba_diffs)
	alpha_strength = np.clip(max_abs_delta / DIFF_CLIP, 0.0, 1.0)

	fig, axes = plt.subplots(1, 3, figsize=(18, 6))

	axes[0].imshow(base_frame)
	axes[0].set_title("Reference: first episode initial frame")
	axes[0].axis("off")

	axes[1].imshow(overlay)
	axes[1].set_title("Overlay of (episode_i - reference) with alpha")
	axes[1].axis("off")

	im = axes[2].imshow(alpha_strength, cmap="magma", vmin=0.0, vmax=1.0)
	axes[2].set_title("Max subtraction magnitude (alpha driver)")
	axes[2].axis("off")
	cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
	cbar.set_label("Normalized |difference|")

	fig.suptitle(
		f"Initial frame subtraction overlay across {num_episodes} episodes\n"
		f"key={IMAGE_KEY} | reference=episode 0 | clip={DIFF_CLIP} | alpha_scale={ALPHA_SCALE}",
		fontsize=11,
	)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
