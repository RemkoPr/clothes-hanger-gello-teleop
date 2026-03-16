import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
from loguru import logger

BATCH_SIZE = 48
NUM_WORKERS = 8
DEVICE = torch.device("cuda")
KEYS = ("scene_image", "wrist_wilson_image")

dataset = LeRobotDataset(repo_id="whatevs", root="datasets/legacy/b-n200-PREPR-INSTR0")
logger.info(f"Num episodes: {dataset.num_episodes}")
logger.info(f"Num frames: {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)

diff_frame_indices = []
scene_image_diffs = []
wrist_wilson_image_diffs = []

prev_scene = None
prev_wrist = None
prev_episode_index = None
frames_processed = 0

for batch in loader:
    scene = batch["observation.images.scene_image"].to(DEVICE, non_blocking=True)     # (B, C, H, W)
    wrist = batch["observation.images.wrist_wilson_image"].to(DEVICE, non_blocking=True)
    episode_index = batch["episode_index"]  # (B,) on CPU
    B = len(scene)

    # diff across batch boundary first (index frames_processed-1), so output stays sorted
    if prev_scene is not None and episode_index[0].item() == prev_episode_index:
        scene_diff_boundary = (scene[0] - prev_scene).abs().mean()
        wrist_diff_boundary = (wrist[0] - prev_wrist).abs().mean()
        diff_frame_indices.append(frames_processed - 1)
        scene_image_diffs.append(scene_diff_boundary.item())
        wrist_wilson_image_diffs.append(wrist_diff_boundary.item())

    # diff within the batch: only between frames of the same episode
    same_episode_intra = (episode_index[1:] == episode_index[:-1])  # (B-1,) bool
    scene_diff_intra  = (scene[1:]  - scene[:-1]).abs().mean(dim=(1, 2, 3))
    wrist_diff_intra  = (wrist[1:]  - wrist[:-1]).abs().mean(dim=(1, 2, 3))
    intra_indices = torch.arange(frames_processed, frames_processed + B - 1)
    diff_frame_indices.extend(intra_indices[same_episode_intra].tolist())
    scene_image_diffs.extend(scene_diff_intra[same_episode_intra].tolist())
    wrist_wilson_image_diffs.extend(wrist_diff_intra[same_episode_intra].tolist())

    prev_scene = scene[-1]
    prev_wrist = wrist[-1]
    prev_episode_index = episode_index[-1].item()

    frames_processed += B
    if frames_processed % 1000 < BATCH_SIZE:
        logger.info(f"Processed {frames_processed}/{len(dataset)} frames")

logger.info(f"Done. Total diffs: {len(scene_image_diffs)}")

plt.plot(diff_frame_indices, scene_image_diffs, label="scene_image_diff")
plt.plot(diff_frame_indices, wrist_wilson_image_diffs, label="wrist_wilson_image_diff")
plt.xlabel("Frame Index")
plt.ylabel("Absolute Difference")
plt.title("Camera Image Differences")
plt.legend()
plt.show()