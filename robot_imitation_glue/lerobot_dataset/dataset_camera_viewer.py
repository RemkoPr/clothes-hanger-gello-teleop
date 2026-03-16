import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numpy as np
from loguru import logger

dataset = LeRobotDataset(repo_id="whatevs", root="datasets/b")
logger.info(f"Num episodes: {dataset.num_episodes}")
logger.info(f"Num frames: {len(dataset)}")

camera_keys = dataset.meta.camera_keys
logger.info(f"Camera keys: {camera_keys}")

assert len(camera_keys) >= 2, f"Expected at least 2 camera keys, got: {camera_keys}"
cam0_key, cam1_key = camera_keys[0], camera_keys[1]

num_frames = len(dataset)


def get_frame_images(idx: int):
    sample = dataset[idx]
    # tensors are (C, H, W) float in [0, 1] — convert to (H, W, C) uint8
    def to_hwc(t: torch.Tensor) -> np.ndarray:
        arr = t.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return arr

    return to_hwc(sample[cam0_key]), to_hwc(sample[cam1_key])


# ── build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.18)

img0, img1 = get_frame_images(0)
im0 = axes[0].imshow(img0)
im1 = axes[1].imshow(img1)
axes[0].set_title(cam0_key, fontsize=9)
axes[1].set_title(cam1_key, fontsize=9)
for ax in axes:
    ax.axis("off")

frame_title = fig.suptitle("Frame 0", fontsize=11)

# ── slider ────────────────────────────────────────────────────────────────────
ax_slider = plt.axes([0.15, 0.08, 0.60, 0.04])
slider = Slider(ax_slider, "Frame", 0, num_frames - 1, valinit=0, valstep=1)

# ── textbox ───────────────────────────────────────────────────────────────────
ax_textbox = plt.axes([0.80, 0.08, 0.08, 0.04])
textbox = TextBox(ax_textbox, "Go to: ", initial="0")


def _update_display(idx: int):
    i0, i1 = get_frame_images(idx)
    im0.set_data(i0)
    im1.set_data(i1)
    frame_title.set_text(f"Frame {idx} / {num_frames - 1}")
    fig.canvas.draw_idle()


def on_slider_change(val):
    idx = int(slider.val)
    textbox.set_val(str(idx))
    _update_display(idx)


def on_textbox_submit(text):
    try:
        idx = int(text)
        idx = max(0, min(num_frames - 1, idx))
    except ValueError:
        return
    slider.set_val(idx)  # triggers on_slider_change


slider.on_changed(on_slider_change)
textbox.on_submit(on_textbox_submit)

plt.show()
