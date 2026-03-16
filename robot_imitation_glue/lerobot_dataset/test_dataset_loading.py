from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _print_progress(current: int, total: int, label: str = "Progress") -> None:
    """Print an in-place progress bar on one terminal line."""
    width = 30
    ratio = 1.0 if total <= 0 else min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


dataset = LeRobotDataset(repo_id="whatevs", root="datasets/b")
#dataset = LeRobotDataset(repo_id="whatevs", root="datasets/b-n200-PREPR-INSTR1")
print(f"Num episodes: {dataset.num_episodes}")
print(dataset[0]["scene_image"].shape)


'''a = []
total_frames = len(dataset)
for i in range(total_frames):
    gripper_val = dataset[i]["gripper_on_static_robot"]
    a.append(gripper_val.ndim)
    _print_progress(i + 1, total_frames, label="Scanning frames")
# check if all elements of a are 0
print(set(a))'''
