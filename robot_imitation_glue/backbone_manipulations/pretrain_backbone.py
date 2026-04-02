from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import _replace_submodules
from lerobot.processor.core import TransitionKey
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from loguru import logger
import csv
from pathlib import Path
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18
from robot_imitation_glue.agents.lerobot_agent import make_lerobot_policy_for_inference


# We choose a model trained on b-n200-PREPR-INSTR1 and load its pre- and postprocessor. These
# depend solely on dataset statistics, so it doesn't matter exactly which model we load.
# It should be a INSTR1 model, trained on INSTR1 preprocessed dataset, because we want 
# to train on "normalised" clothes hanger values, which are found in the state (idx 7..10)
# of b-n200-PREPR-INSTR1. 
checkpoint_path = f"/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/b-n200-INSTR1-300k-1enc"
policy, preprocessor, postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
# release policy from GPU memory since we only need the pre/postprocessors for normalization stats
del policy

_preprocessor_normalizer_step = next(
    step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)
)
_postprocessor_unnormalizer_step = next(
    step for step in postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)
)


# ----- Config -----
DATASET_ROOT = "datasets/b-n200-PREPR-INSTR1"  # Trained on PREPROCESSED dataset: eval_diffusion will do the obs_preprocessor as well
REPO_ID = "whatevs"

# Train with both camera images as separate samples (no concatenation).
IMAGE_KEYS = (
    "observation.images.scene_image",
    "observation.images.wrist_wilson_image",
)

TARGET_KEY = "observation.state"
TARGET_SLICE = slice(7, 11)  # indices 7,8,9,10

BATCH_SIZE = 40
N_FOLDS = 5
TRAIN_ON_ALL_DATA = True
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
NUM_WORKERS = 4
SEED = 42
HEARTBEAT_EVERY_STEPS = 200
WANDB_PROJECT = "lerobot"
WANDB_RUN_NAME = f"bb-ALLDATA-{DATASET_ROOT.split('/')[-1]}"
WANDB_ENTITY = "remko-pr-ghent-university"

SAVE_PATH = lambda fold, epoch: (  # for cross validation
    f"outputs/vision_backbones/bb_{DATASET_ROOT.split('/')[-1]}-fold{fold + 1}-epoch{epoch}.pth"
)
SAVE_PATH_ALL_DATA = lambda epoch: (  # for training on all data (no validation split, no folds)
    f"outputs/vision_backbones/bb_{DATASET_ROOT.split('/')[-1]}-all-data-epoch{epoch}.pth"
)
FOLDS_CSV_PATH = f"outputs/vision_backbones/bb_ALLDATA_{DATASET_ROOT.split('/')[-1]}-fold_assignments.csv"


def normalize_image_and_state_with_lerobot_preprocessor(
    image_key: str,
    image: torch.Tensor,
    target_key: str,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    transition = {
        TransitionKey.OBSERVATION.value: {
            image_key: image,
            target_key: state,
        }
    }
    out = _preprocessor_normalizer_step(transition)
    normalized_obs = out[TransitionKey.OBSERVATION.value]
    return normalized_obs[image_key], normalized_obs[target_key]


def unnormalize_state_slice_with_lerobot_postprocessor(
    state_slice: torch.Tensor,
    target_key: str,
    target_slice: slice,
) -> torch.Tensor:
    if target_slice.stop is None or target_slice.start is None:
        raise ValueError("TARGET_SLICE must define both start and stop.")

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


class ImageToStateDataset(Dataset):
    def __init__(
        self,
        dataset: LeRobotDataset,
        image_keys: tuple[str, ...],
        target_key: str,
        target_slice: slice,
    ):
        self.dataset = dataset
        self.image_keys = image_keys
        self.target_key = target_key
        self.target_slice = target_slice
        self.n_image_keys = len(image_keys)

        if self.n_image_keys == 0:
            raise ValueError("image_keys must contain at least one image key.")

    def __len__(self):
        return len(self.dataset) * self.n_image_keys

    def __getitem__(self, idx):
        sample_idx = idx // self.n_image_keys
        image_key_idx = idx % self.n_image_keys
        image_key = self.image_keys[image_key_idx]

        item = self.dataset[sample_idx]
        image = item[image_key]
        target = item[self.target_key]

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        target = target.float()
        image, target = normalize_image_and_state_with_lerobot_preprocessor(
            image_key=image_key,
            image=image,
            target_key=self.target_key,
            state=target,
        )
        target = target[self.target_slice]
        if target.shape[0] != 4:
            raise ValueError(
                f"Expected target size 4 from {self.target_key}[{self.target_slice}], got {target.shape} at idx={idx}."
            )

        return image, target


def build_episode_folds(dataset: ImageToStateDataset, n_folds: int, seed: int):
    """Create episode-level folds to avoid frame leakage across train/validation."""
    episodes = dataset.dataset.meta.episodes
    starts = episodes["dataset_from_index"]
    n_episodes = len(starts)

    if n_episodes == 0:
        raise ValueError("No episodes found in dataset metadata.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > n_episodes:
        raise ValueError(f"n_folds ({n_folds}) cannot exceed number of episodes ({n_episodes}).")

    generator = torch.Generator().manual_seed(seed)
    episode_ids = torch.randperm(n_episodes, generator=generator).tolist()
    folds = [list(map(int, fold.tolist())) for fold in torch.tensor(episode_ids).chunk(n_folds)]
    return folds


def episode_ids_to_sample_indices(dataset: ImageToStateDataset, episode_ids: list[int]):
    episodes = dataset.dataset.meta.episodes
    starts = episodes["dataset_from_index"]
    ends = episodes["dataset_to_index"]
    n_image_keys = dataset.n_image_keys

    indices = []
    for ep_idx in episode_ids:
        start = int(starts[ep_idx])
        end = int(ends[ep_idx])
        for frame_idx in range(start, end):
            base = frame_idx * n_image_keys
            for cam_idx in range(n_image_keys):
                indices.append(base + cam_idx)
    return indices


def write_fold_assignments_csv(path: str, folds: list[list[int]], n_episodes: int):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    all_episode_ids = set(range(n_episodes))
    with path_obj.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "split", "episode_idx"])
        writer.writeheader()
        for fold_idx, val_ids in enumerate(folds, start=1):
            val_set = set(val_ids)
            train_ids = sorted(all_episode_ids - val_set)
            for ep_idx in train_ids:
                writer.writerow({"fold": fold_idx, "split": "train", "episode_idx": ep_idx})
            for ep_idx in sorted(val_set):
                writer.writerow({"fold": fold_idx, "split": "val", "episode_idx": ep_idx})


def build_model():
    backbone = resnet18(weights=None)
    backbone = _replace_submodules(
        root_module=backbone,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
    )
    backbone.fc = nn.Linear(backbone.fc.in_features, 4)
    return backbone


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train=True,
):
    model.train(train)
    running_loss = 0.0
    running_mae_raw = 0.0
    n_samples = 0

    for step_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if step_idx % HEARTBEAT_EVERY_STEPS == 0:
            logger.info(f"running step {step_idx}")

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            preds_norm = model(images)
            loss = criterion(preds_norm, targets)
            if train:
                loss.backward()
                optimizer.step()

        # Post-process to original units for easier tracking.
        preds_raw = unnormalize_state_slice_with_lerobot_postprocessor(
            preds_norm.detach(),
            target_key=TARGET_KEY,
            target_slice=TARGET_SLICE,
        )
        targets_raw = unnormalize_state_slice_with_lerobot_postprocessor(
            targets.detach(),
            target_key=TARGET_KEY,
            target_slice=TARGET_SLICE,
        )
        mae_raw = torch.mean(torch.abs(preds_raw - targets_raw))

        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        running_mae_raw += float(mae_raw.item()) * batch_size
        n_samples += batch_size

    mean_loss = running_loss / max(1, n_samples)
    mean_mae_raw = running_mae_raw / max(1, n_samples)
    return mean_loss, mean_mae_raw


def main():
    valid_image_keys = {"observation.images.scene_image", "observation.images.wrist_wilson_image"}
    if not set(IMAGE_KEYS).issubset(valid_image_keys):
        raise ValueError("IMAGE_KEYS may only contain scene_image and/or wrist_wilson_image keys.")

    torch.manual_seed(SEED)
    device = "cuda"

    lerobot_dataset = LeRobotDataset(repo_id=REPO_ID, root=DATASET_ROOT)
    logger.info(f"loaded dataset from {DATASET_ROOT} with {len(lerobot_dataset)} samples")

    pre_stats = _preprocessor_normalizer_step.stats
    post_stats = _postprocessor_unnormalizer_step.stats
    for image_key in IMAGE_KEYS:
        if image_key not in pre_stats or "mean" not in pre_stats[image_key] or "std" not in pre_stats[image_key]:
            raise KeyError(f"Loaded preprocessor is missing MEAN_STD stats for image key: {image_key}")
    if TARGET_KEY not in pre_stats:
        raise KeyError(f"Loaded preprocessor is missing stats for target key: {TARGET_KEY}")
    if TARGET_KEY not in post_stats or "min" not in post_stats[TARGET_KEY] or "max" not in post_stats[TARGET_KEY]:
        raise KeyError(f"Loaded postprocessor is missing MIN_MAX stats for target key: {TARGET_KEY}")

    target_min = torch.as_tensor(post_stats[TARGET_KEY]["min"]).float()[TARGET_SLICE]
    target_max = torch.as_tensor(post_stats[TARGET_KEY]["max"]).float()[TARGET_SLICE]

    full_dataset = ImageToStateDataset(
        dataset=lerobot_dataset,
        image_keys=IMAGE_KEYS,
        target_key=TARGET_KEY,
        target_slice=TARGET_SLICE,
    )

    if TRAIN_ON_ALL_DATA:
        logger.info("training on all data (no validation split, no cross-validation folds)")

        train_loader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        train_steps_per_epoch = len(train_loader)
        total_train_steps = train_steps_per_epoch * EPOCHS
        logger.info(
            f"all-data training | train samples={len(full_dataset)} | "
            f"train steps/epoch={train_steps_per_epoch} | total train steps={total_train_steps}"
        )

        wandb.init(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}-all-data",
            entity=WANDB_ENTITY,
            config={
                "dataset_root": DATASET_ROOT,
                "image_keys": list(IMAGE_KEYS),
                "image_normalization": "MEAN_STD",
                "target_key": TARGET_KEY,
                "target_slice": [TARGET_SLICE.start, TARGET_SLICE.stop],
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "train_on_all_data": True,
                "train_steps_per_epoch": train_steps_per_epoch,
                "total_train_steps": total_train_steps,
            },
            reinit=True,
        )

        criterion = nn.MSELoss()
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_train_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_mae_raw = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
            )

            logger.info(
                f"all-data training | epoch {epoch:03d} | "
                f"train_loss(norm)={train_loss:.6f} | train_mae(raw)={train_mae_raw:.6f}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss_norm": train_loss,
                    "train/train_mae_raw": train_mae_raw,
                },
                step=epoch,
            )

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                ckpt_path = SAVE_PATH_ALL_DATA(epoch)
                torch.save(
                    {
                        "image_keys": list(IMAGE_KEYS),
                        "image_normalization": "MEAN_STD",
                        "image_mean": {
                            image_key: torch.as_tensor(pre_stats[image_key]["mean"]).float().cpu()
                            for image_key in IMAGE_KEYS
                        },
                        "image_std": {
                            image_key: torch.as_tensor(pre_stats[image_key]["std"]).float().cpu()
                            for image_key in IMAGE_KEYS
                        },
                        "target_key": TARGET_KEY,
                        "target_slice": [TARGET_SLICE.start, TARGET_SLICE.stop],
                        "target_normalization": "MIN_MAX[-1,1]",
                        "target_min": target_min.cpu(),
                        "target_max": target_max.cpu(),
                        "state_dict": model.state_dict(),
                        "best_train_loss": best_train_loss,
                    },
                    ckpt_path,
                )
                logger.info(f"saved best checkpoint to: {ckpt_path}")

        wandb.finish()
        return

    folds = build_episode_folds(full_dataset, n_folds=N_FOLDS, seed=SEED)
    write_fold_assignments_csv(FOLDS_CSV_PATH, folds, n_episodes=len(full_dataset.dataset.meta.episodes["dataset_from_index"]))
    logger.info(f"saved fold episode assignments to: {FOLDS_CSV_PATH}")

    criterion = nn.MSELoss()
    fold_best_vals = []
    total_train_steps_all_folds = 0

    all_episode_ids = set(range(len(full_dataset.dataset.meta.episodes["dataset_from_index"])))
    for fold_idx, val_episode_ids in enumerate(folds):
        val_episode_set = set(val_episode_ids)
        train_episode_ids = sorted(all_episode_ids - val_episode_set)
        val_episode_ids = sorted(val_episode_set)

        train_indices = episode_ids_to_sample_indices(full_dataset, train_episode_ids)
        val_indices = episode_ids_to_sample_indices(full_dataset, val_episode_ids)

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        train_steps_per_epoch = len(train_loader)
        total_train_steps = train_steps_per_epoch * EPOCHS
        total_train_steps_all_folds += total_train_steps
        logger.info(
            f"fold {fold_idx + 1}/{N_FOLDS} | train episodes={len(train_episode_ids)} | "
            f"val episodes={len(val_episode_ids)} | train steps/epoch={train_steps_per_epoch} | "
            f"total train steps={total_train_steps}"
        )

        wandb.init(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}-fold{fold_idx + 1}",
            entity=WANDB_ENTITY,
            config={
                "dataset_root": DATASET_ROOT,
                "image_keys": list(IMAGE_KEYS),
                "image_normalization": "MEAN_STD",
                "target_key": TARGET_KEY,
                "target_slice": [TARGET_SLICE.start, TARGET_SLICE.stop],
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "n_folds": N_FOLDS,
                "fold_idx": fold_idx + 1,
                "train_steps_per_epoch": train_steps_per_epoch,
                "total_train_steps": total_train_steps,
            },
            reinit=True,
        )

        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val = float("inf")
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_mae_raw = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
            )
            val_loss, val_mae_raw = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train=False,
            )

            logger.info(
                f"fold {fold_idx + 1}/{N_FOLDS} | epoch {epoch:03d} | "
                f"train_loss(norm)={train_loss:.6f} | val_loss(norm)={val_loss:.6f} "
                f"| train_mae(raw)={train_mae_raw:.6f} | val_mae(raw)={val_mae_raw:.6f}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "fold": fold_idx + 1,
                    "train/loss_norm": train_loss,
                    "train/train_mae_raw": train_mae_raw,
                    "val/loss_norm": val_loss,
                    "val/val_mae_raw": val_mae_raw,
                },
                step=epoch,
            )

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = SAVE_PATH(fold_idx, epoch)
                torch.save(
                    {
                        "image_keys": list(IMAGE_KEYS),
                        "image_normalization": "MEAN_STD",
                        "image_mean": {
                            image_key: torch.as_tensor(pre_stats[image_key]["mean"]).float().cpu()
                            for image_key in IMAGE_KEYS
                        },
                        "image_std": {
                            image_key: torch.as_tensor(pre_stats[image_key]["std"]).float().cpu()
                            for image_key in IMAGE_KEYS
                        },
                        "target_key": TARGET_KEY,
                        "target_slice": [TARGET_SLICE.start, TARGET_SLICE.stop],
                        "target_normalization": "MIN_MAX[-1,1]",
                        "target_min": target_min.cpu(),
                        "target_max": target_max.cpu(),
                        "state_dict": model.state_dict(),
                        "best_val_loss": best_val,
                        "best_val_mae_raw": val_mae_raw,
                        "fold_idx": fold_idx + 1,
                        "val_episode_ids": val_episode_ids,
                    },
                    ckpt_path,
                )
                logger.info(f"saved best checkpoint to: {ckpt_path}")

        fold_best_vals.append(best_val)
        wandb.finish()

    logger.info(f"total train steps across all folds: {total_train_steps_all_folds}")
    logger.info(f"cross-validation summary | mean best val loss: {float(torch.tensor(fold_best_vals).mean()):.6f}")


if __name__ == "__main__":
    main()