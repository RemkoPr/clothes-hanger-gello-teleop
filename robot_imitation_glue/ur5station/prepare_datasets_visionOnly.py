import numpy as np
from ur_analytic_ik import ur5e
import cv2
import torch

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset
from robot_imitation_glue.ur5station.data_collection import policy_action_to_abs_se3_converter

# scenarios:

# 1. convert to joint actions and joint configuration state with absolute gripper


features_to_drop = ["wrist_image_original", "scene_image_original"]
crop_width_range = (320+0, 960-140-0)
crop_width = crop_width_range[1] - crop_width_range[0]
resize = (crop_width, 720)


def features_transform(features):
    features["observation.state"] = features.pop("state")
    features["observation.state"]["shape"] = (7,)
    features["observation.images.wrist_image"] = features.pop("wrist_image")
    features["observation.images.wrist_image"]["shape"] = (3, resize[1], resize[0])
    features["observation.images.scene_image"] = features.pop("scene_image")
    features["observation.images.scene_image"]["shape"] = (3, resize[1], resize[0])
    features["action"]["shape"] = (7,)

    print("processed features:")
    print(features)
    return features


def joints_frame_transform(frame):
    current_joints = frame["joints"].numpy()
    current_gripper = np.array([frame["gripper_states"][1]])
    action_joints = frame["action"][:6]  #.numpy()
    action_gripper = np.array([frame["action"][6]])  #.numpy()
    clothes_hanger = frame["clothes_hanger"].numpy()
    # crop and resize
    scene_image = torch.tensor(cv2.resize(np.array(frame["scene_image"])[:, crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_CUBIC))
    wrist_image = torch.tensor(cv2.resize(np.array(frame["wrist_image"])[:, crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_CUBIC))

    new_frame = frame.copy()
    new_frame.pop("scene_image")
    new_frame.pop("wrist_image")
    new_frame.pop("state")
    new_frame["observation.state"] = np.concatenate([current_joints, current_gripper]).astype(np.float32)
    new_frame["action"] = np.concatenate([action_joints, action_gripper]).astype(np.float32)
    new_frame["observation.images.scene_image"] = scene_image
    new_frame["observation.images.wrist_image"] = wrist_image

    return new_frame


transform_dataset(
    root_dir="datasets/clothes-hanger-raw",
    new_root_dir="datasets/clothes-hanger-v2-prepr-w500-visionOnly",
    repo_id="clothes-hanger-repo-visionOnly",
    transform_fn=joints_frame_transform,  # 
    transform_features_fn=features_transform,
    features_to_drop=features_to_drop,
    episodes_to_drop=[]#[i for i in range(2, 50)],
)
