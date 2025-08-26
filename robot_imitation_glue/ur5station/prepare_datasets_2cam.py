import numpy as np
from ur_analytic_ik import ur5e
import cv2
import torch

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset
from robot_imitation_glue.ur5station.data_collection import policy_action_to_abs_se3_converter


VISION_ONLY = True
WRIST_CAM_TO_INCLUDE = "wilson"
WRIST_CAM_TO_EXCLUDE = "sophie"

features_to_drop = [f"wrist_{WRIST_CAM_TO_EXCLUDE}_image_original", f"wrist_{WRIST_CAM_TO_INCLUDE}_image_original", f"wrist_{WRIST_CAM_TO_EXCLUDE}_image", "scene_image_original"]
crop_width_range = (320+70, 960-70)
crop_width = crop_width_range[1] - crop_width_range[0]
crop_height_range = (0, 720)
crop_height = crop_height_range[1] - crop_height_range[0]
crop_width = crop_width_range[1] - crop_width_range[0]
resize = (crop_width, crop_height)


def features_transform(features):
    features["observation.state"] = features.pop("state")
    if VISION_ONLY:
        features["observation.state"]["shape"] = (7,)
    else:
        features["observation.state"]["shape"] = (11,)
    features[f"observation.images.wrist_{WRIST_CAM_TO_INCLUDE}_image"] = features.pop(f"wrist_{WRIST_CAM_TO_INCLUDE}_image")
    features[f"observation.images.wrist_{WRIST_CAM_TO_INCLUDE}_image"]["shape"] = (3, resize[1], resize[0])
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
    scene_image = torch.tensor(cv2.resize(np.array(frame["scene_image"])[crop_height_range[0]:crop_height_range[1], crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_LINEAR))
    wrist_image = torch.tensor(cv2.resize(np.array(frame[f"wrist_{WRIST_CAM_TO_INCLUDE}_image"])[crop_height_range[0]:crop_height_range[1], crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR))
    
    new_frame = frame.copy()
    new_frame.pop("scene_image")
    new_frame.pop(f"wrist_{WRIST_CAM_TO_INCLUDE}_image")
    new_frame.pop("state")
    if VISION_ONLY:
        new_frame["observation.state"] = np.concatenate([current_joints, current_gripper]).astype(np.float32)
    else:
        new_frame["observation.state"] = np.concatenate([current_joints, current_gripper, clothes_hanger]).astype(np.float32)
    new_frame["action"] = np.concatenate([action_joints, action_gripper]).astype(np.float32)
    new_frame["observation.images.scene_image"] = scene_image
    new_frame[f"observation.images.wrist_{WRIST_CAM_TO_INCLUDE}_image"] = wrist_image

    return new_frame


transform_dataset(
    root_dir="datasets/clothes-hanger-v3-raw",
    new_root_dir=f"datasets/clothes-hanger-v3p7-w{resize[0]}h{resize[1]}-2cam-visionOnly-n100",
    repo_id="clothes-hanger-repo-v3p7",
    transform_fn=joints_frame_transform,  # 
    transform_features_fn=features_transform,
    features_to_drop=features_to_drop,
    episodes_to_drop=[i for i in range(70)] + [i for i in range(170, 180)]#[140, 171]#[i for i in range(2, 179)], #
)
