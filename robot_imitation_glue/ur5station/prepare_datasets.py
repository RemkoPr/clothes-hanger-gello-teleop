from functools import partial
import numpy as np
import cv2
import torch

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset

# scenarios:

# 1. convert to joint actions and joint configuration state with absolute gripper

CH_THRESHOLD = 0

features_to_drop = ['scene_image_original', 'wrist_wilson_image_original', 'wrist_sophie_image', 'wrist_sophie_image_original']
crop_width_range = (320+0, 960-0)
crop_height_range = (0, 720)
crop_height = crop_height_range[1] - crop_height_range[0]
crop_width = crop_width_range[1] - crop_width_range[0]
resize = (2*crop_width//3, 2*crop_height//3)


def features_transform(features, include_instr=True, instr_as_action=False):
    features["observation.state"] = features.pop("state")
    if include_instr:
        features["observation.state"]["shape"] = (11,)
    else:
        features["observation.state"]["shape"] = (7,)
    features["observation.images.wrist_wilson_image"] = features.pop("wrist_wilson_image")
    features["observation.images.wrist_wilson_image"]["shape"] = (3, resize[1], resize[0])
    features["observation.images.scene_image"] = features.pop("scene_image")
    features["observation.images.scene_image"]["shape"] = (3, resize[1], resize[0])
    if not instr_as_action:
        features["action"]["shape"] = (7,)
    else:
        features["action"]["shape"] = (11,)

    print("processed features:")
    print(features)
    return features


def joints_frame_transform(frame, init_frame=None, include_instr=True, instr_as_action=False):
    current_joints = frame["teleop_robot_joints"].numpy()
    current_gripper = frame["gripper_on_static_robot"]  # convoluted expression, change to simply "frame["gripper_on_static_robot"]"
    action_joints = frame["action"][:6]  #.numpy()
    action_gripper = np.array([frame["action"][6]])  #.numpy()
    if init_frame is not None:
        clothes_hanger = np.maximum(init_frame["clothes_hanger"].numpy() - frame["clothes_hanger"].numpy(), 0)
    else:
        clothes_hanger = frame["clothes_hanger"].numpy()
    if CH_THRESHOLD:
        clothes_hanger = (clothes_hanger > CH_THRESHOLD).astype(np.float32)
    # crop and resize
    scene_image = torch.tensor(cv2.resize(np.array(frame["scene_image"])[crop_height_range[0]:crop_height_range[1], crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_LINEAR))
    wrist_wilson_image = torch.tensor(cv2.resize(np.flip(np.array(frame["wrist_wilson_image"]), axis=0)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR))

    new_frame = frame.copy()
    new_frame.pop("scene_image")
    new_frame.pop("wrist_wilson_image")
    new_frame.pop("state")
    if include_instr:
        new_frame["observation.state"] = np.concatenate([current_joints, current_gripper, clothes_hanger]).astype(np.float32) 
    else:
        new_frame["observation.state"] = np.concatenate([current_joints, current_gripper]).astype(np.float32)
    if include_instr and instr_as_action:
        raise ValueError("Cannot include instrumentation in state and also use it as action")
    if not instr_as_action:
        new_frame["action"] = np.concatenate([action_joints, action_gripper]).astype(np.float32)
    else:
        new_frame["action"] = np.concatenate([action_joints, action_gripper, clothes_hanger]).astype(np.float32)
    new_frame["observation.images.scene_image"] = scene_image
    new_frame["observation.images.wrist_wilson_image"] = wrist_wilson_image

    return new_frame


INCLUDE_INSTR = False
INSTR_AS_ACTION = True
root_dir = "datasets/b"
transform_dataset(
    root_dir=root_dir,
    new_root_dir=root_dir + f"-n200-PREPR-INSTR{1 if INCLUDE_INSTR else 0}{'ACT' if INSTR_AS_ACTION else ''}",
    transform_fn=partial(joints_frame_transform, include_instr=INCLUDE_INSTR, instr_as_action=INSTR_AS_ACTION),  # 
    transform_features_fn=partial(features_transform, include_instr=INCLUDE_INSTR, instr_as_action=INSTR_AS_ACTION),
    features_to_drop=features_to_drop,
    episodes_to_drop=[21],#[i for i in range(1, 179)], #[140, 171]#
    frames_to_drop=[0],  # drop every first frame: this is the clotheshanger baseline measurement in home pose
    normalise_ch_values=True,
    remove_corrupted_frames=True
)
