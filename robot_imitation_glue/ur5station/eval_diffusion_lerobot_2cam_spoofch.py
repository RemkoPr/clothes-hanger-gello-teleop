import numpy as np
import torch
import os
import cv2
from loguru import logger
import rerun as rr

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.gello.gello_agent import GelloAgent, DynamixelConfig
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.ur5station.data_collection import policy_action_converter_nop
from robot_imitation_glue.ur5station.ur5_robot_env import (
    GELLO_AGENT_PORT,
    UR5eStation,
    convert_gello_actions_to_joint_space_robot_pose
)

if __name__ == "__main__":
    checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-13/18-32-38_clothes-hanger-v3.5-2cam/checkpoints/100000/pretrained_model"
    train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p5-w500h720-2cam"  # preprocessed train dataset
    #checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-15/15-09-52_clothes-hanger-v3.6-2cam-visionOnly/checkpoints/100000/pretrained_model"
    #train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p6-w500h720-2cam-visionOnly"  # preprocessed train dataset
    #checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-28/18-40-57_clothes-hanger-v3.8-visionAugmentedByInstrRollouts/checkpoints/100000/pretrained_model"
    #train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p8-visionAugmentedByInstrRollouts"  # preprocessed train dataset
    #checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-18/18-41-35_clothes-hanger-v3.7-2cam-n50/checkpoints/100000/pretrained_model"
    #train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p7-w500h720-2cam-n50"  # preprocessed train dataset
    
    
    eval_scenarios_dataset_path = train_dataset_path  # Potentially set to train dataset to mimic initial state of certain training episode to check if policy works on "seen sample"
    #eval_scenarios_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p7-2cam-n50-EVAL"
    eval_scenarios_dataset = None
    #eval_scenarios_dataset = LeRobotDataset(repo_id="repo-id", root=eval_scenarios_dataset_path)

    eval_dataset_name = "spoofed_binary_clothes_hanger_EVAL" #"clothes-hanger-v3p8-visionAugmentedByInstrRollouts-EVAL"  # Name for dataset where to store rollout observations

    WRIST_CAM_TO_INCLUDE = "wilson"
    WRIST_CAM_TO_EXCLUDE = "sophie"

    crop_width_range = (320+70, 960-70)
    crop_width = crop_width_range[1] - crop_width_range[0]
    crop_height_range = (0, 720)
    crop_height = crop_height_range[1] - crop_height_range[0]
    crop_width = crop_width_range[1] - crop_width_range[0]
    resize = (crop_width, crop_height)
    train_crop_cutoff = (0, 0)  # 10 left and right, 0 top and bottom

    def image_preprocessor(obs_dict):
        scene_img = obs_dict["scene_image"]
        wrist_img = obs_dict[f"wrist_{WRIST_CAM_TO_INCLUDE}_image"]

        #scene_image = torch.tensor(scene_img).float() / 255.0
        #wrist_image = torch.tensor(wrist_img).float() / 255.0
        #scene_img = np.zeros(scene_img.shape)
        scene_image = torch.tensor(cv2.resize(np.array(scene_img)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_LINEAR)).float() / 255.0
        wrist_image = cv2.resize(np.array(wrist_img)[:, crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR)
        wrist_image = np.array(wrist_image)[train_crop_cutoff[0]:-train_crop_cutoff[0] if train_crop_cutoff[0] > 0 else None, train_crop_cutoff[1]:-train_crop_cutoff[1] if train_crop_cutoff[1] > 0 else None]
        wrist_image = torch.tensor(wrist_image).float() / 255.0
        
        rr.log("prepr_scene_image", rr.Image(scene_image))
        rr.log(f"prepr_wrist_{WRIST_CAM_TO_INCLUDE}_image", rr.Image(wrist_image))

        scene_image = scene_image.permute(2, 0, 1).unsqueeze(0)
        wrist_image = wrist_image.permute(2, 0, 1).unsqueeze(0)

        return {
            "observation.images.scene_image": scene_image,
            f"observation.images.wrist_{WRIST_CAM_TO_INCLUDE}_image": wrist_image,
        }

    def preprocessor(obs_dict):
        current_joints = obs_dict["joints"]
        current_gripper = np.array([obs_dict["gripper_states"][1]])
        clothes_hanger = obs_dict["clothes_hanger_spoof"]
        rr.log("clothes_hanger_spoof", rr.Scalars(clothes_hanger), rr.SeriesLines(widths=10))

        state = np.concatenate([current_joints, current_gripper, clothes_hanger]).astype(np.float32)
        state = torch.tensor(state).float().unsqueeze(0)
        prepr_dict = image_preprocessor(obs_dict)
        prepr_dict["observation.state"] = state
        return prepr_dict

    env = UR5eStation()
    env.reset()

    config = DynamixelConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],
        joint_offsets=(np.array([40, 16, 25, 40, 15, 7]) * np.pi / 16).tolist(),
        joint_signs=[1, 1, -1, 1, 1, 1],
        gripper_config=(7, 194, 152),
    )
    input("Hold Gello arm in similar pose as teleop robot and press Enter to start evaluation")
    teleop_agent = GelloAgent(config, GELLO_AGENT_PORT, start_joints=env.teleop_robot.get_joint_configuration())

    policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
    lerobot_agent = LerobotAgent(policy, "cuda", preprocessor)

    # create a dataset recorder

    '''if os.path.exists(f'datasets/{eval_dataset_name}'):
        logger.warning(f"Dataset {eval_dataset_name} already exists, removing it to start fresh.")
        os.system(f"rm -rf datasets/{eval_dataset_name}")'''

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=f"datasets/{eval_dataset_name}",
        dataset_name=eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    input("Press Enter to start evaluation (should hold your teleop in place now!)")

    rr.init("robot_imitation_glue", spawn=True)
    eval(
        env,
        teleop_agent,
        lerobot_agent,
        dataset_recorder,
        policy_to_pose_converter=policy_action_converter_nop,
        teleop_to_pose_converter=convert_gello_actions_to_joint_space_robot_pose,
        fps=10,
        eval_dataset=eval_scenarios_dataset,
        eval_dataset_image_keys=["scene_image", "wrist_" + WRIST_CAM_TO_INCLUDE + "_image"],  # images to overlay with eval episode to align initial state
        env_observation_image_keys=["scene_image", "wrist_" + WRIST_CAM_TO_INCLUDE + "_image"],  # images to monitor during evaluation
        eval_dataset_episode=4
    )
