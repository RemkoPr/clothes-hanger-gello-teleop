import numpy as np
import torch
import os
import cv2
from loguru import logger

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
    #checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-06-17/18-16-06_clothes-hanger-v1/checkpoints/100000/pretrained_model"
    #train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-prepr-cropped-w500"  # preprocessed train dataset
    #checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-06-23/14-18-43_clothes-hanger-v1-visionOnly/checkpoints/100000/pretrained_model"
    #train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-prepr-cropped-w500-visionOnly"  # preprocessed train dataset
    checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-07-17/12-34-09_clothes-hanger-v3/checkpoints/100000/pretrained_model"
    train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3-prepr-w500"  # preprocessed train dataset
    
    eval_scenarios_dataset_path = train_dataset_path  # "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-eval-scenarios"  # Potentially set to train dataset to mimic initial state of certain training episode to check if policy works on "seen sample"

    eval_dataset_name = "clothes-hanger-v3-prepr-w500-EVAL"  # Name for 'new' dataset of all rollouts for evaluation after 
    #eval_dataset_name = "tmp"

    crop_width_range = (320+25, 960-140-25)
    crop_width = crop_width_range[1] - crop_width_range[0]
    resize = (crop_width, 720)

    def preprocessor(obs_dict, VISION_ONLY=False):
        scene_img = obs_dict["scene_image"]
        wrist_img = obs_dict["wrist_image"]
        state = obs_dict["state"]

        current_joints = obs_dict["joints"]
        current_gripper = np.array([obs_dict["gripper_states"][1]])
        clothes_hanger = obs_dict["clothes_hanger"]
        if VISION_ONLY:
            state = np.concatenate([current_joints, current_gripper]).astype(np.float32)
        else:
            state = np.concatenate([current_joints, current_gripper, clothes_hanger]).astype(np.float32)
        state = torch.tensor(state).float().unsqueeze(0)

        #scene_image = torch.tensor(scene_img).float() / 255.0
        #wrist_image = torch.tensor(wrist_img).float() / 255.0
        #scene_img = np.zeros(scene_img.shape)
        scene_image = torch.tensor(cv2.resize(np.array(scene_img)[:, crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_CUBIC)).float() / 255.0
        wrist_image = torch.tensor(cv2.resize(np.array(wrist_img)[:, crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_CUBIC)).float() / 255.0
        scene_image = scene_image.permute(2, 0, 1)
        wrist_image = wrist_image.permute(2, 0, 1)

        # unsqueeze images
        scene_image = scene_image.unsqueeze(0)
        wrist_image = wrist_image.unsqueeze(0)

        return {
            "observation.images.scene_image": scene_image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": state,
        }

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

    eval_scenarios_dataset = None#LeRobotDataset(repo_id="", root=eval_scenarios_dataset_path)
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    eval(
        env,
        teleop_agent,
        lerobot_agent,
        dataset_recorder,
        policy_to_pose_converter=policy_action_converter_nop,
        teleop_to_pose_converter=convert_gello_actions_to_joint_space_robot_pose,
        fps=10,
        eval_dataset=eval_scenarios_dataset,
        eval_dataset_image_key="observation.images.scene_image",
        env_observation_image_key="scene_image",
    )
