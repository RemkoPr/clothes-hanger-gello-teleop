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
    #checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-13/18-32-38_clothes-hanger-v3.5-2cam/checkpoints/100000/pretrained_model"
    #train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p5-w500h720-2cam"  # preprocessed train dataset
    #checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-15/15-09-52_clothes-hanger-v3.6-2cam-visionOnly/checkpoints/100000/pretrained_model"
    #train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p6-w500h720-2cam-visionOnly"  # preprocessed train dataset
    checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2025-08-28/18-40-57_clothes-hanger-v3.8-visionAugmentedByInstrRollouts/checkpoints/100000/pretrained_model"
    train_dataset_path = "/storage/rproesma/clothes-hanger/datasets/clothes-hanger-v3p8-visionAugmentedByInstrRollouts"  # preprocessed train dataset
    
    
    mock_observations = {
        "observation.images.scene_image": (torch.tensor(np.random.randint(0, 255, (3, 500, 720), dtype=np.uint8)).float() / 255.0).unsqueeze(0),
        "observation.images.wrist_wilson_image": (torch.tensor(np.random.randint(0, 255, (3, 500, 720), dtype=np.uint8)).float() / 255.0).unsqueeze(0),
        "observation.state": torch.tensor(np.random.randint(7, dtype=np.uint8)).float().unsqueeze(0)
    }

    print(np.array(mock_observations["observation.images.scene_image"]).shape)

    def preprocessor(obs_dict):
        return obs_dict
    
    policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
    lerobot_agent = LerobotAgent(policy, "cuda", preprocessor)

    inference_times = []
    for i in range(1000):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        action = lerobot_agent.get_action(mock_observations)
        end_time.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        inference_times.append(start_time.elapsed_time(end_time))

    print(f'Average inference time over 1000 runs: {np.mean(inference_times):.2f} ms +/- {np.std(inference_times):.2f} ms')
