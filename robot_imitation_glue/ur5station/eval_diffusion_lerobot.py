from functools import partial
import numpy as np
import torch
import os
import cv2
from loguru import logger
import rerun as rr

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation


def obs_preprocessor(obs_dict, include_instr=True, crop_width_range=(320, 960),
                     crop_height_range=(0, 720)):
    crop_height = crop_height_range[1] - crop_height_range[0]
    crop_width = crop_width_range[1] - crop_width_range[0]
    resize = (2*crop_width//3, 2*crop_height//3)
    scene_img = obs_dict["scene_image"]
    wrist_wilson_img = obs_dict["wrist_wilson_image"]

    current_joints = obs_dict["teleop_robot_joints"]
    current_gripper = obs_dict["gripper_on_static_robot"]
    if hasattr(current_gripper, "shape") and len(current_gripper.shape) == 0:
        current_gripper = current_gripper.unsqueeze(0)
    clothes_hanger = obs_dict["clothes_hanger"]
    if not include_instr:
        state = np.concatenate([current_joints, current_gripper]).astype(np.float32)
    else:
        state = np.concatenate([current_joints, current_gripper, clothes_hanger]).astype(np.float32)
    state = torch.tensor(state).float().unsqueeze(0)
    
    scene_image = cv2.resize(np.array(scene_img)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_LINEAR)
    scene_image = torch.tensor(scene_image).float() / 255.0
    wrist_wilson_image = cv2.resize(np.flip(np.array(wrist_wilson_img), axis=0)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR)
    wrist_wilson_image = torch.tensor(wrist_wilson_image).float() / 255.0
    
    rr.log("obs_prepr: scene_image", rr.Image(scene_image))
    rr.log("obs_prepr: wrist_wilson_image", rr.Image(wrist_wilson_image))

    scene_image = scene_image.permute(2, 0, 1).unsqueeze(0)
    wrist_wilson_image = wrist_wilson_image.permute(2, 0, 1).unsqueeze(0)

    return {
        "observation.images.scene_image": scene_image,
        "observation.images.wrist_wilson_image": wrist_wilson_image,
        "observation.state": state,
    }


if __name__ == "__main__":
    logger.level("INFO")

    INCLUDE_INSTR = True
    MODEL_NAME = f"b-n200-INSTR{1 if INCLUDE_INSTR else 0}-300k-1enc"
    checkpoint_path = f"/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/{MODEL_NAME}"
    train_dataset_path = f"/storage/rproesma/clothes-hanger/datasets/b-n200-PREPR-INSTR{1 if INCLUDE_INSTR else 0}"  # preprocessed train dataset
    
    eval_scenarios_dataset_path = train_dataset_path  # "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-eval-scenarios"  # Potentially set to train dataset to mimic initial state of certain training episode to check if policy works on "seen sample"

    eval_dataset_name = f"{MODEL_NAME}-EVAL"  # Name for 'new' dataset of all rollouts for evaluation after 
    #eval_dataset_name = "tmp"

    
    env = UR5eStation(mode="EVAL")
    env.reset()

    policy, preprocessor, postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
    lerobot_agent = LerobotAgent(policy, preprocessor, postprocessor, "cuda", observation_preprocessor=partial(obs_preprocessor, include_instr=INCLUDE_INSTR))

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
        lerobot_agent,
        dataset_recorder,
        fps=10,
        eval_dataset=eval_scenarios_dataset,
        eval_dataset_image_key="observation.images.scene_image",
        env_observation_image_key="scene_image",
    )
