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


def img_preprocessor(img, img_type, crop_width_range=(320, 960),
                     crop_height_range=(0, 720)):
    try:
        crop_height = crop_height_range[1] - crop_height_range[0]
        crop_width = crop_width_range[1] - crop_width_range[0]
        resize = (2*crop_width//3, 2*crop_height//3)
        if img_type == "scene_image":
            processed_img = cv2.resize(np.array(img)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_LINEAR)
        elif img_type == "wrist_wilson_image":
            processed_img = cv2.resize(np.flip(np.array(img), axis=0)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unknown image type: {img_type}")
        processed_img = torch.tensor(processed_img).float() / 255.0
        rr.log(f"obs_prepr: {img_type}", rr.Image(processed_img))
        return processed_img.permute(2, 0, 1).unsqueeze(0)
    
    except Exception as e:
        logger.error(f"Error processing image {img}, {img.shape}: {e}")
        raise e
    


def obs_preprocessor(obs_dict, include_instr=True, spoof=False, crop_width_range=(320, 960),
                     crop_height_range=(0, 720)):
    scene_img = obs_dict["scene_image"]
    wrist_wilson_img = obs_dict["wrist_wilson_image"]

    current_joints = obs_dict["teleop_robot_joints"]
    current_gripper = obs_dict["gripper_on_static_robot"]
    if hasattr(current_gripper, "shape") and len(current_gripper.shape) == 0:
        current_gripper = torch.tensor(current_gripper).float().unsqueeze(0)
    if not include_instr:
        state = np.concatenate([current_joints, current_gripper]).astype(np.float32)
    elif not spoof:
        clothes_hanger = obs_dict["clothes_hanger"]
        state = np.concatenate([current_joints, current_gripper, clothes_hanger]).astype(np.float32)
    else:
        clothes_hanger_spoof = obs_dict["clothes_hanger_spoof"]
        state = np.concatenate([current_joints, current_gripper, clothes_hanger_spoof]).astype(np.float32)
    state = torch.tensor(state).float().unsqueeze(0)
    
    scene_image = img_preprocessor(scene_img, "scene_image", crop_width_range, crop_height_range)
    wrist_wilson_image = img_preprocessor(wrist_wilson_img, "wrist_wilson_image", crop_width_range, crop_height_range)
    
    return {
        "observation.images.scene_image": scene_image,
        "observation.images.wrist_wilson_image": wrist_wilson_image,
        "observation.state": state,
    }


if __name__ == "__main__":
    logger.level("INFO")

    INCLUDE_INSTR = False
    INSTR_AS_ACTION = False
    PRETR = True
    SPOOF = False
    GENERALISE = True
    assert(not (INSTR_AS_ACTION and INCLUDE_INSTR))
    assert(not (INSTR_AS_ACTION and PRETR))
    assert(not (PRETR and INCLUDE_INSTR))
    assert(not (SPOOF and not INCLUDE_INSTR))
    MODEL_NAME = f"b-n200-INSTR{1 if INCLUDE_INSTR else 0}{'ACT' if INSTR_AS_ACTION else ''}{'PRETR' if PRETR else ''}-300k-1enc"
    checkpoint_path = f"/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/{MODEL_NAME}"
    train_dataset_path = f"/storage/rproesma/clothes-hanger/datasets/b-n200-PREPR-INSTR{1 if INCLUDE_INSTR else 0}{'ACT' if INSTR_AS_ACTION else ''}"  # preprocessed train dataset
    
    eval_scenarios_dataset_path = train_dataset_path  # "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-eval-scenarios"  # Potentially set to train dataset to mimic initial state of certain training episode to check if policy works on "seen sample"

    eval_dataset_name = f"{MODEL_NAME}{'SPOOF' if SPOOF else ''}-EVAL{'GEN' if GENERALISE else ''}3"  # Name for 'new' dataset of all rollouts for evaluation after 
    #eval_dataset_name = "tmp2"

    
    env = UR5eStation(mode="EVAL", ch="SPOOF" if SPOOF else "REAL")
    env.reset()

    policy, preprocessor, postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
    lerobot_agent = LerobotAgent(policy, preprocessor, postprocessor, "cuda", observation_preprocessor=partial(obs_preprocessor, include_instr=INCLUDE_INSTR, spoof=SPOOF))

    # create a dataset recorder

    '''if os.path.exists(f'datasets/{eval_dataset_name}'):
        logger.warning(f"Dataset {eval_dataset_name} already exists, removing it to start fresh.")
        os.system(f"rm -rf datasets/{eval_dataset_name}")'''

    action_dim = 11 if INSTR_AS_ACTION else 7
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((action_dim,), dtype=np.float32),
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
        instr_as_action=INSTR_AS_ACTION
    )
