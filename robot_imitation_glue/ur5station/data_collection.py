from datetime import datetime
from pathlib import Path

from loguru import logger
import numpy as np

from robot_imitation_glue.collect_data import collect_data
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation


def abs_se3_to_policy_action_converter(robot_pose, gripper_pose, abs_se3_action, gripper_action):
    """absolute poses, encoded as position, x-vector of rotation, y-vector of rotation, gripper action"""
    abs_position_target = abs_se3_action[:3, 3]
    abs_rotation_x = abs_se3_action[:3, 0]
    abs_rotation_y = abs_se3_action[:3, 1]

    policy_action = np.zeros(10)
    policy_action[:3] = abs_position_target
    policy_action[3:6] = abs_rotation_x
    policy_action[6:9] = abs_rotation_y
    policy_action[9] = gripper_action
    policy_action = policy_action.astype(np.float32)
    return policy_action


def abs_joints_to_policy_action_converter(robot_pose, gripper_pose, abs_joint_action, gripper_action):
    policy_action = np.zeros(7)
    policy_action[:6] = abs_joint_action
    policy_action[6] = gripper_action
    policy_action = policy_action.astype(np.float32)
    return policy_action

def policy_action_converter_nop(robot_pose, gripper_pose, policy_action):
    return policy_action[:6], policy_action[6]

def policy_action_to_abs_se3_converter(robot_pose, gripper_pose, policy_action):
    policy_action = policy_action.astype(np.float64)
    abs_position_target = policy_action[:3]
    rot6D = policy_action[3:9]
    gripper_action = policy_action[9]

    # convert to se3
    x = rot6D[:3] / np.linalg.norm(rot6D[:3])
    y = rot6D[3:] - np.dot(rot6D[3:], x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)

    target_pose = np.eye(4)
    target_pose[:3, 3] = abs_position_target
    target_pose[:3, 0] = x
    target_pose[:3, 1] = y
    target_pose[:3, 2] = z
    return target_pose, gripper_action

def policy_action_to_abs_joint_converter(robot_pose, gripper_pose, policy_action):
    policy_action = policy_action.astype(np.float64)
    target_joint_pose = policy_action[:6]
    gripper_action = policy_action[6]
    return target_joint_pose, gripper_action


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true")
    #parser.add_argument("--dataset_name", type=str, default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    parser.add_argument("--dataset_name", type=str, default=f"b")
    args = parser.parse_args()

    env = UR5eStation()

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros(7),
        root_dataset_dir=Path(f"datasets/{args.dataset_name}/"),
        dataset_name="none",
        fps=10,
    )

    collect_data(
        env,
        dataset_recorder,
        frequency=10,
    )
