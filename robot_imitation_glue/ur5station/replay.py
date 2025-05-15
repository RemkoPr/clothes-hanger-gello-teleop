from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.lerobot_dataset.replay_episode import replay_episode
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation, abs_joint_policy_action_to_joint_pose

if __name__ == "__main__":
    try:
        env = UR5eStation()
        dataset = LeRobotDataset(repo_id="", root="datasets/2025-05-15_19-54-25")
        replay_episode(
            env, dataset, abs_joint_policy_action_to_joint_pose, image_key="scene_image", dataset_image_key="observation.images.scene_image", episode_idx=0
        )

    finally:
        env.close()
