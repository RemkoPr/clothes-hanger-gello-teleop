"""Gello teleop agent"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np

from robot_imitation_glue.agents.gello.dynamixel_robot import DynamixelRobot
from robot_imitation_glue.base import BaseAgent


@dataclass
class DynamixelConfig:
    joint_ids: List[int]
    joint_offsets: List[float]
    joint_signs: List[int]
    gripper_config: Optional[Tuple[int, float, float]] = None

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets) == len(self.joint_signs)


class GelloAgent(BaseAgent):
    """
    takes in a Dynamixel config for a Gello teleop arm and creates the teleop agent
    """

    def __init__(self, dynamixel_config: DynamixelConfig, port: str, start_joints: List[float] = None):
        self.robot = DynamixelRobot(
            joint_ids=dynamixel_config.joint_ids,
            joint_offsets=dynamixel_config.joint_offsets,
            joint_signs=dynamixel_config.joint_signs,
            gripper_config=dynamixel_config.gripper_config,
            port=port,
            start_joints=start_joints,
            real=True,
        )
        init_joints = self.robot.get_joint_state()
        self.joint_offsets = np.array([0.0 for _ in range(init_joints.size)])
        for idx, joint in enumerate(init_joints):
            while joint > 2*np.pi:
                joint -= 2*np.pi
                self.joint_offsets[idx] -= 2*np.pi
            while joint < -2*np.pi:
                joint += 2*np.pi
                self.joint_offsets[idx] += 2*np.pi
        logger.info(f"Gello initialised with joint offsets: {self.joint_offsets}")


    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get the action from the agent
        """

        return self.robot.get_joint_state() + self.joint_offsets


if __name__ == "__main__":
    import time

    import numpy as np

    config = DynamixelConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],
        joint_offsets=(np.array([0, 4, -6, 2, 0, 6]) * np.pi / 4).tolist(),
        joint_signs=[1, 1, -1, 1, 1, 1],
        gripper_config=(7, 194, 152),
    )
    agent = GelloAgent(config, "/dev/ttyUSB1")
    while True:
        action = agent.get_action({})
        print(action)
        time.sleep(0.1)
