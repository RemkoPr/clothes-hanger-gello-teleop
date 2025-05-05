import os
import numpy as np
from robot_imitation_glue.grippers.schunk_process import SchunkGripperProcess
from airo_robots.manipulators.hardware.ur_rtde import URrtde



SOPHIE_IP = "10.42.0.162"
WILSON_IP = "10.42.0.163"
HOME_JOINTS_WILSON = np.array([0, -60, -90, -210, 0, 0]) * np.pi / 180


sophie = URrtde(WILSON_IP, URrtde.UR3E_CONFIG, gripper=None)

sophie.move_to_joint_configuration(HOME_JOINTS_WILSON).wait()