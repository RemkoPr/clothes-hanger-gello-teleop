import os
import time
from robot_imitation_glue.grippers.schunk_process import SchunkGripperProcess
from airo_robots.manipulators.hardware.ur_rtde import URrtde


SCHUNK_GRIPPER_HOST = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:5:1.0-port0,11,115200,8E1"
#SCHUNK_GRIPPER_HOST = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:8:1.0-port0,14,115200,8E1"

gripper2 = SchunkGripperProcess(SCHUNK_GRIPPER_HOST)
time.sleep(2)
gripper2.max_grasp_force = gripper2.gripper_specs.min_force  # minimal force for EGK40 is 55N
gripper2.speed = gripper2.gripper_specs.max_speed

gripper2.move(0.08).wait()

time.sleep(3)

gripper2.move(0.0).wait()

