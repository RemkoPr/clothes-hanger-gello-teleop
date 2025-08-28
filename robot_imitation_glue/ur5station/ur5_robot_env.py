"""
Environement for Robot Station with 2 realsense cameras and a UR3e robot with a Robotiq Gripper

Actions as absolute target pose (rotation vector) in robot base frame and absolute gripper width
Proprioception as robot pose (euler angles) in robot base frame and gripper width
"""

import os
import time

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.cameras.zed.zed import Zed
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_spatial_algebra.se3 import SE3Container, normalize_so3_matrix
from ur_analytic_ik import ur5e
from clothes_hanger import ClothesHanger, ClothesHangerMock
from functools import partial

from robot_imitation_glue.agents.gello import DynamixelConfig, GelloAgent
from robot_imitation_glue.base import BaseEnv
from robot_imitation_glue.grippers.schunk_process import SchunkGripperProcess
from robot_imitation_glue.ipc_camera import RGBCameraPublisher, RGBCameraSubscriber

# env consists of 1 zed scene camera, 1 wrist  realsense cameras and a UR5e robot + Schunk gripper

WRIST_SOPHIE_REALSENSE_SERIAL = "817612070315"
WRIST_SOPHIE_CAM_RGB_TOPIC = "wrist_sophie_rgb"
WRIST_SOPHIE_CAM_RESOLUTION_TOPIC = "wrist_sophie_resolution"

WRIST_WILSON_REALSENSE_SERIAL = "130322271048"
WRIST_WILSON_CAM_RGB_TOPIC = "wrist_wilson_rgb"
WRIST_WILSON_CAM_RESOLUTION_TOPIC = "wrist_wilson_resolution"

SCENE_ZED_SERIAL = "38633712"
SCENE_CAM_RGB_TOPIC = "scene_rgb"
SCENE_CAM_RESOLUTION_TOPIC = "scene_resolution"

WILSON_IP = "10.42.0.163"
SOPHIE_IP = "10.42.0.162"
SCHUNK_TCP_OFFSET = 0.184
CLOTHES_HANGER_GRASP_WIDTH = 0.02
INIT_GRASPS = False

SCHUNK_WILSON_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:5:1.0-port0,11,115200,8E1"
SCHUNK_SOPHIE_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:3:1.0-port0,14,115200,8E1"

HOLD_SHIRT_JOINTS_WILSON_HORIZONTAL = np.array([160, -130, 84, 44, 94, 87]) * np.pi / 180
HOLD_SHIRT_JOINTS_WILSON_VERTICAL = np.array([157, -92, 29, -24, -91, 83]) * np.pi / 180
HOME_JOINTS_SOPHIE = np.array([0, -150, 125, -150, -85, 0]) * np.pi / 180

GELLO_AGENT_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT792DZ5-if00-port0"
logger = loguru.logger


MAX_TRANSLATION = 0.15
MAX_JOINT_DELTA = 10 * np.pi / 180

class CameraFactory:
    def create_wrist_camera(serial_number):
        return Realsense(resolution=Realsense.RESOLUTION_720, fps=30, serial_number=serial_number)

    def create_scene_camera():
        return Zed(
            resolution=Zed.RESOLUTION_720, fps=30, depth_mode=Zed.NONE_DEPTH_MODE, serial_number=SCENE_ZED_SERIAL
        )


class UR5eStation(BaseEnv):
    # RealSense wrist camera: rotated 4 teeth
    ACTION_SPEC = None
    PROPRIO_OBS_SPEC = None

    def __init__(self):

        # set up robot and gripper
        # logger.info("connecting to gripper.")

        # set environment variable for bks gripper comm
        #os.environ["BKS_HOST"] = SCHUNK_WILSON_PORT
        self.gripper_wilson = SchunkGripperProcess(SCHUNK_WILSON_PORT)
        self.gripper_sophie = SchunkGripperProcess(SCHUNK_SOPHIE_PORT)
        time.sleep(2)
        self.gripper_wilson.max_grasp_force = self.gripper_wilson.gripper_specs.min_force  # minimal force for EGK40 is 55N
        self.gripper_wilson.speed = self.gripper_wilson.gripper_specs.max_speed
        self.gripper_sophie.max_grasp_force = self.gripper_sophie.gripper_specs.min_force  # minimal force for EGK40 is 55N
        self.gripper_sophie.speed = self.gripper_sophie.gripper_specs.max_speed
        logger.info("connecting to Wilson.")
        self.wilson = URrtde(WILSON_IP, URrtde.UR3E_CONFIG, gripper=self.gripper_wilson)
        logger.info("connecting to Sophie.")
        self.sophie = URrtde(SOPHIE_IP, URrtde.UR3E_CONFIG, gripper=self.gripper_sophie)

        self.teleop_robot = self.sophie
        self.hold_robot = self.wilson
        self.gripper_teleop = self.gripper_sophie
        self.gripper_hold = self.gripper_wilson

        self.clothes_hanger = ClothesHanger()
        init_vals = self.clothes_hanger.read()  # Test if clothes hanger can be read to catch errors early

        if INIT_GRASPS:
            self.wilson.gripper.open()
            self.sophie.gripper.open()
        wilson_awaitable = self.wilson.move_to_joint_configuration(
            HOLD_SHIRT_JOINTS_WILSON_VERTICAL
        )
        sophie_awaitable = self.sophie.move_to_joint_configuration(
            HOME_JOINTS_SOPHIE
        )  # do not wait, let cameras initialize first'''

        logger.info("Creating Wilson wrist camera publisher.")
        self._wrist_wilson_camera_publisher = RGBCameraPublisher(
            partial(CameraFactory.create_wrist_camera, serial_number=WRIST_WILSON_REALSENSE_SERIAL),
            WRIST_WILSON_CAM_RGB_TOPIC,
            WRIST_WILSON_CAM_RESOLUTION_TOPIC,
            100,
        )
        self._wrist_wilson_camera_publisher.start()

        logger.info("Creating Sophie wrist camera publisher.")
        self._wrist_sophie_camera_publisher = RGBCameraPublisher(
            partial(CameraFactory.create_wrist_camera, serial_number=WRIST_SOPHIE_REALSENSE_SERIAL),
            WRIST_SOPHIE_CAM_RGB_TOPIC,
            WRIST_SOPHIE_CAM_RESOLUTION_TOPIC,
            100,
        )
        self._wrist_sophie_camera_publisher.start()

        logger.info("Creating Wilson wrist camera subscriber.")
        self._wrist_wilson_camera_subscriber = RGBCameraSubscriber(
            WRIST_WILSON_CAM_RESOLUTION_TOPIC,
            WRIST_WILSON_CAM_RGB_TOPIC,
        )

        logger.info("Creating Sophie wrist camera subscriber.")
        self._wrist_sophie_camera_subscriber = RGBCameraSubscriber(
            WRIST_SOPHIE_CAM_RESOLUTION_TOPIC,
            WRIST_SOPHIE_CAM_RGB_TOPIC,
        )

        logger.info("Creating scene camera publisher.")
        self._scene_camera_publisher = RGBCameraPublisher(
            CameraFactory.create_scene_camera,
            SCENE_CAM_RGB_TOPIC,
            SCENE_CAM_RESOLUTION_TOPIC,
            100,
        )
        self._scene_camera_publisher.start()

        logger.info("Creating scene camera subscriber.")
        self._scene_camera_subscriber = RGBCameraSubscriber(
            SCENE_CAM_RESOLUTION_TOPIC,
            SCENE_CAM_RGB_TOPIC,
        )

        # wait for first images
        time.sleep(2)

        wilson_awaitable.wait()
        sophie_awaitable.wait()
        self.wilson_base_pose = self.wilson.get_tcp_pose()

        init_vals = self.clothes_hanger.read()
        offsets = init_vals - np.array([227, 239, 217, 211])  # Set initial offsets. TODO: make offset depend on training dataset
        self.clothes_hanger.set_offsets(offsets)
        logger.warning(f"Using clothes hanger offsets: {offsets}")
        
        if INIT_GRASPS:
            input("Grasp shirt?")
            self.gripper_hold.move(0.0, speed=2*self.gripper_hold.gripper_specs.min_speed, force=self.gripper_hold.gripper_specs.max_force).wait()
            input("Grasp clothes hanger?")
            self.gripper_teleop.move(CLOTHES_HANGER_GRASP_WIDTH, speed=2*self.gripper_teleop.gripper_specs.min_speed, force=self.gripper_teleop.gripper_specs.min_force).wait()


    def get_joint_configuration(self):
        return self.teleop_robot.get_joint_configuration()

    def get_robot_pose_euler(self):
        """
        pose as [x,y,z,rx,ry,rz] in robot base frame using Euler angles
        """
        hom_pose = self.teleop_robot.get_tcp_pose()
        rotation_vector = SE3Container.from_homogeneous_matrix(hom_pose).orientation_as_euler_angles
        position = hom_pose[:3, 3]
        return np.concatenate((position, rotation_vector), axis=0)

    def get_robot_pose_se3(self):
        return self.teleop_robot.get_tcp_pose()

    def move_robot_to_tcp_pose(self, pose):
        self.teleop_robot.move_to_tcp_pose(pose).wait()

    def move_teleop_robot_to_joint_pose(self, joint_config):
        self.teleop_robot.move_to_joint_configuration(joint_config).wait()

    def move_teleop_robot_to_home_pose(self):
        self.teleop_robot.move_to_joint_configuration(HOME_JOINTS_SOPHIE).wait()

    def move_gripper(self, width):
        self.gripper_teleop.move(width).wait()

    def get_gripper_openings(self):
        return np.array([self.gripper_teleop.get_current_width(), self.gripper_hold.get_current_width()])

    def _set_robot_target_pose(self, target_pose):
        # target_pose is a 4x4 homogeneous transformation matrix
        self.wilson.servo_to_tcp_pose()

    def get_observations(self):

        start_time = time.time()
        wrist_wilson_image = self._wrist_wilson_camera_subscriber.get_rgb_image_as_int()
        wrist_sophie_image = self._wrist_sophie_camera_subscriber.get_rgb_image_as_int()
        scene_image = self._scene_camera_subscriber.get_rgb_image_as_int()
        robot_state = self.get_joint_configuration().astype(np.float32)  # set to joint configuration
        gripper_states = self.get_gripper_openings().astype(np.float32)
        joints = self.teleop_robot.get_joint_configuration().astype(np.float32)
        clothes_hanger_values = self.clothes_hanger.read().astype(np.float32)

        #state = np.concatenate((robot_state, gripper_states), axis=0)
        state = np.concatenate((robot_state, gripper_states, clothes_hanger_values), axis=0)  # not directly used, state is instead formed in ur5station/prepare_datasets

        # resize images

        wrist_wilson_image_resized = cv2.resize(wrist_wilson_image, (1280, 720), interpolation=cv2.INTER_CUBIC)
        wrist_sophie_image_resized = cv2.resize(wrist_sophie_image, (1280, 720), interpolation=cv2.INTER_CUBIC)
        scene_image_resized = cv2.resize(scene_image, (1280, 720), interpolation=cv2.INTER_CUBIC)

        obs_dict = {
            "wrist_sophie_image_original": wrist_sophie_image,
            "wrist_wilson_image_original": wrist_wilson_image,
            "scene_image_original": scene_image,
            "wrist_wilson_image": wrist_wilson_image_resized,
            "wrist_sophie_image": wrist_sophie_image_resized,
            "scene_image": scene_image_resized,
            "state": state,
            "robot_pose": robot_state,
            "gripper_states": gripper_states,
            "joints": joints,
            "clothes_hanger": clothes_hanger_values,
        }
        logger.info(f"get_observations time: {time.time() - start_time}")

        # add to rerun
        # rr.log("wrist",rr.Image(wrist_image))
        # rr.log("scene",rr.Image(scene_image))

        return obs_dict

    def act(self, robot_pose, gripper_pose, timestamp, disable_gripper=False, control_space="JOINT"):

        if isinstance(gripper_pose, np.ndarray):
            gripper_pose = gripper_pose[0].item()

        # move robot to target pose
        current_time = time.time()
        duration = timestamp - current_time
        if duration < 0:
            logger.warning("Action duration is negative, setting it to 0")
            duration = 0
        logger.debug(f"Moving robot to pose \n {robot_pose} with duration {duration}")

        if control_space == "TOOL":
            robot_pose_se3 = robot_pose.copy()
            robot_pose_se3[:3, :3] = normalize_so3_matrix(robot_pose_se3[:3, :3])

            y_coord = robot_pose_se3[1, 3]
            z_coord = robot_pose_se3[2, 3]

            if z_coord < 0.0:
                logger.warning("Z coordinate is below zero, not executing action.")
                return
            if y_coord < -40.0:
                logger.warning("Y coordinate is beyond camera safety plane, not executing action.")
                return

            valid_pose = True
            if np.linalg.norm(robot_pose_se3[:3, 3] - self.teleop_robot.get_tcp_pose()[:3, 3]) > MAX_TRANSLATION:
                logger.warning("TCP pose is too far from current pose, clipping translation.")
                # clip the translation.
                direction = robot_pose_se3[:3, 3] - self.teleop_robot.get_tcp_pose()[:3, 3]
                direction = direction / np.linalg.norm(direction)
                robot_pose_se3[:3, 3] = self.teleop_robot.get_tcp_pose()[:3, 3] + 0.5 * MAX_TRANSLATION * direction
                valid_pose = True

            if robot_pose_se3[2, 3] < 0.0:
                logger.warning("Z coordinate is below zero . not executing action")
                valid_pose = False

            if not self.teleop_robot.is_tcp_pose_reachable(robot_pose_se3):
                logger.warning("TCP pose is not reachable, not executing action")
                valid_pose = False
            if valid_pose:
                self.teleop_robot.servo_to_tcp_pose(robot_pose_se3, duration)
        elif control_space == "JOINT":
            robot_pose_se3 = ur5e.forward_kinematics_with_tcp(*robot_pose[:6], np.eye(4))

            y_coord = robot_pose_se3[1, 3]
            z_coord = robot_pose_se3[2, 3]

            if z_coord < 0.0:
                logger.warning("Z coordinate is below zero, not executing action.")
                return
            if y_coord < -40.0:
                logger.warning("Y coordinate is beyond camera safety plane, not executing action.")
                return

            valid_pose = True
            joint_diff = abs(robot_pose[:6] - self.teleop_robot.get_joint_configuration())
            if np.max(joint_diff) > MAX_JOINT_DELTA:
                logger.warning(f"Joints {joint_diff > MAX_JOINT_DELTA} move too far from current pose, clipping rotation.")
                robot_pose[:6] = self.teleop_robot.get_joint_configuration() + np.clip(
                    robot_pose[:6] - self.teleop_robot.get_joint_configuration(),
                    -MAX_JOINT_DELTA,
                    MAX_JOINT_DELTA,
                )
                valid_pose = True

            if robot_pose_se3[2, 3] < 0.0:
                logger.warning("Z coordinate is below zero . not executing action")
                valid_pose = False

            '''if not self.teleop_robot.is_tcp_pose_reachable(robot_pose_se3):
                logger.warning("TCP pose is not reachable, not executing action")
                valid_pose = False'''
            if valid_pose:
                self.teleop_robot.servo_to_joint_configuration(robot_pose, duration)
        else:
            raise ValueError(f"Unknown control space: {control_space}")


        # move gripper to target width
        if not disable_gripper:
            gripper_width = self.gripper_hold.gripper_specs.max_width - np.clip(
                gripper_pose, self.gripper_hold.gripper_specs.min_width, self.gripper_hold.gripper_specs.max_width
            )
            if gripper_width < self.gripper_hold.gripper_specs.min_width + 0.005:
                gripper_width = self.gripper_hold.gripper_specs.min_width
            logger.debug(f"Setting gripper width to {gripper_width}")
            time_before_gripper = time.time()
            self.gripper_hold.servo(gripper_width)
            time_after_gripper = time.time()
            logger.debug(f"Gripper servo time: {time_after_gripper - time_before_gripper}")

        # do not wait, handling timings is the responsibility of the caller
        return

    def toggle_holding_gripper(self):
        logger.info("Toggling holding gripper.")
        if self.gripper_hold.get_current_width() > 0.04:
            logger.info("Closing holding gripper.")
            self.gripper_hold.close().wait()
        else:
            logger.info("Opening holding gripper.")
            self.gripper_hold.open().wait()

    def move_hold_robot_random_translation(self, max_translation=0.05):
        translation = np.random.uniform(-max_translation, max_translation, size=3)
        translation[2] = 0.0
        new_pose = self.wilson_base_pose.copy()
        new_pose[:3, 3] += translation
        self.hold_robot.move_linear_to_tcp_pose(new_pose).wait()

    def close(self):
        self._wrist_camera_publisher.stop()
        self._scene_camera_publisher.stop()
        self.gripper_wilson.shutdown()


def convert_abs_gello_actions_to_se3(current_pose, current_gripper_state, action: np.ndarray):
    del current_pose, current_gripper_state
    tcp_pose = np.eye(4)
    tcp_pose[2, 3] = SCHUNK_TCP_OFFSET
    joints = action[:6]
    gripper = action[6]
    gripper = (1 - gripper) * 0.08  # convert to stroke width
    pose = ur5e.forward_kinematics_with_tcp(*joints, tcp_pose)
    return pose, gripper


def convert_gello_actions_to_joint_space_robot_pose(current_pose, current_gripper_state, action: np.ndarray):
    del current_pose, current_gripper_state
    joints = action[:6]
    gripper = action[6]
    gripper = (1 - gripper) * 0.08  # convert to stroke width
    return joints, gripper


def abs_joint_policy_action_to_se3(current_pose, current_gripper_state, action: np.ndarray):
    del current_pose, current_gripper_state
    joints = action[:6]
    gripper = action[6]
    tcp_pose = np.eye(4)
    tcp_pose[2, 3] = SCHUNK_TCP_OFFSET
    pose = ur5e.forward_kinematics_with_tcp(*joints, tcp_pose)
    return pose, gripper

def abs_joint_policy_action_to_joint_pose(current_pose, current_gripper_state, action: np.ndarray):
    del current_pose, current_gripper_state
    joints = action[:6]
    gripper = action[6]
    return joints, gripper


'''dynamixel_config = DynamixelConfig(
    joint_ids=[1, 2, 3, 4, 5, 6],
    joint_offsets=(np.array([40, 16, 25, 40, 15, 7]) * np.pi / 16).tolist(),
    joint_signs=[1, 1, -1, 1, 1, 1],
    gripper_config=(7, 194, 152),
)'''

if __name__ == "__main__":
    # set cli logging level to debug

    env = UR5eStation()

    agent = GelloAgent(dynamixel_config, GELLO_AGENT_PORT)

    cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
    cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wrist", 640, 480)
    cv2.resizeWindow("scene", 640, 480)

    input("Press Enter to start teleoperation")
    action = agent.get_action(env.get_observations())
    robot_se3, gripper = convert_abs_gello_actions_to_se3(action)
    env.wilson.servo_to_tcp_pose(robot_se3, 1.0)

    while True:
        loop_time = time.time()
        obs = env.get_observations()
        print(time.time())
        print(obs["wrist_image"].shape)
        cv2.imshow("wrist", obs["wrist_image"])
        cv2.imshow("scene", obs["scene_image"])

        action = agent.get_action(obs)
        robot_se3, gripper = convert_abs_gello_actions_to_se3(action)
        env.act(robot_pose_se3=robot_se3, gripper_pose=gripper, timestamp=time.time() + 0.1)
        print(obs["state"])

        loop_duration = time.time() - loop_time
        # wait for 100ms - loop time
        key = cv2.waitKey(max(1, int(100 - loop_duration * 1000)))

    env.robot.gripper.move(0.04).wait()
    time.sleep(5)
    env.robot.gripper.move(0.0).wait()
