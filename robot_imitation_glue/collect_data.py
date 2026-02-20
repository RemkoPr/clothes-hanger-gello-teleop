import time

# create type for callable that takes obs and returns action
from typing import Callable

import cv2
import loguru
import numpy as np
import rerun as rr

from robot_imitation_glue.base import BaseAgent, BaseEnv
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.utils import precise_wait
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation, convert_gello_actions_to_joint_space_robot_pose
from robot_imitation_glue.ur5station.shirt_initialiser import ShirtInitialiser

converter_callable = Callable[dict[str, np.ndarray], np.ndarray]

logger = loguru.logger


class State:
    initialising = False
    is_recording = False
    is_stopped = False
    is_paused = False


class Event:
    start_init = False
    start_recording = False
    stop_recording = False
    delete_last = False
    pause = False
    resume = False
    quit = False
    do_toggle_gripper = False
    do_reset = False
    do_randomise_hold_pose = False
    cancel_recording = False

    def clear(self):
        for attr in self.__dict__:
            setattr(self, attr, False)


def init_keyboard_listener(event: Event, state: State):
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            #if hasattr(key, "char"):
            #    print(f"Key pressed: {key.char}")
            # "space bar"
            if key == keyboard.Key.enter and not state.initialising and not state.is_recording:
                event.start_init = True

            elif key == keyboard.Key.enter and state.initialising and not state.is_recording:
                event.start_recording = True

            elif key == keyboard.Key.enter and state.is_recording:
                event.stop_recording = True

            elif hasattr(key, "char") and key.char == "p" and not state.is_recording and not state.is_paused:
                # pause the episode
                event.pause = True

            elif hasattr(key, "char") and key.char == "p" and state.is_paused:
                # resume the episode
                event.resume = True

            elif hasattr(key, "char") and key.char == "c" and state.is_recording:
                event.cancel_recording = True

            elif hasattr(key, "char") and key.char == "g" and state.is_paused:
                # open the gripper that should hold the T-shirt
                event.do_toggle_gripper = True

            elif hasattr(key, "char") and key.char == "r" and state.is_paused:
                # move robot to initial pose
                event.do_reset = True

            elif hasattr(key, "char") and key.char == "m" and state.is_paused:
                event.do_randomise_hold_pose = True

            elif hasattr(key, "char") and key.char == "q":
                event.quit = True

            elif hasattr(key, "char") and key.char == "d" and not state.is_recording:
                # delete the last episode
                event.delete_last = True
        except Exception as e:
            logger.error(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener


def collect_data(  # noqa: C901
    env: UR5eStation,
    dataset_recorder: LeRobotDatasetRecorder,
    frequency=10,
    teleop_to_pose_converter: converter_callable = None,
    abs_pose_to_policy_action: converter_callable = None,
):

    # rr.init("robot_imitation_glue", spawn=True)  # if port 9876 is free. else, run in terminal `rerun viewer --port 9877` and uncomment next two lines    rr.spawn(port=9877)
    
    #rr.init("robot_imitation_glue", spawn=False)
    #rr.connect_grpc("rerun+http://127.0.0.1:9877/proxy")

    rr.init("robot_imitation_glue2")
    rr.spawn(port=9875, memory_limit="20%", connect=True)

    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)
    shirt_initialiser = ShirtInitialiser()

    # move robot to initial teleop pose
    env.teleop_robot.move_to_joint_configuration(env.teleop_agent.get_action()[:6], joint_speed=0.1).wait()

    control_period = 1 / frequency
    while not state.is_stopped:
        try:
            cycle_end_time = time.time() + control_period

            #before_observation_time = time.time()
            #observation = env.get_observations()
            #after_observation_time = time.time()
            #observation_time = after_observation_time - before_observation_time
            #print("observation time: ", observation_time)

            # update & handle state machine events
            if not state.initialising and event.start_init:
                logger.info("======================= Initialising demo")
                state.initialising = True
                #env.move_hold_robot_random_translation()
                shirt_initialiser.init_random_line()
                env.move_teleop_robot_to_home_pose(joint_speed=0.2)
                observation = env.get_observations()
                dataset_recorder.start_episode()
                dataset_recorder.record_step(observation, np.array([0]*7).astype(np.float32))
                if not state.is_paused:
                    env.teleop_robot.move_to_joint_configuration(env.teleop_agent.get_action()[:6], joint_speed=0.2).wait()

            elif state.initialising and not state.is_recording and event.start_recording:
                logger.info("======================= Start recording (start_recording event received and already got ch baseline)")
                state.is_recording = True
                state.initialising = False
                if state.is_paused:
                    env.teleop_robot.move_to_joint_configuration(env.teleop_agent.get_action()[:6], joint_speed=0.2).wait()
                state.is_paused = False

            elif state.is_recording and event.stop_recording:
                logger.info("======================= Stop recording")
                state.is_recording = False
                # save episode
                dataset_recorder.save_episode()
                dataset_recorder.finish_recording()
                dataset_recorder.reinitialize_dataset()
                
                state.is_paused = True

            elif (state.is_recording or state.initialising) and event.cancel_recording:
                logger.info("======================= Cancel and stop recording")
                state.is_recording = False
                state.initialising = False
                time.sleep(0.5)  # avoid erasing images that are currently being written
                dataset_recorder.clear_episode()

            elif event.delete_last and not state.is_recording:
                logger.info("======================= Delete last episode")
                raise NotImplementedError("delete last episode not implemented")

            elif event.pause and not state.is_recording:
                state.is_paused = True
                logger.info("======================= Pause teleop")

            elif event.resume and state.is_paused:
                state.is_paused = False
                logger.info("======================= Resume teleop, first move slowly to current teleop pose")
                action = env.teleop_agent.get_action()
                logger.debug(f"Action: {action}")
                env.teleop_robot.move_to_joint_configuration(action[:6]).wait()
                logger.info("======================= Resuming teleop.")

            elif event.do_toggle_gripper and state.is_paused:
                logger.info("======================= Toggling gripper")
                env.toggle_holding_gripper()

            elif event.do_randomise_hold_pose and state.is_paused:
                logger.info("======================= Randomising T-shirt hold pose")
                env.move_hold_robot_random_translation()

            elif event.do_reset and state.is_paused:
                logger.info("======================= Resetting robot to initial pose")
                # move robot to initial pose
                env.move_teleop_robot_to_home_pose(joint_speed=0.2)

            elif event.quit:
                logger.info("Quitting...")
                state.is_stopped = True
                state.initialising = False
                state.is_recording = False
                listener.stop()
                time.sleep(0.5)  # avoid erasing images that are currently being written
                dataset_recorder.clear_episode()  # if quit during recording, clear unfinished episode
                dataset_recorder.finish_recording()
                logger.info("Finished dataset recording and quit data collection.")
                return

            # clear all events
            event.clear()

            observation = env.get_observations()
            # update GUI.
            vis_img = observation["scene_image"].copy()
            wrist_img = observation["wrist_wilson_image"].copy()

            if state.initialising:
                shirt_initialiser.annotate_image(wrist_img)
                cv2.putText(vis_img, "INTIALISING", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
            if state.is_recording:
                cv2.putText(vis_img, "RECORDING", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if state.is_paused:
                cv2.putText(vis_img, "PAUSED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(
                vis_img,
                f" # episodes: {dataset_recorder.n_recorded_episodes}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            rr.log("image", rr.Image(vis_img, rr.ColorModel.RGB))
            #rr.log("wrist_sophie_image", rr.Image(observation["wrist_sophie_image"], rr.ColorModel.RGB))
            rr.log("Init instruction", rr.Image(wrist_img, rr.ColorModel.RGB))
            rr.log("obs: wrist_wilson_image", rr.Image(observation["wrist_wilson_image"], rr.ColorModel.RGB))
            rr.log("obs: scene_image", rr.Image(observation["scene_image"], rr.ColorModel.RGB))

            # if paused, do not collect teleop or execute action
            if state.is_paused:
                time.sleep(0.1)
                continue

            action = env.teleop_agent.get_action()
            observation["original_gripper_action"] = action[6]
            action[6] = env.gripper_on_static_robot.gripper_specs.max_width if action[6] > env.gripper_on_static_robot.gripper_specs.max_width - 0.004 else env.gripper_on_static_robot.gripper_specs.min_width
            logger.info(f"Action: {action}")

            # store the actions in absolute format, to facilitate any action conversion later on.
            # observation["target_abs_robot_se3e_pose"] = new_robot_target_se3_pose
            # observation["target_abs_gripper_pose"] = new_gripper_target_width

            env.act(
                robot_pose=action[:6],
                gripper_pose=action[6],
                timestamp=time.time() + control_period,
                disable_gripper=False
            )

            if state.is_recording:
                dataset_recorder.record_step(observation, action.astype(np.float32))

            # wait for end of the control period
            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)
            else:
                logger.warning("cycle time exceeded control period")


            # TODO: we now use 'integration' to get the next target pose instead of using the current pose.
            # this is to avoid 'shaking' of the robot, as is done in diffusion policy teleop for example.
            # but need to verify that this does not cause mismatch between teleop and policy.
            # and should also check if the distance between the target and the actual robot does not diverge too much.
        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"An error occurred, clearing current episode and finishing recording. \nOriginal error: {e}")
            listener.stop()
            time.sleep(0.5)  # avoid erasing images that are currently being written
            dataset_recorder.clear_episode()
            dataset_recorder.finish_recording()
            logger.info("Finished dataset recording and quit data collection.")
            return


if __name__ == "__main__":
    # create dummy env, agent and recorder to test flow.
    import os
    from pathlib import Path

    from scipy.spatial.transform import Rotation as R

    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.robot_env import UR3eStation
    from robot_imitation_glue.spacemouse_agent import SpaceMouseAgent

    env = UR5eStation()

    dataset_name = "test_dataset"

    def delta_action_to_abs_se3_converter(robot_pose_se3, gripper_state, action):
        # convert spacemouse action to ur3e action
        # we take the action to consist of a delta position, delta rotation and delta gripper width.
        # the delta rotation is interpreted as expressed in a frame with the same orientation as the base frame but with the origin at the EEF.
        # in this way, when rotating the spacemouse, the robot eef will not move around, while at the same time the axes of orientation
        # do not depend on the current orientation of the EEF.

        # the delta position is intepreted in the world frame and also applied on the EEF frame.

        delta_pos = action[:3]
        delta_rot = action[3:6]
        gripper_action = action[6]

        robot_trans = robot_pose_se3[:3, 3]
        robot_SO3 = robot_pose_se3[:3, :3]

        new_robot_trans = robot_trans + delta_pos
        # rotation is now interpreted as euler and not as rotvec
        # similar to Diffusion Policy.
        # however, rotvec seems more principled (related to twist)
        new_robot_SO3 = R.from_euler("xyz", delta_rot).as_matrix() @ robot_SO3

        new_robot_SE3 = np.eye(4)
        new_robot_SE3[:3, :3] = new_robot_SO3
        new_robot_SE3[:3, 3] = new_robot_trans

        new_gripper_state = gripper_state + gripper_action
        new_gripper_state = np.clip(new_gripper_state, 0, 0.085)

        return new_robot_SE3, new_gripper_state

    def abs_se3_to_relative_policy_action_converter(robot_pose, gripper_pose, abs_se3_action, gripper_action):
        relative_se3 = np.linalg.inv(robot_pose) @ abs_se3_action

        relative_pos = relative_se3[:3, 3]
        relative_euler = R.from_matrix(relative_se3[:3, :3]).as_euler("xyz")
        relative_gripper = gripper_action - gripper_pose

        return np.concatenate((relative_pos, relative_euler, relative_gripper), axis=0).astype(np.float32)

    agent = SpaceMouseAgent()

    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=Path(f"datasets/{dataset_name}"),
        dataset_name=dataset_name,
        fps=10,
        use_videos=True,
    )

    collect_data(
        env,
        agent,
        dataset_recorder,
        10,
        delta_action_to_abs_se3_converter,
        abs_se3_to_relative_policy_action_converter,
    )

    env.close()
    agent.close()
