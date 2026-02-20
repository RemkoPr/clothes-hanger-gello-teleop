import time

import cv2
import loguru
import numpy as np
import rerun as rr

from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.base import BaseAgent, BaseDatasetRecorder, BaseEnv
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.utils import precise_wait
from robot_imitation_glue.ur5station.shirt_initialiser import ShirtInitialiser

logger = loguru.logger


class State:
    initialising = False
    rollout_active = False
    is_stopped = False
    is_paused = False


class Event:
    start_init = False
    start_rollout = False
    stop_rollout = False
    delete_last = False
    pause = False
    resume = False
    quit = False
    do_toggle_gripper = False
    do_reset = False
    do_randomise_hold_pose = False
    cancel_rollout = False

    def clear(self):
        for attr in self.__dict__:
            setattr(self, attr, False)


def init_keyboard_listener(event: Event, state: State):
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.enter and not state.initialising and not state.rollout_active:
                event.start_init = True

            elif key == keyboard.Key.enter and state.initialising and not state.rollout_active:
                event.start_rollout = True

            elif key == keyboard.Key.enter and state.rollout_active:
                event.stop_rollout = True

            elif hasattr(key, "char") and key.char == "p" and not state.rollout_active and not state.is_paused:
                event.pause = True

            elif hasattr(key, "char") and key.char == "p" and state.is_paused:
                event.resume = True

            elif hasattr(key, "char") and key.char == "c" and state.rollout_active:
                event.cancel_rollout = True

            elif hasattr(key, "char") and key.char == "g" and state.is_paused:
                event.do_toggle_gripper = True

            elif hasattr(key, "char") and key.char == "r" and state.is_paused:
                event.do_reset = True

            elif hasattr(key, "char") and key.char == "m" and state.is_paused:
                event.do_randomise_hold_pose = True

            elif hasattr(key, "char") and key.char == "q":
                event.quit = True

            elif hasattr(key, "char") and key.char == "d" and not state.rollout_active:
                event.delete_last = True
        except Exception as e:
            logger.error(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener


def eval(  # noqa: C901
    env: BaseEnv,
    policy_agent: BaseAgent,
    recorder: BaseDatasetRecorder,
    fps=10,
    eval_dataset: LeRobotDataset = None,
    eval_dataset_image_keys: list = None,
    env_observation_image_keys: list = None,
    eval_dataset_episode: int = -1,
    # Legacy singular params — kept for backward compatibility
    eval_dataset_image_key: str = None,
    env_observation_image_key: str = None,
):
    """
    Evaluate a (policy) agent on a robot environment.

    Uses the same state machine as collect_data.py:
        paused  --[Enter]--> initialising  --[Enter]--> rollout_active  --[Enter]--> save & pause

    Args:
        env: robot environment
        policy_agent: policy agent
        recorder: dataset recorder
        fps: control frequency
        eval_dataset: optional dataset to show initial scene overlay
        eval_dataset_image_keys: keys for overlay images from eval_dataset
        env_observation_image_keys: keys for live observation images to display
        eval_dataset_episode: which episode from eval_dataset to show (-1 = none)
        eval_dataset_image_key: (legacy) singular version of eval_dataset_image_keys
        env_observation_image_key: (legacy) singular version of env_observation_image_keys
    """
    # Handle legacy singular key params
    if eval_dataset_image_keys is None:
        eval_dataset_image_keys = [eval_dataset_image_key] if eval_dataset_image_key else ["scene_image"]
    if env_observation_image_keys is None:
        env_observation_image_keys = [env_observation_image_key] if env_observation_image_key else ["scene_image"]

    rr.init("robot_imitation_glue_eval")
    rr.spawn(port=9875, memory_limit="20%", connect=True)

    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)
    shirt_initialiser = ShirtInitialiser()

    # Move teleop robot to current teleop agent pose
    env.teleop_robot.move_to_joint_configuration(env.teleop_agent.get_action()[:6], joint_speed=0.1).wait()

    # Load initial overlay images from eval dataset if provided
    initial_images = {}
    instruction = None
    if eval_dataset is not None and eval_dataset_episode > -1:
        for eval_key in eval_dataset_image_keys:
            step_idx = eval_dataset.episode_data_index["from"][eval_dataset_episode].item()
            initial_image = eval_dataset[step_idx][eval_key]
            initial_image = initial_image.permute(1, 2, 0).numpy()
            initial_image = (initial_image * 255).astype(np.uint8)
            initial_images[eval_key] = initial_image
            instruction = eval_dataset[step_idx].get("task", None)
        if instruction:
            logger.info(f"Eval episode {eval_dataset_episode} instruction: {instruction}")

    control_period = 1 / fps
    while not state.is_stopped:
        try:
            cycle_end_time = time.time() + control_period

            # ── State machine event handling ──────────────────────────
            if not state.initialising and not state.rollout_active and event.start_init:
                logger.info("======================= Initialising rollout")
                state.initialising = True
                #env.move_hold_robot_random_translation()
                shirt_initialiser.init_random_line()
                env.move_teleop_robot_to_home_pose(joint_speed=0.2)
                observation = env.get_observations()
                recorder.start_episode()
                recorder.record_step(observation, np.array([0] * 7).astype(np.float32))
                env.clothes_hanger.init_baseline(observation["clothes_hanger"])
                policy_agent.reset()
                if not state.is_paused:
                    env.teleop_robot.move_to_joint_configuration(env.teleop_agent.get_action()[:6], joint_speed=0.2).wait()

            elif state.initialising and not state.rollout_active and event.start_rollout:
                logger.info("======================= Start rollout")
                state.rollout_active = True
                state.initialising = False
                state.is_paused = False

            elif state.rollout_active and event.stop_rollout:
                logger.info("======================= Stop rollout")
                state.rollout_active = False
                recorder.save_episode()
                recorder.finish_recording()
                recorder.reinitialize_dataset()
                logger.info(f"Saved episode {recorder.n_recorded_episodes}")
                state.is_paused = True

            elif (state.rollout_active or state.initialising) and event.cancel_rollout:
                logger.info("======================= Cancel rollout")
                state.rollout_active = False
                state.initialising = False
                time.sleep(0.5)
                recorder.clear_episode()
                state.is_paused = True

            elif event.delete_last and not state.rollout_active:
                logger.info("======================= Delete last episode")
                raise NotImplementedError("delete last episode not implemented")

            elif event.pause and not state.rollout_active:
                state.is_paused = True
                logger.info("======================= Pause teleop")

            elif event.resume and state.is_paused:
                state.is_paused = False
                logger.info("======================= Resume teleop, first move slowly to current teleop pose")
                action = env.teleop_agent.get_action()
                logger.debug(f"Action: {action}")
                env.teleop_robot.move_to_joint_configuration(action[:6], joint_speed=0.2).wait()
                logger.info("======================= Resuming teleop.")

            elif event.do_toggle_gripper and state.is_paused:
                logger.info("======================= Toggling gripper")
                env.toggle_holding_gripper()

            elif event.do_randomise_hold_pose and state.is_paused:
                logger.info("======================= Randomising T-shirt hold pose")
                env.move_hold_robot_random_translation()

            elif event.do_reset and state.is_paused:
                logger.info("======================= Resetting robot to initial pose")
                env.move_teleop_robot_to_home_pose(joint_speed=0.2)

            elif event.quit:
                logger.info("Quitting...")
                state.is_stopped = True
                state.initialising = False
                state.rollout_active = False
                listener.stop()
                time.sleep(0.5)
                recorder.clear_episode()
                recorder.finish_recording()
                logger.info("Finished evaluation and quit.")
                return

            # Clear all events
            event.clear()

            # ── Observations & GUI ────────────────────────────────────
            observation = env.get_observations()
            vis_img = observation.get("scene_image", np.zeros((480, 640, 3), dtype=np.uint8)).copy()
            wrist_img = observation.get("wrist_wilson_image", np.zeros((480, 640, 3), dtype=np.uint8)).copy()

            if state.initialising:
                shirt_initialiser.annotate_image(wrist_img)
                cv2.putText(vis_img, "INITIALISING", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
            if state.rollout_active:
                cv2.putText(vis_img, "ROLLOUT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if state.is_paused:
                cv2.putText(vis_img, "PAUSED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(
                vis_img,
                f" # episodes: {recorder.n_recorded_episodes}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            rr.log("clothes_hanger", rr.Scalars(observation["clothes_hanger"]))
            rr.log("image", rr.Image(vis_img, rr.ColorModel.RGB))
            rr.log("Init instruction", rr.Image(wrist_img, rr.ColorModel.RGB))
            for obs_key in env_observation_image_keys:
                if obs_key in observation:
                    rr.log(f"obs: {obs_key}", rr.Image(observation[obs_key], rr.ColorModel.RGB))

            # Show eval dataset overlay images (blended with live)
            if eval_dataset is not None and initial_images:
                for eval_key in eval_dataset_image_keys:
                    if eval_key in initial_images and eval_key in observation:
                        blended = cv2.addWeighted(observation[eval_key], 0.5, initial_images[eval_key], 0.5, 0)
                        rr.log(f"initial_{eval_key}", rr.Image(blended, rr.ColorModel.RGB))

            # ── Paused: do nothing ────────────────────────────────────
            if state.is_paused:
                time.sleep(0.1)
                continue

            # ── Rollout phase: policy controls the robot ──────────────
            if state.rollout_active:
                action = policy_agent.get_action(observation)
                logger.debug(f"policy action: {action}")

                env.act(
                    robot_pose=action[:6],
                    gripper_pose=action[6],
                    timestamp=time.time() + control_period,
                    disable_gripper=False,
                )

                recorder.record_step(observation, action.astype(np.float32))

            # ── Teleop phase (initialising or free movement) ──────────
            else:
                action = env.teleop_agent.get_action()
                logger.info(f"Action: {action}")

                env.act(
                    robot_pose=action[:6],
                    gripper_pose=action[6],
                    timestamp=time.time() + control_period,
                    disable_gripper=False,
                )

            # Wait for end of control period
            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)
            else:
                logger.warning("cycle time exceeded control period")

        except (KeyboardInterrupt, Exception) as e:
            import traceback
            traceback.print_exc()
            logger.error(f"An error occurred, clearing current episode and finishing recording.\nOriginal error: {e}")
            listener.stop()
            time.sleep(0.5)
            recorder.clear_episode()
            recorder.finish_recording()
            logger.info("Finished evaluation and quit.")
            return


if __name__ == "__main__":
    """example of how to use the eval function"""
    import os

    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.mock import MockAgent, MockEnv, mock_agent_to_pose_converter

    env = MockEnv()
    env.reset()
    teleop_agent = MockAgent()
    policy_agent = MockAgent()

    if os.path.exists("datasets/demo"):
        dataset = LeRobotDataset(repo_id="mock", root="datasets/demo")
    else:
        dataset = None

    if os.path.exists("datasets/test_dataset"):
        os.system("rm -rf datasets/test_dataset")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir="datasets/test_dataset",
        dataset_name="test_dataset",
        fps=10,
        use_videos=True,
    )

    eval(
        env,
        teleop_agent,
        policy_agent,
        dataset_recorder,
        fps=2,
        eval_dataset=dataset,
    )
