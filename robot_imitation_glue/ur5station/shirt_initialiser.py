from functools import partial
import time
import rerun as rr
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from loguru import logger
import random
import cv2
import numpy as np

from robot_imitation_glue.ipc_camera import RGBCameraPublisher, RGBCameraSubscriber


class ShirtInitialiser:
    def __init__(self):
        self.relative_start_point = (0.8, 0.52)  # height, width
        self.angle_range = (100, 260)
        self.distance_range = (0.05, 0.3)
        self.arrow_length_px = 1000
        self.angle = None
        self.distance = None

    def init_random_line(self):
        self.angle = random.uniform(*self.angle_range)
        self.distance = random.uniform(*self.distance_range)

    def annotate_image(self, image):
        self.draw_line_on_image(image)
        self.draw_distance_on_image(image)

    def draw_distance_on_image(self, image, color=(255, 0, 0), thickness=5):
        # put distance in text on image near start point
        image_size = image.shape[0], image.shape[1]  # (height, width)
        text_position = (int(image_size[1] * self.relative_start_point[1]) - 50,
                         int(image_size[0] * self.relative_start_point[0]) + 100)
        cv2.putText(image, f"{self.distance*100:.0f} cm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness)

    def draw_line_on_image(self, image, color=(255, 0, 0), thickness=5):
        image_size = image.shape[0], image.shape[1]  # (height, width)
        start_point = (int(image_size[1] * self.relative_start_point[1]),
                        int(image_size[0] * self.relative_start_point[0]))
                       
        end_point = (
            int(start_point[0] + self.arrow_length_px * np.sin(np.radians(self.angle))),
            int(start_point[1] + self.arrow_length_px * np.cos(np.radians(self.angle))),
        )
        # keep end_point within image bounds
        end_point = (max(0, min(end_point[0], image_size[1] - 1)), max(0, min(end_point[1], image_size[0] - 1)))
        logger.info(f"image shape: {image.shape}, start_point: {start_point}, end_point: {end_point}, angle: {self.angle}")
        cv2.line(image, start_point, end_point, color, thickness)


def _do_not_use_this_camera_create_function(serial_number):
    # Needs to be declared here for multiprocessing to work, 
    # even though it should be below if __name__ == "__main__":
    return Realsense(
        resolution=Realsense.RESOLUTION_720,
        fps=30,
        serial_number=serial_number,
        enable_depth=False,
        enable_pointcloud=False,
    )



if __name__ == "__main__":
    WRIST_WILSON_REALSENSE_SERIAL = "130322271048"
    WRIST_WILSON_CAM_RGB_TOPIC = "wrist_wilson_rgb"
    WRIST_WILSON_CAM_RESOLUTION_TOPIC = "wrist_wilson_resolution"

    rr.init("robot_imitation_glue")
    rr.spawn(port=9875, memory_limit="50%", connect=True)

    logger.info("Creating Wilson wrist camera publisher.")
    _wrist_wilson_camera_publisher = RGBCameraPublisher(
        partial(_do_not_use_this_camera_create_function, serial_number=WRIST_WILSON_REALSENSE_SERIAL),
        WRIST_WILSON_CAM_RGB_TOPIC,
        WRIST_WILSON_CAM_RESOLUTION_TOPIC,
        100,
    )
    _wrist_wilson_camera_publisher.start()

    logger.info("Creating Wilson wrist camera subscriber.")
    _wrist_wilson_camera_subscriber = RGBCameraSubscriber(
        WRIST_WILSON_CAM_RESOLUTION_TOPIC,
        WRIST_WILSON_CAM_RGB_TOPIC,
    )
    shirt_initialiser = ShirtInitialiser()
    while True:
        wrist_wilson_image = _wrist_wilson_camera_subscriber.get_rgb_image_as_int()
        shirt_initialiser.init_random_line()
        shirt_initialiser.annotate_image(wrist_wilson_image)
        rr.log("wrist_wilson_image", rr.Image(wrist_wilson_image, rr.ColorModel.RGB))
        time.sleep(0.2)