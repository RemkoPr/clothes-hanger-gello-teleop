from ipc_camera import RGBCameraPublisher, RGBCameraSubscriber
from functools import partial
from airo_camera_toolkit.cameras.realsense.realsense import Realsense


REALSENSE_SERIAL = "130322271048"
CAM_RGB_TOPIC = "wrist_wilson_rgb"
CAM_RESOLUTION_TOPIC = "wrist_wilson_resolution"

class CameraFactory:
    def create_wrist_camera(serial_number):
        return Realsense(resolution=Realsense.RESOLUTION_720, fps=30, serial_number=serial_number, enable_depth=False, enable_pointcloud=False)

        
_camera_publisher = RGBCameraPublisher(
            partial(CameraFactory.create_wrist_camera, serial_number=REALSENSE_SERIAL),
            CAM_RGB_TOPIC,
            CAM_RESOLUTION_TOPIC,
            100,
        )
_camera_publisher.start()

_camera_subscriber = RGBCameraSubscriber(
            CAM_RESOLUTION_TOPIC,
            CAM_RGB_TOPIC,
        )


image = _camera_subscriber.get_rgb_image_as_int()