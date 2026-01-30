from airo_robots.manipulators.hardware.ur_rtde import URrtde
from grippers.schunk_process import SchunkGripperProcess
import time
from loguru import logger


PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:13.3:1.0-port0,12,115200,8E1"
PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:1:1.0-port0,11,115200,8E1"
gripper = SchunkGripperProcess(usb_interface=PORT)
#print(gripper.get_current_width())
time.sleep(1)
gripper.move(0.03)
while True:
    time.sleep(1)