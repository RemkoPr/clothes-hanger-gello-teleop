import time

import serial

from schunk import SchunkGripperProcess

ORIGINAL_CLOTHES_HANGER_GRASP_WIDTH = 0.020
BCH_GRASP_WIDTH = 0.0055
SCHUNK_SOPHIE_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:2:1.0-port0,12,115200,8E1"
SCHUNK_WILSON_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:1:1.0-port0,11,115200,8E1"
SCHUNK_WILSON_PORT = "/dev/ttyUSB0,11"
#SCHUNK_SOPHIE_PORT = "/dev/ttyUSB1,12"

gripper1 = SchunkGripperProcess(SCHUNK_WILSON_PORT)

gripper1.move(BCH_GRASP_WIDTH + 0.00).wait()
time.sleep(0.5)
gripper1.shutdown()
