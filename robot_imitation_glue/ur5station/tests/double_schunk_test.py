import os
import time
from robot_imitation_glue.grippers.schunk_process import SchunkGripperProcess


#SCHUNK_GRIPPER_HOST1 = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0,11,115200,8E1"  # run bks_scan -H <usb> to find the slaveID, run dmesg | grep tty to find the usb port
#SCHUNK_GRIPPER_HOST1 = "/dev/ttyUSB3,11,115200,8E1"
SCHUNK_GRIPPER_HOST1 = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:5:1.0-port0,11,115200,8E1"
#SCHUNK_GRIPPER_HOST2 = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0,14,115200,8E1"
#SCHUNK_GRIPPER_HOST2 = "/dev/ttyUSB2,14,115200,8E1"
SCHUNK_GRIPPER_HOST2 = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:8:1.0-port0,14,115200,8E1"


#os.environ["BKS_HOST"] = SCHUNK_GRIPPER_HOST1
gripper1 = SchunkGripperProcess(SCHUNK_GRIPPER_HOST1)
gripper2 = SchunkGripperProcess(SCHUNK_GRIPPER_HOST2)
time.sleep(2)
gripper1.max_grasp_force = gripper1.gripper_specs.min_force  # minimal force for EGK40 is 55N
gripper1.speed = gripper1.gripper_specs.max_speed
gripper2.max_grasp_force = gripper2.gripper_specs.min_force  # minimal force for EGK40 is 55N
gripper2.speed = gripper2.gripper_specs.max_speed

while True:
    gripper1.move(0.01)
    gripper2.move(0.01).wait()
    time.sleep(1)
    gripper1.move(0.03)
    gripper2.move(0.03).wait()
    time.sleep(1)
