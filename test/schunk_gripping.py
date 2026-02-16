from airo_robots.grippers.hardware.schunk_process import SchunkGripperProcess

ORIGINAL_CLOTHES_HANGER_GRASP_WIDTH = 0.02
BCH_GRASP_WIDTH = 0.0055
SCHUNK_SOPHIE_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:13.3:1.0-port0,12,115200,8E1"
SCHUNK_WILSON_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:1:1.0-port0,11,115200,8E1"
gripper = SchunkGripperProcess(SCHUNK_SOPHIE_PORT)

gripper.move(BCH_GRASP_WIDTH + 0.005)