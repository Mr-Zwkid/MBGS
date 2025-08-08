import rtde_control
import rtde_receive
import numpy as np
import sys
import os
import time
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from lerobot.common.robot_devices.motors.configs import UR5Config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

class UR5:
    def __init__(self, config: UR5Config, home_pose=None):
        self.config = config
        self.rtde_c = None
        self.rtde_r = None

        self.IP = self.config.ip
        if home_pose is not None:
            self.home = home_pose
        else:
            self.home = self.config.home_pos
        self.lmt_x = self.config.lmt_x
        self.lmt_y = self.config.lmt_y
        self.lmt_z = self.config.lmt_z
        self.lmt_dist = self.config.lmt_dist
        self.time = self.config.time
        self.lookahead_time = self.config.lookahead_time
        self.gain = self.config.gain

        self.is_connected = False
        self.logs = {}
        self.motors = config.motors
        self.motors_joint = config.motors_joint

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"UR5 robot {self.IP} is already connected."
            )
        self.rtde_c = rtde_control.RTDEControlInterface(self.config.ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.config.ip)
        self.go_home()
        self.is_connected = True

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"UR5 robot {self.IP} is not connected."
            )
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        self.is_connected = False

    def go_home(self):
        self.rtde_c.moveL(self.home, 0.1, 0.1)
        return True

    def get_pose(self):
        pose=self.rtde_r.getActualTCPPose()
        return pose

    def get_joint(self):
        joint=self.rtde_r.getTargetQ()
        return joint

    def _safe_check(self, pose):
        # TODO smarter safety check
        # check if pose in safe range
        return self.lmt_x[0] <= pose[0] <= self.lmt_x[1] and \
            self.lmt_y[0] <= pose[1] <= self.lmt_y[1] and \
            self.lmt_z[0] <= pose[2] <= self.lmt_z[1] and \
            np.linalg.norm(pose[:3]) <= self.lmt_dist

    def track(self, target_pose):        
        """
        :param target_pose: 目标6D位姿 [x, y, z, rx, ry, rz]
        """
        if self._safe_check(target_pose):

            # 获取当前位姿
            current_pose = self.get_pose()
            if not self._safe_check(current_pose):
                print('Dangerous Position!!!!')
                self.rtde_c.servoStop()
                raise ValueError(
                    f"Current position {current_pose} is out of range."
                )

            self.rtde_c.servoL(target_pose.tolist(), 1, 1, self.time, self.lookahead_time, self.gain)
            return True
        else:
            print('Target Position Out of Range!!!!')
            self.rtde_c.servoStop()
            return False

    def read(self, data_name):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"UR5 robot {self.IP} is not connected."
            )
        start_time = time.perf_counter()
        if data_name == "Present_Position":
            values = np.array(self.get_pose())
        elif data_name == "Present_Joint":
            values = np.array(self.get_joint())