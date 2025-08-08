import os
import cv2
import time
import threading
import keyboard
import argparse
import numpy as np
import pyrealsense2 as rs
import csv
from datetime import datetime
from tactile_collect import PxTactileSensor, Recorder

import rtde_control
import rtde_receive
import sys
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
        
        return values

class DataCollector:
    def __init__(self, tactile_port='COM3', data_mode='force', sensor_numbering=1, 
                 video_width=1280, video_height=720, video_fps=30, save_folder="experiment_data", ur5_ip: str = None):
        # 创建保存文件夹
        self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(save_folder, self.experiment_name)
        os.makedirs(self.save_path, exist_ok=True)
        
        # 触觉传感器初始化
        self.px_sensor = PxTactileSensor(port=tactile_port, baudrate=460800, timeout=1.0)
        self.recorder = Recorder(self.px_sensor, sensor_numbering)
        
        if data_mode.lower() in {'force', 'fof'}:
            self.fn_task = self.recorder.record_field_of_force
        else:
            self.fn_task = self.recorder.record_field_of_magnetic
        
        # 视频录制初始化
        self.video_w = video_width
        self.video_h = video_height
        self.video_fps = video_fps
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.video_w, self.video_h, rs.format.bgr8, self.video_fps)
        self.pipeline.start(config)

        # UR5（仅记录）初始化
        self.ur_ip = ur5_ip
        self.ur_r = None              # rtde_receive 接口
        self.ur_csv_file = None       # 文件句柄
        self.ur_csv = None            # csv.writer
        if self.ur_ip:
            try:
                self.ur_r = rtde_receive.RTDEReceiveInterface(self.ur_ip)
                print(f"UR5 RTDEReceive connected: {self.ur_ip}")
            except Exception as e:
                print(f"Failed to connect UR5 RTDEReceive ({self.ur_ip}): {e}")
                self.ur_r = None
        
        # 同步控制
        self.recording_event = threading.Event()
        self.stop_event = threading.Event()
        self.start_time = None
        self.key_events = []
        self.keyboard_listening = False
        self.video_out = None

    def record_tactile(self):
        """触觉数据收集线程"""
        print("Tactile sensor ready...")
        self.recording_event.wait()  # 等待开始信号
        
        # 启动触觉记录线程
        self.recorder_thread = threading.Thread(target=self.fn_task, name='px_tactile_recording')
        self.recorder_thread.start()
        time.sleep(0.5)  # 确保线程启动
        
        self.recorder.start_recording()
        print("Tactile recording started")
        
        # 等待停止信号
        self.stop_event.wait()
        self.recorder.exit_recording()
        self.recorder_thread.join()
        print("Tactile recording stopped")

    def record_video(self):
        """视频录制线程"""
        video_path = os.path.join(self.save_path, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_out = cv2.VideoWriter(video_path, fourcc, self.video_fps, (self.video_w, self.video_h))
        
        # 若启用 UR5 记录，准备逐帧 CSV
        if self.ur_r is not None and self.ur_csv is None:
            try:
                ur_csv_path = os.path.join(self.save_path, "ur5_frame_data.csv")
                self.ur_csv_file = open(ur_csv_path, 'w', newline='')
                self.ur_csv = csv.writer(self.ur_csv_file)
                self.ur_csv.writerow([
                    'frame_number', 'rs_timestamp_ms', 'sys_time_ns',
                    'x', 'y', 'z', 'rx', 'ry', 'rz',
                    'q0', 'q1', 'q2', 'q3', 'q4', 'q5'
                ])
                print(f"UR5 per-frame CSV: {ur_csv_path}")
            except Exception as e:
                print(f"Failed to open UR5 CSV: {e}")
                self.ur_csv = None
        
        print("Video recorder ready...")
        self.recording_event.wait()  # 等待开始信号
        
        print("Video recording started")
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                    
                color_image = np.asanyarray(color_frame.get_data())
                
                # 获取帧标识并记录当前系统时间，保证严格按帧对齐
                frame_no = color_frame.get_frame_number()
                rs_ts_ms = float(color_frame.get_timestamp())  # RealSense 设备时间戳（毫秒）
                sys_ns = time.time_ns()                        # 系统单调时间（纳秒）
                
                # 读取 UR5 当前状态（与该帧对齐）
                if self.ur_r is not None and self.ur_csv is not None:
                    try:
                        pose = self.ur_r.getActualTCPPose()  # [x,y,z,rx,ry,rz]
                        q = self.ur_r.getActualQ()           # 6 关节角
                        # 写入一行，与该视频帧严格对齐
                        self.ur_csv.writerow(
                            [frame_no, rs_ts_ms, sys_ns] +
                            (pose if pose else [None]*6) +
                            (q if q else [None]*6)
                        )
                    except Exception as e:
                        # 出错时写入占位，避免错位
                        print(f"UR5 read error at frame {frame_no}: {e}")
                        if self.ur_csv is not None:
                            self.ur_csv.writerow([frame_no, rs_ts_ms, sys_ns] + [None]*12)
                
                cv2.imshow("Data Collection", color_image)
                self.video_out.write(color_image)
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                    print("ESC pressed, stopping...")
                    self.stop_event.set()
                    break
        except Exception as e:
            print(f"Video recording error: {e}")
        finally:
            if self.video_out is not None:
                self.video_out.release()
            if self.ur_csv_file is not None:
                try:
                    self.ur_csv_file.flush()
                    self.ur_csv_file.close()
                except Exception:
                    pass
                self.ur_csv_file = None
                self.ur_csv = None
            cv2.destroyAllWindows()
            print("Video recording stopped")

    def keyboard_listener(self):
        """键盘事件监听线程"""
        print("Keyboard listener ready...")
        self.recording_event.wait()  # 等待开始信号
        
        self.keyboard_listening = True
        print("Keyboard listener started")
        
        def on_key_event(e):
            if self.keyboard_listening and e.event_type == keyboard.KEY_DOWN:
                if e.name == 'enter':
                    # 处理停止信号
                    if self.recording_event.is_set():
                        print("ENTER pressed, stopping...")
                        self.stop_event.set()
                        return
                if len(e.name) == 1 and e.name.isalpha():  # 只记录单个字母键
                    if self.start_time is not None:
                        elapsed = time.time() - self.start_time
                        self.key_events.append((elapsed, e.name))
                        print(f"Key pressed: {e.name} at {elapsed:.3f}s")
        
        keyboard.hook(on_key_event)
        
        # 等待停止信号
        self.stop_event.wait()
        keyboard.unhook_all()
        self.keyboard_listening = False
        print("Keyboard listener stopped")

    def save_key_events(self):
        """保存按键事件到CSV"""
        if not self.key_events:
            print("No key events recorded")
            return
            
        csv_path = os.path.join(self.save_path, "key_events.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'key'])
            for event in self.key_events:
                writer.writerow([round(event[0], 3), event[1]])
        print(f"Key events saved to {csv_path}")

    def run(self):
        try:
            # 启动所有数据收集线程
            tactile_thread = threading.Thread(target=self.record_tactile)
            video_thread = threading.Thread(target=self.record_video)
            keyboard_thread = threading.Thread(target=self.keyboard_listener)
            
            tactile_thread.start()
            video_thread.start()
            keyboard_thread.start()
            
            # 等待第一次Enter开始
            print("\n" + "="*50)
            input("Press ENTER to start data collection...")
            print("[START] Data collection started")
            
            self.start_time = time.time()
            self.recording_event.set()  # 通知所有线程开始
            
            # 主线程等待停止信号
            while not self.stop_event.is_set():
                time.sleep(0.1)
            
            print("\n[STOP] Stopping data collection...")
            
            # 等待线程结束
            tactile_thread.join(timeout=2.0)
            video_thread.join(timeout=1.0)
            keyboard_thread.join(timeout=1.0)
            
            # 保存数据
            tactile_path = os.path.join(self.save_path, "tactile_data.csv")
            self.recorder.save_data(tactile_path)
            self.save_key_events()
            
            print("\nExperiment completed!")
            print(f"All data saved to: {self.save_path}")
            
        except Exception as e:
            print(f"Error during experiment: {e}")
        finally:
            # 确保资源释放
            self.stop_event.set()
            self.recording_event.set()
            self.pipeline.stop()
            if hasattr(self, 'recorder'):
                self.recorder.exit_recording()
            if self.video_out is not None:
                self.video_out.release()
            if getattr(self, 'ur_r', None) is not None:
                try:
                    self.ur_r.disconnect()
                except Exception:
                    pass
                self.ur_r = None
            cv2.destroyAllWindows()
            print("Resources cleaned up")

if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser(description="Multi-modal Data Collection")
    parser.add_argument("--tactile_port", type=str, default="/dev/ttyACM0", help="Tactile sensor COM port")  # ACM1 ACM0
    parser.add_argument("--data_mode", type=str, default="force", choices=['force', 'fof', 'mag'], 
                        help="Tactile data collection mode")
    parser.add_argument("--sensor_number", type=int, default=0, help="Tactile sensor numbering")
    parser.add_argument("--video_width", type=int, default=1280, help="Video width")
    parser.add_argument("--video_height", type=int, default=720, help="Video height")
    parser.add_argument("--video_fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--save_folder", type=str, default="ACM0_480", help="Root save folder") # ACM1_487   ACM0_480
    parser.add_argument("--ur5_ip", type=str, default=None, help="UR5 robot IP for per-frame pose/joint logging")
    
    args = parser.parse_args()
    
    # 运行数据收集
    collector = DataCollector(
        tactile_port=args.tactile_port,
        data_mode=args.data_mode,
        sensor_numbering=args.sensor_number,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        save_folder=args.save_folder,
        ur5_ip=args.ur5_ip
    )
    collector.run()