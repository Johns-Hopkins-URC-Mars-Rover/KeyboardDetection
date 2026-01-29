#!/usr/bin/env python3

import sys
import time
import subprocess
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ArucoGateSubscriber(Node):
    def __init__(self):
        super().__init__("aruco_gate_subscriber")

        # -------- Parameters --------
        self.declare_parameter("image_topic", "/image_raw")  # <-- your topic
        self.declare_parameter("dict_id", int(cv2.aruco.DICT_4X4_250))
        self.declare_parameter("expected_ids", [1, 2, 3, 4])
        self.declare_parameter("stable_seconds", 10.0)
        self.declare_parameter("assumed_fps", 30)
        self.declare_parameter("decay_per_bad_frame", 3)

        # Launch this when stable:
        self.declare_parameter("launch_script", "KeyboardDetection.py")
        self.declare_parameter("log_filename", "keyboard_launch.log")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.dict_id = int(self.get_parameter("dict_id").value)
        self.expected_ids = set(int(x) for x in self.get_parameter("expected_ids").value)
        self.stable_seconds = float(self.get_parameter("stable_seconds").value)
        self.assumed_fps = int(self.get_parameter("assumed_fps").value)
        self.decay_per_bad_frame = int(self.get_parameter("decay_per_bad_frame").value)

        self.launch_script = str(self.get_parameter("launch_script").value)
        self.log_filename = str(self.get_parameter("log_filename").value)

        # -------- Stability state --------
        self.stable_frames_target = max(1, int(self.stable_seconds * self.assumed_fps))
        self.stable_counter = 0
        self.launched = False

        # -------- Paths/logging --------
        self.script_dir = Path(__file__).resolve().parent
        self.log_path = self.script_dir / self.log_filename

        # -------- CV / ArUco setup --------
        self.bridge = CvBridge()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dict_id)
        self.use_new = hasattr(cv2.aruco, "ArucoDetector")
        if self.use_new:
            params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        else:
            self.params = cv2.aruco.DetectorParameters_create()

        # -------- ROS Subscriber --------
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)

        self.get_logger().info(f"Subscribed to: {self.image_topic}")
        self.get_logger().info(f"Expected IDs: {sorted(self.expected_ids)}")
        self.get_logger().info(f"Stable: {self.stable_seconds}s (~{self.stable_frames_target} frames)")
        self.get_logger().info(f"Will launch: {self.launch_script}")
        self.get_logger().info(f"Python: {sys.executable}")
        self.get_logger().info(f"Log: {self.log_path}")

    def detect_ids(self, frame_bgr):
        if self.use_new:
            corners, ids, _ = self.detector.detectMarkers(frame_bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, self.aruco_dict, parameters=self.params)

        if ids is None or len(ids) == 0:
            return set()

        return set(int(x) for x in ids.flatten())

    def launch_once(self):
        if self.launched:
            return

        target = (self.script_dir / self.launch_script).resolve()
        if not target.exists():
            self.get_logger().error(f"Launch script not found: {target}")
            return

        self.get_logger().info(f"Launching: {target}")

        with open(self.log_path, "w") as f:
            f.write(f"Launched at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Launcher python: {sys.executable}\n")
            f.write(f"Working dir: {self.script_dir}\n\n")

        log_f = open(self.log_path, "a")

        subprocess.Popen(
            [sys.executable, str(target)],
            stdout=log_f,
            stderr=log_f,
            cwd=str(self.script_dir),
        )

        self.launched = True

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        detected = self.detect_ids(frame)
        all_present = self.expected_ids.issubset(detected)

        # Stability counter with decay (robust to flicker)
        if all_present:
            self.stable_counter = min(self.stable_frames_target, self.stable_counter + 1)
        else:
            self.stable_counter = max(0, self.stable_counter - self.decay_per_bad_frame)

        progress_sec = self.stable_counter / float(self.assumed_fps)

        # occasional log
        if msg.header.stamp.sec % 2 == 0 and msg.header.stamp.nanosec < 50_000_000:
            missing = sorted(list(self.expected_ids - detected))
            self.get_logger().info(
                f"Detected={sorted(list(detected))} Missing={missing} Stable={progress_sec:.1f}/{self.stable_seconds}s"
            )

        if self.stable_counter >= self.stable_frames_target and not self.launched:
            self.launch_once()


def main():
    rclpy.init()
    node = ArucoGateSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()