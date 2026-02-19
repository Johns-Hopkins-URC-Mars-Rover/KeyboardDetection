# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import sys
import subprocess
import rclpy
from rclpy.node import Node
from pathlib import Path

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

# put in as params???
DICT = cv2.aruco.DICT_4X4_250
CAM_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

EXPECTED_IDS = {1, 2, 3, 4}
STABLE_DURATION = 0.1          # seconds
ASSUMED_FPS = 30              # used for stability counter target
STABLE_FRAMES_TARGET = STABLE_DURATION * ASSUMED_FPS

# Tolerance: if markers flicker for a moment, don't reset to 0
DECAY_PER_BAD_FRAME = 3       # higher = stricter; lower = more forgiving

KEYBOARD_SCRIPT = "KeyboardDetection.py"
LOG_PATH = Path(__file__).with_name("keyboard_launch.log")
LAUNCH_KEY = "temp"


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
        self.stable_counter = 0
        self.keyboard_launched = False
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10)
        self.br = CvBridge()

        self.params = cv2.aruco.DetectorParameters_create()
        # self.publisher = self.create_publisher(
        #     Image,
        #     'temp',
        #     10
        # )
    def listener_callback(self, msg):
        # self.get_logger().info('Receiving video frame') 
        try:
            current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            corners, ids, rejected = cv2.aruco.detectMarkers(current_frame, self.aruco_dict, parameters=self.params)

            out = current_frame.copy()
            detected_ids = set()

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(out, corners, ids)

                detected_ids = ids.flatten()

                for i, c in enumerate(corners):
                    pts = c[0].astype(int)
                    cx, cy = pts.mean(axis=0).astype(int)
                    marker_id = int(ids[i][0])
                    cv2.putText(out, f"id:{marker_id}", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                all_present = EXPECTED_IDS.issubset(detected_ids)

                # Stability counter logic
                if all_present:
                    self.stable_counter = min(STABLE_FRAMES_TARGET, self.stable_counter + 1)
                else:
                    self.stable_counter = max(0, self.stable_counter - DECAY_PER_BAD_FRAME)

                # On-screen status
                if all_present:
                    cv2.putText(out, "All markers detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                # Progress bar text
                progress_sec = self.stable_counter / ASSUMED_FPS
                cv2.putText(out, f"Stable: {progress_sec:.1f}s / {STABLE_DURATION}s",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Launch KeyboardDetection once, when stable enough
                if all_present and not self.keyboard_launched:
                    # Redirect stdout/stderr to log so you can debug instantly
                    self.get_logger().info('Receiving video frame') 
                    subprocess.Popen([sys.executable, KEYBOARD_SCRIPT, LAUNCH_KEY], cwd=str(Path(__file__).parent))
                    self.keyboard_launched = True

            else:
                self.stable_counter = max(0, self.stable_counter - DECAY_PER_BAD_FRAME)
                cv2.putText(out, "No markers", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("ArUco Live", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return
            elif key == ord('s'):
                fname = f"aruco_{time.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, out)
                print(f"Saved {fname}")
        except CvBridgeError as e:
            self.get_logger().info('Error ')


def main(args=None):
    rclpy.init(args=args)

    camera_subscriber = CameraSubscriber()

    rclpy.spin(camera_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
