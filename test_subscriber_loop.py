import cv2
import numpy as np
import time

class FakeROSImage:
    """Simulates sensor_msgs/msg/Image for testing."""
    def __init__(self, frame):
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.encoding = "bgr8"
        self.data = frame.tobytes()
        self.step = frame.shape[1] * 3
        self.header = type("Header", (), {})()
        self.header.stamp = time.time()
        self.header.frame_id = "camera_frame"


class FakeCameraSubscriber:
    def __init__(self):
        print("Fake Camera Subscriber initialized (no ROS)")
        self.frame_count = 0

    def listener_callback(self, fake_msg):
        # Convert fake ROS message → numpy array
        frame = np.frombuffer(fake_msg.data, dtype=np.uint8)
        frame = frame.reshape(fake_msg.height, fake_msg.width, 3)

        cv2.imshow("Fake Subscriber Feed", frame)
        cv2.waitKey(1)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print(f"[Fake Subscriber] Received {self.frame_count} frames")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    subscriber = FakeCameraSubscriber()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Wrap frame in Fake ROS Image
        fake_msg = FakeROSImage(frame)

        # Pass to your simulated subscriber
        subscriber.listener_callback(fake_msg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()