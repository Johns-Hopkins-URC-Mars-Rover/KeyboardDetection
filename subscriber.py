import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriber(Node):
    def __init__(self):
        super().__init('camera_subscriber')
        self.subscription = self.create_subscription(Image, 'camera/image', self.listener_callback, 10)
        self.bridge = CvBridge()
    
    def listener_callback(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        cv2.imshow("Live Camera Feed", frame)
        cv2.waitKey(1)

        # log every 30th frame instead of every frame to avoid spam
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f"Received {self.frame_count} frames")

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("CameraSubscriber interrupted, shutting down...")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()