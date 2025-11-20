import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriber(Node):
    def __init__(self):
        super.__init('camera_subscriber')
        self.subscription = self.create_subscription(Image, 'camer/image', self.listener_callback, 10)
        self.bridge = CvBridge()
    
    def listener_callback(self, messgae):
        frame = self.bridge.imgmsg_to_cv2(messgae, 'brg8')
        cv2.imshow("Live Camera Feed", frame)
        cv2.waitKey(1)
        self.get_logger().info("Frame recieved")

def main():
    rclpy.init(args = None)
    node = CameraSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()