import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher = self.create_publisher(Image, 'camera/image', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened(): 
            self.get_logger().error("Error: Accessing camera")
            exit()
        
        self.timer = self.create_timer(1/30, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().error("Error: No frames captured")
            return None
        
        message = self.bridge.cv2_to_imgmsg(frame, encoding = 'bgr8')

        # timestamp and frame name
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera_frame"

        self.publisher.publish(message)
        self.get_logger().info("Published Image")

def main():
    rclpy.init(args = None)
    node = CameraPublisher()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__" :
    main()









