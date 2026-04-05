#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraStreamNode(Node):
    def __init__(self):
        super().__init__('camera_stream_node')

        # Parameters 
        self.declare_parameter('camera_source', '0')   # '0' = webcam, or file path
        self.declare_parameter('frame_rate', 10.0)     # FPS

        camera_source = self.get_parameter('camera_source').get_parameter_value().string_value
        frame_rate    = self.get_parameter('frame_rate').get_parameter_value().double_value

        # Publisher 
        self.publisher_ = self.create_publisher(Image, '/camera_frames', 10)

        # OpenCV capture 
        if camera_source.isdigit():
            self.cap = cv2.VideoCapture(int(camera_source))
        else:
            self.cap = cv2.VideoCapture(camera_source)

        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera source: {camera_source}')
            raise RuntimeError('Camera source unavailable')

        self.bridge = CvBridge()

        #  Timer publishes at the requested frame rate
        period = 1.0 / frame_rate
        self.timer = self.create_timer(period, self.publish_frame)
        self.get_logger().info(f'CameraStreamNode started | source={camera_source} | fps={frame_rate}')

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to grab frame – looping or end of file.')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
