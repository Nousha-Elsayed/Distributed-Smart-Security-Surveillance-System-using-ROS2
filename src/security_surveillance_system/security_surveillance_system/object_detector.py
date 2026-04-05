#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('model_path', 'yolov8n.pt')

        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        model_path          = self.get_parameter('model_path').get_parameter_value().string_value

        # Load YOLO model
        if YOLO_AVAILABLE:
            self.model = YOLO(model_path)
            self.get_logger().info(f'YOLO model loaded: {model_path}')
        else:
            self.model = None
            self.get_logger().warn('ultralytics not installed – running in DEMO mode')

        # Publisher and Subscriber
        self.publisher_  = self.create_publisher(String, '/detected_objects', 10)
        self.subscription = self.create_subscription(
            Image, '/camera_frames', self.detection_callback, 10)

        self.bridge = CvBridge()
        self.get_logger().info('ObjectDetectionNode ready.')

    def detection_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detections = []

        if self.model is not None:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'label':      result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': {
                            'x1': float(box.xyxy[0][0]),
                            'y1': float(box.xyxy[0][1]),
                            'x2': float(box.xyxy[0][2]),
                            'y2': float(box.xyxy[0][3]),
                        }
                    })
        else:
            # Demo / fallback when YOLO is not installed
            import random, time
            if random.random() > 0.5:
                detections.append({
                    'label': 'person', 'confidence': 0.92,
                    'bbox': {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 400}
                })

        payload = json.dumps({
            'stamp':      msg.header.stamp.sec,
            'detections': detections
        })
        out_msg = String()
        out_msg.data = payload
        self.publisher_.publish(out_msg)

        if detections:
            labels = [d['label'] for d in detections]
            self.get_logger().info(f'Detected: {labels}')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
