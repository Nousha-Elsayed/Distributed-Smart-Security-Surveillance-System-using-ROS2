#!/usr/bin/env python3	
 
import rclpy                      # Import ROS 2 Python client library
from rclpy.node import Node       # Import the base Node class
from std_msgs.msg import String   # Import standard String message type 

class SceneAnalysis(Node):  
    def __init__(self):
        super().__init__("scene_analysis_node")  
        self.get_logger().info("Scene Analysizer Node has been started")

        # This allows adjusting the safety threshold at runtime
        self.declare_parameter("danger_distance", 1.0)

        # Publishering scene analysis
        self.publisher_scene_analysis = self.create_publisher(String, '/scene_analysis', 10)

        # Subscribing to YOLO detections and depth estimation data
        self.subscription_detection = self.create_subscription(String, '/detected_objects', self.detection_callback, 10)
        self.subscription_depth = self.create_subscription(String, '/object_depth', self.depth_callback, 10)

        self.timer = self.create_timer(1.0, self.scene_process)

    # Dummy callbacks (REQUIRED)
    def detection_callback(self, msg):
        self.get_logger().info(f"Detected: {msg.data}")

    def depth_callback(self, msg):
        self.get_logger().info(f"Depth: {msg.data}")

    def scene_process(self):
        msg = String()
        msg.data = "test message"
        self.publisher_scene_analysis.publish(msg)
        self.get_logger().info("Publishing test message")

# Main function to run the node
def main(args=None):
    rclpy.init(args=args)         # Initialize ROS 2
    node = SceneAnalysis()         # Create an instance of your node
    rclpy.spin(node)              # Keep the node running to process callbacks
    rclpy.shutdown()              # Shutdown ROS 2 cleanly when done


# Standard Python entry point check
if __name__ == "__main__":
    main()
