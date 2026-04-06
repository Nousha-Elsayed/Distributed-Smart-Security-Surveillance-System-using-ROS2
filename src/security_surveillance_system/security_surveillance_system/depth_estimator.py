#!/usr/bin/env python3
"""
Node 3: Depth Estimation Node
-------------------------------
Responsibility:
    - Subscribe to /camera_frames
    - Run DepthAnythingV2 to generate a depth map
    - Publish estimated depth info to /object_depth

Topics Subscribed:
    /camera_frames   (sensor_msgs/Image)

Topics Published:
    /object_depth    (std_msgs/String – JSON payload)

Parameters:
    depth_model_path - model size: 'Small' | 'Base' | 'Large'  (default: 'Small')
"""

import json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2 as _DAV2
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False


# Map string → model config expected by DepthAnythingV2
MODEL_CONFIGS = {
    'Small': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'Base':  {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'Large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('depth_model_path', 'Small')
        model_size = self.get_parameter('depth_model_path').get_parameter_value().string_value

        # ── Load model ───────────────────────────────────────────────────────
        self.model = None
        if DEPTH_AVAILABLE:
            cfg = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS['Small'])
            self.model = _DAV2(**cfg)
            self.model.load_state_dict(
                torch.load(f'depth_anything_v2_vit{cfg["encoder"][3]}.pth', map_location='cpu'))
            self.model.eval()
            self.get_logger().info(f'DepthAnythingV2 loaded | size={model_size}')
        else:
            self.get_logger().warn('DepthAnythingV2 not installed – DEMO mode active')

        # ── Publisher / Subscriber ───────────────────────────────────────────
        self.publisher_   = self.create_publisher(String, '/object_depth', 10)
        self.subscription = self.create_subscription(
            Image, '/camera_frames', self.depth_callback, 10)

        self.bridge = CvBridge()
        self.get_logger().info('DepthEstimationNode ready.')

    def depth_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.model is not None:
            import torch
            depth_map = self.model.infer_image(frame)          # H×W float array
        else:
            # Demo: synthetic gradient depth map
            h, w = frame.shape[:2]
            y_idx = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
            depth_map = (y_idx * 10.0).astype(np.float32)     # 0–10 m range

        # Summarise: centre-crop mean depth as a simple scalar + grid stats
        h, w = depth_map.shape[:2] if depth_map.ndim == 3 else depth_map.shape
        cx, cy = w // 2, h // 2
        crop = depth_map[cy-30:cy+30, cx-30:cx+30]

        payload = json.dumps({
            'stamp':         msg.header.stamp.sec,
            'mean_depth_m':  float(np.mean(depth_map)),
            'min_depth_m':   float(np.min(depth_map)),
            'max_depth_m':   float(np.max(depth_map)),
            'center_depth_m': float(np.mean(crop)) if crop.size > 0 else 0.0,
            # Full map flattened to list for downstream nodes
            'depth_map_shape': list(depth_map.shape),
            'depth_map_flat':  depth_map.flatten().tolist(),
        })

        out_msg = String()
        out_msg.data = payload
        self.publisher_.publish(out_msg)
        self.get_logger().debug(f'Depth published | mean={np.mean(depth_map):.2f}m')


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
