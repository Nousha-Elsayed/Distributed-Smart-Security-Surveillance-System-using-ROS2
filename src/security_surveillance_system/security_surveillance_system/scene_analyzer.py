#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SceneAnalysisNode(Node):
    def __init__(self):
        super().__init__('scene_analysis_node')

        # ── Parameters 
        self.declare_parameter('danger_distance', 1.5)
        self.danger_dist = self.get_parameter('danger_distance').get_parameter_value().double_value

        # latest messages cache 
        self.latest_detections: dict | None = None
        self.latest_depth:      dict | None = None

        # Publisher 
        self.publisher_ = self.create_publisher(String, '/scene_analysis', 10)

        #  Subscribers 
        self.create_subscription(String, '/detected_objects', self._det_cb,   10)
        self.create_subscription(String, '/object_depth',     self._depth_cb, 10)

        # ─ Fusion timer (10 Hz) 
        self.create_timer(0.1, self._fuse_and_publish)
        self.get_logger().info(f'SceneAnalysisNode ready | danger_distance={self.danger_dist}m')

    #  Callbacks 
    def _det_cb(self, msg: String):
        try:
            self.latest_detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Bad JSON from /detected_objects')

    def _depth_cb(self, msg: String):
        try:
            self.latest_depth = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Bad JSON from /object_depth')

    #  Fusion logic 
    def _fuse_and_publish(self):
        if self.latest_detections is None or self.latest_depth is None:
            return

        center_depth = self.latest_depth.get('center_depth_m', 99.0)
        mean_depth   = self.latest_depth.get('mean_depth_m',   99.0)

        enriched = []
        for det in self.latest_detections.get('detections', []):
            # Simple heuristic: assign center depth to every detection
            # (A full implementation would sample the depth map at the bbox centre)
            assigned_depth = center_depth
            is_close  = assigned_depth < self.danger_dist
            is_unusual = det['label'] not in ['person', 'car', 'truck', 'bicycle', 'dog']

            enriched.append({
                'label':      det['label'],
                'confidence': det['confidence'],
                'bbox':       det['bbox'],
                'depth_m':    assigned_depth,
                'too_close':  is_close,
                'unusual':    is_unusual,
            })

        scene_flags = {
            'any_too_close': any(e['too_close']  for e in enriched),
            'any_unusual':   any(e['unusual']    for e in enriched),
            'object_count':  len(enriched),
        }

        payload = json.dumps({
            'stamp':       self.latest_detections.get('stamp', 0),
            'mean_depth_m': mean_depth,
            'objects':     enriched,
            'scene_flags': scene_flags,
        })

        out_msg = String()
        out_msg.data = payload
        self.publisher_.publish(out_msg)

        if scene_flags['any_too_close'] or scene_flags['any_unusual']:
            self.get_logger().warn(f'Scene alert! flags={scene_flags}')


def main(args=None):
    rclpy.init(args=args)
    node = SceneAnalysisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
