"""
perception_node.py (라인 선택 버전)
───────────────────────────────────
설정은 config.py에서 관리
"""

import os
import sys
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import OBS_CONFIG, LINE_CONFIG
from waypoint_loader import load_waypoints
from sac_model import build_observation

LIDAR_SIZE = OBS_CONFIG['lidar_size']
LIDAR_MIN  = OBS_CONFIG['lidar_range_min']
LIDAR_MAX  = OBS_CONFIG['lidar_range_max']
NUM_LINES  = LINE_CONFIG['num_lines']


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self._load_waypoints()
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.speed = 0.0
        self.odom_received = False

        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.obs_pub = self.create_publisher(Float32MultiArray, '/perception/observation', 10)
        self.get_logger().info(f'perception node started (obs_dim={self.obs_dim})')

    def _load_waypoints(self):
        try:
            csv = LINE_CONFIG['centerline_csv']
            if os.path.exists(csv):
                wp = load_waypoints(centerline_path=csv, num_lines=NUM_LINES,
                                    line_spacing=LINE_CONFIG['line_spacing'])
            else:
                wp = load_waypoints(map_path=LINE_CONFIG['map_path'], num_lines=NUM_LINES,
                                    line_spacing=LINE_CONFIG['line_spacing'])
            self.waypoints_lines = wp['lines']
            self.obs_dim = LIDAR_SIZE + 1 + NUM_LINES * 2
        except Exception as e:
            self.get_logger().error(f'웨이포인트 로드 실패: {e}')
            self.waypoints_lines = None
            self.obs_dim = LIDAR_SIZE

    def odom_callback(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.heading = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y**2 + q.z**2))
        self.speed = msg.twist.twist.linear.x
        self.odom_received = True

    def lidar_callback(self, msg):
        lidar = self.process_lidar(msg)
        if self.waypoints_lines is not None and self.odom_received:
            obs = build_observation(lidar, self.position, self.heading,
                                    self.speed, self.waypoints_lines, NUM_LINES)
        else:
            obs = lidar
        out = Float32MultiArray()
        out.data = obs.tolist()
        self.obs_pub.publish(out)

    def process_lidar(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX)
        ranges = np.clip(ranges, LIDAR_MIN, LIDAR_MAX)
        ranges = (ranges - LIDAR_MIN) / (LIDAR_MAX - LIDAR_MIN)
        step = max(1, len(ranges) // LIDAR_SIZE)
        ranges = ranges[::step][:LIDAR_SIZE]
        if len(ranges) < LIDAR_SIZE:
            ranges = np.pad(ranges, (0, LIDAR_SIZE - len(ranges)), constant_values=1.0)
        return ranges


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()