"""
decision_node.py (Hybrid SAC 버전)
──────────────────────────────────
설정은 config.py에서 관리

"""

import os
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import torch

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH,
)
from sac_model import SAC, get_obs_dim
from waypoint_loader import load_waypoints
from pure_pursuit import PurePursuitController

NUM_LINES = LINE_CONFIG['num_lines']
OBS_DIM = get_obs_dim(OBS_CONFIG['lidar_size'], NUM_LINES)


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')
        self._load_waypoints()
        self.controller = PurePursuitController(max_speed=SPEED_MAX, min_speed=SPEED_MIN)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SAC(OBS_DIM, MODEL_CONFIG['action_dim'],
                         MODEL_CONFIG['hidden_dims'], num_lines=NUM_LINES).to(self.device)

        if os.path.exists(MODEL_SAVE_PATH):
            ckpt = torch.load(MODEL_SAVE_PATH, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                self.model.load_state_dict(ckpt['model_state'])
            else:
                self.model.load_state_dict(ckpt)
            self.get_logger().info(f'모델 로드: {MODEL_SAVE_PATH}')
        else:
            self.get_logger().warn(f'모델 없음: {MODEL_SAVE_PATH}')
        self.model.eval()

        self.x, self.y, self.heading, self.speed = 0.0, 0.0, 0.0, 0.0
        self.odom_received = False

        self.obs_sub = self.create_subscription(Float32MultiArray, '/perception/observation', self.obs_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.action_pub = self.create_publisher(Float32MultiArray, '/decision/action', 10)
        self.get_logger().info('decision node started')

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
        except Exception as e:
            self.get_logger().error(f'웨이포인트 로드 실패: {e}')
            self.waypoints_lines = None

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.heading = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y**2 + q.z**2))
        self.speed = msg.twist.twist.linear.x
        self.odom_received = True

    def obs_callback(self, msg):
        obs = np.array(msg.data, dtype=np.float32)
        if len(obs) != OBS_DIM or not self.odom_received or self.waypoints_lines is None:
            return

        # Hybrid SAC: action = [line_idx(정수), speed_val(-1~1)]
        action = self.model.select_action(obs, training=False)
        line_idx = self.model.action_to_line_index(action)
        waypoints = self.waypoints_lines[line_idx]

        steering, pp_speed = self.controller.compute(
            self.x, self.y, self.heading, self.speed, waypoints)
        final_speed = min(self.model.action_to_speed(action, SPEED_MIN, SPEED_MAX), pp_speed)

        out = Float32MultiArray()
        out.data = [float(steering), float(final_speed)]
        self.action_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()