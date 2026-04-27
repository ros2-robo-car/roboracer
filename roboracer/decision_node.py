import rclpy
from rclpy.node import Node

import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from roboracer.sac_model import SAC

from std_msgs.msg import Float32MultiArray


# ── 설정 ─────────────────────────────────────────────────────────────────────
OBS_DIM    = 108   # perception_node 출력 크기
ACTION_DIM = 2     # 조향각, 속도
HIDDEN_DIMS = [1024, 512, 1024, 1024, 512, 256]
self.model = SAC(OBS_DIM, ACTION_DIM, HIDDEN_DIMS).to(self.device)

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          '../models/sac_model.pth')


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        # ── 모델 로드 ─────────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = SAC(OBS_DIM, ACTION_DIM, HIDDEN_DIM).to(self.device)

        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device)
            )
            self.get_logger().info(f'모델 로드 완료: {MODEL_PATH}')
        else:
            self.get_logger().warn(f'모델 파일 없음: {MODEL_PATH} → 랜덤 가중치로 실행')

        self.model.eval()

        # ── Subscriber ───────────────────────────────────────────────────────
        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            '/perception/observation',
            self.obs_callback,
            10
        )

        # ── Publisher ────────────────────────────────────────────────────────
        self.action_pub = self.create_publisher(
            Float32MultiArray, '/decision/action', 10
        )

        self.get_logger().info('decision node started')

    # ── 콜백 ─────────────────────────────────────────────────────────────────
    def obs_callback(self, msg: Float32MultiArray):
        obs = np.array(msg.data, dtype=np.float32)

        # obs 크기 검증
        if len(obs) != OBS_DIM:
            self.get_logger().warn(
                f'obs 크기 불일치: 받은 크기 {len(obs)}, 기대 크기 {OBS_DIM}'
            )
            return

        # SAC 추론 (training=False → 최적 행동 선택)
        action = self.model.select_action(obs, training=False)

        # action 발행
        out = Float32MultiArray()
        out.data = action.tolist()
        self.action_pub.publish(out)

        self.get_logger().debug(
            f'action → 조향각: {action[0]:.3f}, 속도: {action[1]:.3f}'
        )


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