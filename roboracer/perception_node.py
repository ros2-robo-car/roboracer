import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


# LiDAR 전처리 설정
LIDAR_RANGE_MIN   = 0.1    # 최소 유효 거리 (m)
LIDAR_RANGE_MAX   = 10.0   # 최대 유효 거리 (m)
LIDAR_OUTPUT_SIZE = 108    # 다운샘플링 후 빔 수


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # ── Subscriber ───────────────────────────────────────────────────────
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )

        # ── Publisher ────────────────────────────────────────────────────────
        self.obs_pub = self.create_publisher(
            Float32MultiArray, '/perception/observation', 10
        )

        self.get_logger().info('perception node started (LiDAR only)')

    # ── 콜백 ─────────────────────────────────────────────────────────────────
    def lidar_callback(self, msg: LaserScan):
        lidar_obs = self.process_lidar(msg)   # shape: (108,)

        out = Float32MultiArray()
        out.data = lidar_obs.tolist()
        self.obs_pub.publish(out)

    # ── LiDAR 전처리 ─────────────────────────────────────────────────────────
    def process_lidar(self, msg: LaserScan) -> np.ndarray:
        ranges = np.array(msg.ranges, dtype=np.float32)

        # NaN / Inf → 최대 거리로 대체
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_RANGE_MAX)

        # 유효 범위 클리핑
        ranges = np.clip(ranges, LIDAR_RANGE_MIN, LIDAR_RANGE_MAX)

        # 0~1 정규화
        ranges = (ranges - LIDAR_RANGE_MIN) / (LIDAR_RANGE_MAX - LIDAR_RANGE_MIN)

        # 다운샘플링
        step   = max(1, len(ranges) // LIDAR_OUTPUT_SIZE)
        ranges = ranges[::step][:LIDAR_OUTPUT_SIZE]

        # 빔 수가 부족하면 패딩
        if len(ranges) < LIDAR_OUTPUT_SIZE:
            ranges = np.pad(ranges, (0, LIDAR_OUTPUT_SIZE - len(ranges)),
                            constant_values=1.0)

        return ranges  # shape: (108,)


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