import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped


# ── 설정 ─────────────────────────────────────────────────────────────────────
MAX_STEERING_ANGLE = 0.4189   # 최대 조향각 (라디안, 약 24도)
MAX_SPEED          = 3.0      # 최대 속도 (m/s)
MIN_SPEED          = 0.5      # 최소 속도 (m/s)


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # ── Subscriber ───────────────────────────────────────────────────────
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/decision/action',
            self.action_callback,
            10
        )

        # ── Publisher ────────────────────────────────────────────────────────
        # AckermannDriveStamped → F1tenth 실차 및 시뮬레이터 표준 메시지
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        self.get_logger().info('control node started')

    # ── 콜백 ─────────────────────────────────────────────────────────────────
    def action_callback(self, msg: Float32MultiArray):
        action = msg.data

        if len(action) != 2:
            self.get_logger().warn(
                f'action 크기 불일치: 받은 크기 {len(action)}, 기대 크기 2'
            )
            return

        # SAC 출력은 -1~1 범위 → 실제 제어값으로 변환
        steering_angle = float(action[0]) * MAX_STEERING_ANGLE
        speed          = self.scale_speed(float(action[1]))

        # AckermannDriveStamped 메시지 생성 및 발행
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp         = self.get_clock().now().to_msg()
        drive_msg.header.frame_id      = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed          = speed

        self.drive_pub.publish(drive_msg)

        self.get_logger().debug(
            f'drive → 조향각: {steering_angle:.3f} rad, 속도: {speed:.3f} m/s'
        )

    def scale_speed(self, raw: float) -> float:
        """
        SAC 출력 (-1~1) → 실제 속도 (MIN_SPEED~MAX_SPEED)
        음수면 후진 없이 최소 속도로 처리
        """
        normalized = (raw + 1.0) / 2.0
        speed = MIN_SPEED + normalized * (MAX_SPEED - MIN_SPEED)
        return float(np.clip(speed, MIN_SPEED, MAX_SPEED))


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()