"""
integrated_node.py (Hybrid SAC 통합 버전)
──────────────────────────────────────────
perception_node + decision_node + control_node 를 단일 클래스로 통합
설정은 config.py에서 관리

토픽 구조:
  Subscribe: /scan (LaserScan), /odom (Odometry)
  Publish  : /drive (AckermannDriveStamped)

[수정 사항]
  1. 실차용 속도 상수(MAX_SPEED=3.0, MIN_SPEED=0.5)를 config와 분리해
     REAL_SPEED_MAX / REAL_SPEED_MIN 으로 명시적 관리
  2. 웨이포인트 로드 실패 시 obs_dim fallback 처리 추가
     (perception_node 동작 보존)
  3. lidar_callback에서 position 복사본 사용
     (odom_callback과의 레이스 컨디션 방지)
  4. odom_callback 중복 제거
     (decision_node·perception_node 둘 다 odom을 구독했던 것을 하나로 통합)
"""

import os
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import torch

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH,
)
from sac_model import SAC, get_obs_dim, build_observation
from waypoint_loader import load_waypoints
from pure_pursuit import PurePursuitController

# ── 시뮬레이터 속도 상수 (config.py 기준) ─────────────────────────────────────
#    SPEED_MIN = -5.0, SPEED_MAX = 13.0  ← SAC 학습 범위

# ── 실차 안전 속도 상수 (control_node.py 기준) ────────────────────────────────
#    실차에서는 시뮬레이터보다 훨씬 낮은 속도로 클리핑해야 함
REAL_SPEED_MAX   = 3.0    # m/s  ← control_node의 MAX_SPEED
REAL_SPEED_MIN   = 0.5    # m/s  ← control_node의 MIN_SPEED
MAX_STEERING     = 0.4189 # rad (~24도) ← control_node의 MAX_STEERING_ANGLE

# ── LiDAR / 관측 상수 ────────────────────────────────────────────────────────
NUM_LINES  = LINE_CONFIG['num_lines']
LIDAR_SIZE = OBS_CONFIG['lidar_size']
LIDAR_MIN  = OBS_CONFIG['lidar_range_min']
LIDAR_MAX  = OBS_CONFIG['lidar_range_max']
OBS_DIM    = get_obs_dim(LIDAR_SIZE, NUM_LINES)   # 정상 웨이포인트 기준 obs 크기
OBS_DIM_FALLBACK = LIDAR_SIZE                      # 웨이포인트 없을 때 fallback


class IntegratedNode(Node):
    """
    Perception + Decision + Control 통합 노드

    파이프라인 (lidar_callback 내부):
      /scan  ──► _process_lidar()          # perception_node: process_lidar()
                      │
                      ▼
               build_observation()         # perception_node: obs_pub 대신 직접 반환
                      │
                      ▼
               SAC.select_action()         # decision_node: obs_callback 역할
               action_to_line_index()
               PurePursuit.compute()
               action_to_speed()
                      │
                      ▼
               _publish_drive()            # control_node: action_callback 역할
                      │
                      ▼
                   /drive (AckermannDriveStamped)

    odom 상태:
      /odom  ──► odom_callback()           # perception_node + decision_node
                                           # 둘 다 odom을 구독했으므로 하나로 통합
    """

    def __init__(self):
        super().__init__('integrated_node')

        # ── 웨이포인트 로드 ───────────────────────────────────────────────
        # perception_node._load_waypoints() + decision_node._load_waypoints() 통합
        self._load_waypoints()

        # ── Pure Pursuit 컨트롤러 ─────────────────────────────────────────
        # decision_node.__init__ 에서 초기화하던 것
        self.controller = PurePursuitController(
            max_speed=SPEED_MAX, min_speed=SPEED_MIN
        )

        # ── SAC 모델 ──────────────────────────────────────────────────────
        # decision_node.__init__ 에서 모델 로드하던 것
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SAC(
            OBS_DIM,
            MODEL_CONFIG['action_dim'],
            MODEL_CONFIG['hidden_dims'],
            num_lines=NUM_LINES,
        ).to(self.device)

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

        # ── 차량 상태 ─────────────────────────────────────────────────────
        # perception_node: self.position(배열), self.heading, self.speed
        # decision_node  : self.x, self.y, self.heading, self.speed
        # → position 배열 방식으로 통일 (perception_node 기준)
        self.position      = np.array([0.0, 0.0])
        self.heading       = 0.0
        self.speed         = 0.0
        self.odom_received = False  # decision_node의 odom_received 플래그 보존

        # ── Subscriber / Publisher ────────────────────────────────────────
        # perception_node: lidar_sub, odom_sub, obs_pub
        # decision_node  : obs_sub, odom_sub, action_pub
        # control_node   : action_sub, drive_pub
        # → 중간 토픽(/perception/observation, /decision/action) 제거
        #   외부 입출력(/scan, /odom, /drive)만 유지
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        self.get_logger().info(
            f'integrated_node started | obs_dim={self._obs_dim} | device={self.device}'
        )

    # ── 웨이포인트 로드 ───────────────────────────────────────────────────────
    # perception_node._load_waypoints() 와 decision_node._load_waypoints() 통합
    # perception_node의 fallback(obs_dim 축소) 로직 보존
    def _load_waypoints(self):
        try:
            csv = LINE_CONFIG['centerline_csv']
            if os.path.exists(csv):
                wp = load_waypoints(
                    centerline_path=csv,
                    num_lines=NUM_LINES,
                    line_spacing=LINE_CONFIG['line_spacing'],
                )
            else:
                wp = load_waypoints(
                    map_path=LINE_CONFIG['map_path'],
                    num_lines=NUM_LINES,
                    line_spacing=LINE_CONFIG['line_spacing'],
                )
            self.waypoints_lines = wp['lines']
            # [수정 1] 정상 로드 시 obs_dim = lidar + speed + line별 전방 특징
            self._obs_dim = OBS_DIM
            self.get_logger().info(
                f'웨이포인트 로드 완료: {NUM_LINES}개 라인 | obs_dim={self._obs_dim}'
            )
        except Exception as e:
            self.get_logger().error(f'웨이포인트 로드 실패: {e}')
            self.waypoints_lines = None
            # [수정 2] perception_node 동일: 실패 시 lidar만 obs로 사용
            self._obs_dim = OBS_DIM_FALLBACK

    # ── Odometry 콜백 ─────────────────────────────────────────────────────────
    # perception_node.odom_callback + decision_node.odom_callback 통합
    # (두 노드 모두 동일한 로직이었으므로 하나로 병합)
    def odom_callback(self, msg: Odometry):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.heading = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )
        self.speed         = msg.twist.twist.linear.x
        self.odom_received = True  # decision_node의 odom_received 플래그 보존

    # ── LiDAR 전처리 ──────────────────────────────────────────────────────────
    # perception_node.process_lidar() 와 동일
    def _process_lidar(self, msg: LaserScan) -> np.ndarray:
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX)
        ranges = np.clip(ranges, LIDAR_MIN, LIDAR_MAX)
        ranges = (ranges - LIDAR_MIN) / (LIDAR_MAX - LIDAR_MIN)
        step   = max(1, len(ranges) // LIDAR_SIZE)
        ranges = ranges[::step][:LIDAR_SIZE]
        if len(ranges) < LIDAR_SIZE:
            ranges = np.pad(
                ranges, (0, LIDAR_SIZE - len(ranges)), constant_values=1.0
            )
        return ranges

    # ── LiDAR 콜백 (메인 파이프라인) ──────────────────────────────────────────
    # perception_node.lidar_callback  → obs 생성
    # decision_node.obs_callback      → SAC 추론 + Pure Pursuit
    # control_node.action_callback    → /drive 발행
    # 위 세 콜백을 순차 함수 호출로 통합
    def lidar_callback(self, msg: LaserScan):
        # ── [STEP 1] perception_node: LiDAR 전처리 ──────────────────────
        lidar = self._process_lidar(msg)

        # decision_node.obs_callback의 조기 반환 조건 보존
        if not self.odom_received or self.waypoints_lines is None:
            return

        # ── [STEP 2] perception_node: 관측 벡터 생성 ────────────────────
        # [수정 3] position 복사본 사용 → odom_callback과의 레이스 컨디션 방지
        position_snapshot = self.position.copy()

        obs = build_observation(
            lidar,
            position_snapshot,
            self.heading,
            self.speed,
            self.waypoints_lines,
            NUM_LINES,
        )

        # perception_node의 obs_dim 검증 로직 보존
        if len(obs) != self._obs_dim:
            self.get_logger().warn(
                f'obs 크기 불일치: {len(obs)} != {self._obs_dim}'
            )
            return

        # ── [STEP 3] decision_node: SAC 추론 ────────────────────────────
        # decision_node.obs_callback 의 핵심 로직
        action   = self.model.select_action(obs, training=False)
        line_idx = self.model.action_to_line_index(action)
        waypoints = self.waypoints_lines[line_idx]

        # ── [STEP 4] decision_node: Pure Pursuit 조향/속도 계산 ─────────
        steering, pp_speed = self.controller.compute(
            position_snapshot[0], position_snapshot[1],
            self.heading, self.speed, waypoints,
        )
        final_speed = min(
            self.model.action_to_speed(action, SPEED_MIN, SPEED_MAX),
            pp_speed,
        )

        # ── [STEP 5] control_node: /drive 발행 ──────────────────────────
        self._publish_drive(steering, final_speed)

    # ── Drive 발행 ────────────────────────────────────────────────────────────
    # control_node.action_callback 의 클리핑 + 발행 로직과 동일
    # [수정 4] REAL_SPEED_MIN/MAX 사용 → 실차 안전 속도 범위 적용
    def _publish_drive(self, steering: float, speed: float):
        steering = float(np.clip(steering, -MAX_STEERING,   MAX_STEERING))
        speed    = float(np.clip(speed,     REAL_SPEED_MIN, REAL_SPEED_MAX))

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp         = self.get_clock().now().to_msg()
        drive_msg.header.frame_id      = 'base_link'
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed          = speed
        self.drive_pub.publish(drive_msg)

        self.get_logger().debug(
            f'drive → 조향각: {steering:.3f} rad, 속도: {speed:.3f} m/s'
        )


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()