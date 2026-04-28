"""
pure_pursuit.py
───────────────
Pure Pursuit 경로 추종 컨트롤러

입력: 현재 차량 상태 (x, y, heading) + 웨이포인트 배열
출력: 조향각 (rad), 속도 (m/s)

속도·조향·lookahead 등 모든 파라미터는 config.py의
PURE_PURSUIT_CONFIG, SPEED_MIN, SPEED_MAX 에서 관리
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import PURE_PURSUIT_CONFIG, SPEED_MIN, SPEED_MAX
from waypoint_loader import get_nearest_waypoint_idx, get_lookahead_point


# ── config에서 기본값 로드 ────────────────────────────────────────────────────
WHEELBASE      = PURE_PURSUIT_CONFIG['wheelbase']
MIN_LOOKAHEAD  = PURE_PURSUIT_CONFIG['min_lookahead']
MAX_LOOKAHEAD  = PURE_PURSUIT_CONFIG['max_lookahead']
LOOKAHEAD_GAIN = PURE_PURSUIT_CONFIG['lookahead_gain']
MAX_SPEED      = SPEED_MAX
MIN_SPEED      = SPEED_MIN
CURVATURE_GAIN = PURE_PURSUIT_CONFIG['curvature_gain']
MAX_STEERING   = PURE_PURSUIT_CONFIG['max_steering']


class PurePursuitController:
    """
    Pure Pursuit 경로 추종 컨트롤러

    사용법:
        controller = PurePursuitController()
        steering, speed = controller.compute(x, y, heading, speed, waypoints)
    """

    def __init__(self,
                 wheelbase:      float = WHEELBASE,
                 min_lookahead:  float = MIN_LOOKAHEAD,
                 max_lookahead:  float = MAX_LOOKAHEAD,
                 lookahead_gain: float = LOOKAHEAD_GAIN,
                 max_speed:      float = MAX_SPEED,
                 min_speed:      float = MIN_SPEED,
                 curvature_gain: float = CURVATURE_GAIN,
                 max_steering:   float = MAX_STEERING):

        self.wheelbase      = wheelbase
        self.min_lookahead  = min_lookahead
        self.max_lookahead  = max_lookahead
        self.lookahead_gain = lookahead_gain
        self.max_speed      = max_speed
        self.min_speed      = min_speed
        self.curvature_gain = curvature_gain
        self.max_steering   = max_steering

        # 이전 스텝 정보 (바퀴 떨림 억제용)
        self._prev_steering = 0.0
        self._nearest_idx   = 0

    def compute(self,
                x: float,
                y: float,
                heading: float,
                current_speed: float,
                waypoints: np.ndarray) -> tuple:
        """
        Pure Pursuit 계산

        Args:
            x, y:          현재 차량 위치 (m)
            heading:       현재 차량 방향 (rad)
            current_speed: 현재 속도 (m/s)
            waypoints:     추종할 웨이포인트 배열 (N, 2)

        Returns:
            (steering_angle, target_speed)
            steering_angle: 조향각 (rad, -max ~ +max)
            target_speed:   목표 속도 (m/s)
        """
        position = np.array([x, y])

        # 1) 가장 가까운 웨이포인트 찾기
        self._nearest_idx = get_nearest_waypoint_idx(position, waypoints)

        # 2) 속도 기반 lookahead 거리 계산
        lookahead_dist = self._calc_lookahead(abs(current_speed))

        # 3) 목표점 찾기
        target_point, _ = get_lookahead_point(
            position, waypoints, lookahead_dist, self._nearest_idx
        )

        # 4) 조향각 계산
        raw_steering = self._calc_steering(x, y, heading, target_point,
                                           lookahead_dist)

        # 5) 조향각 스무딩 먼저 수행
        #    → 스무딩된 값으로 속도를 계산해야 코너 진입 전 과가속을 막을 수 있음
        steering = self._smooth_steering(raw_steering)

        # 6) 스무딩된 조향각 기반으로 목표 속도 계산
        target_speed = self._calc_speed(
            waypoints,
            self._nearest_idx,
            steering,
            abs(current_speed)
        )

        self._prev_steering = steering

        return steering, target_speed

    # ── 내부 계산 ─────────────────────────────────────────────────────────────

    def _calc_lookahead(self, speed: float) -> float:
        """속도 비례 lookahead 거리 계산"""
        ld = self.min_lookahead + self.lookahead_gain * speed
        return float(np.clip(ld, self.min_lookahead, self.max_lookahead))

    def _calc_steering(self,
                       x: float,
                       y: float,
                       heading: float,
                       target: np.ndarray,
                       lookahead_dist: float) -> float:
        """
        Pure Pursuit 조향각 공식

        steering = arctan(2 * L * sin(alpha) / ld)
        L: 축거, alpha: 목표점까지 각도, ld: lookahead 거리
        """
        dx = target[0] - x
        dy = target[1] - y

        target_angle = np.arctan2(dy, dx)
        alpha = _normalize_angle(target_angle - heading)

        actual_dist = np.sqrt(dx**2 + dy**2)
        if actual_dist < 1e-6:
            return 0.0

        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha),
                              actual_dist)

        return float(np.clip(steering, -self.max_steering, self.max_steering))

    def _calc_speed(self,
                waypoints: np.ndarray,
                nearest_idx: int,
                steering: float,
                current_speed: float) -> float:
        """
        속도 계산 전용 로직
    
        조향용 lookahead와 분리해서,
        속도는 더 먼 전방 곡률을 보고 미리 감속한다.
        """
        base_window  = PURE_PURSUIT_CONFIG['lookahead_window_base']
        speed_window = PURE_PURSUIT_CONFIG['lookahead_window_speed_scale']
    
        # 현재 속도가 빠를수록 더 멀리 보고 감속
        window = int(base_window + speed_window * abs(current_speed))
    
        curvature = self._estimate_max_lookahead_curvature(
            waypoints, nearest_idx, window
        )
    
        # 1) 전방 곡률 기반 감속
        curvature_factor = 1.0 / (1.0 + self.curvature_gain * curvature)
    
        # 2) 현재 조향각 기반 감속
        #    코너를 도는 중인데 곡률이 낮아졌다고 바로 재가속하는 것을 방지
        steering_ratio = abs(steering) / self.max_steering
        steering_gain = PURE_PURSUIT_CONFIG.get('steering_speed_gain', 2.0)
        steering_factor = 1.0 / (1.0 + steering_gain * steering_ratio)
    
        speed = self.max_speed * curvature_factor * steering_factor
    
        return float(np.clip(speed, self.min_speed, self.max_speed))

    def _estimate_max_lookahead_curvature(self,
                                      waypoints: np.ndarray,
                                      nearest_idx: int,
                                      window: int) -> float:
        """
        속도 감속용 전방 최대 곡률 계산
    
        window는 길게 가져가되,
        sample step은 작게 고정해서 멀리 있는 커브도 촘촘하게 감지한다.
        """
        N = len(waypoints)
        if N < 3:
            return 0.0
    
        curvatures = []
    
        # 기존처럼 window // 8로 두면 window가 커질수록 샘플링이 거칠어짐
        step = PURE_PURSUIT_CONFIG.get('curvature_sample_step', 2)
    
        for offset in range(0, window, 1):
            i0 = (nearest_idx + offset) % N
            i1 = (nearest_idx + offset + step) % N
            i2 = (nearest_idx + offset + step * 2) % N
    
            p0, p1, p2 = waypoints[i0], waypoints[i1], waypoints[i2]
    
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p2 - p0)
    
            if a < 1e-6 or b < 1e-6 or c < 1e-6:
                continue
    
            area = abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1])
                - (p2[0] - p0[0]) * (p1[1] - p0[1])
            ) / 2.0
    
            curvatures.append(4.0 * area / (a * b * c))
    
        return float(np.max(curvatures)) if curvatures else 0.0

    def _smooth_steering(self,
                         steering: float,
                         alpha: float = PURE_PURSUIT_CONFIG['smooth_alpha']) -> float:
        """
        조향각 스무딩 (이전 값과 가중 평균)
        alpha가 작을수록 부드럽게 변화
        """
        smoothed = alpha * steering + (1.0 - alpha) * self._prev_steering
        return float(np.clip(smoothed, -self.max_steering, self.max_steering))

    @property
    def nearest_idx(self) -> int:
        """마지막 계산에서의 가장 가까운 웨이포인트 인덱스"""
        return self._nearest_idx


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _normalize_angle(angle: float) -> float:
    """각도를 -pi ~ pi 범위로 정규화"""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


# ── 메인 (테스트용) ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 원형 트랙 테스트
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    radius = 5.0
    waypoints = np.column_stack([radius * np.cos(t), radius * np.sin(t)])

    controller = PurePursuitController()

    x, y, heading = 5.0, 0.0, np.pi / 2
    speed = 1.0
    dt = 0.01

    print(f'{"═"*55}')
    print(f'Pure Pursuit 테스트 (원형 트랙, 반지름 {radius}m)')
    print(f'{"═"*55}')

    errors = []
    for step in range(2000):
        steering, target_speed = controller.compute(x, y, heading, speed,
                                                    waypoints)

        speed   = target_speed
        x      += speed * np.cos(heading) * dt
        y      += speed * np.sin(heading) * dt
        heading += (speed / WHEELBASE) * np.tan(steering) * dt
        heading  = _normalize_angle(heading)

        dist_from_center = abs(np.sqrt(x**2 + y**2) - radius)
        errors.append(dist_from_center)

        if step % 500 == 0:
            print(f'  스텝 {step:5d} | '
                  f'위치: ({x:.2f}, {y:.2f}) | '
                  f'조향: {steering:+.3f} rad | '
                  f'속도: {target_speed:.2f} m/s | '
                  f'오차: {dist_from_center:.4f} m')

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    print(f'{"─"*55}')
    print(f'  평균 오차: {avg_error:.4f} m')
    print(f'  최대 오차: {max_error:.4f} m')
    print(f'{"═"*55}')