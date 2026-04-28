"""
pure_pursuit.py
───────────────
Pure Pursuit 경로 추종 컨트롤러

입력: 현재 차량 상태 (x, y, heading) + 웨이포인트 배열
출력: 조향각 (rad), 속도 (m/s)

waypoint_loader.py 와 함께 사용
"""

import numpy as np
from waypoint_loader import get_nearest_waypoint_idx, get_lookahead_point


# ── 차량 파라미터 (F1tenth 기본값) ────────────────────────────────────────────
WHEELBASE = 0.3302   # 축거 (lf + lr = 0.15875 + 0.17145)

# ── Pure Pursuit 파라미터 ─────────────────────────────────────────────────────
MIN_LOOKAHEAD   = 0.5    # 최소 lookahead 거리 (m)
MAX_LOOKAHEAD   = 2.0    # 최대 lookahead 거리 (m)
LOOKAHEAD_GAIN  = 0.3    # 속도 대비 lookahead 비율 (s)

# ── 속도 파라미터 ─────────────────────────────────────────────────────────────
MAX_SPEED       = 3.0    # 최대 속도 (m/s)
MIN_SPEED       = 0.5    # 최소 속도 (m/s)
CURVATURE_GAIN  = 2.0    # 곡률 대비 감속 비율

# ── 조향 제한 ─────────────────────────────────────────────────────────────────
MAX_STEERING    = 0.4189  # 최대 조향각 (rad, 약 24도)


class PurePursuitController:
    """
    Pure Pursuit 경로 추종 컨트롤러

    사용법:
        controller = PurePursuitController()
        steering, speed = controller.compute(x, y, heading, speed, waypoints)
    """

    def __init__(self,
                 wheelbase:     float = WHEELBASE,
                 min_lookahead: float = MIN_LOOKAHEAD,
                 max_lookahead: float = MAX_LOOKAHEAD,
                 lookahead_gain: float = LOOKAHEAD_GAIN,
                 max_speed:     float = MAX_SPEED,
                 min_speed:     float = MIN_SPEED,
                 curvature_gain: float = CURVATURE_GAIN,
                 max_steering:  float = MAX_STEERING):

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
        target_point, target_idx = get_lookahead_point(
            position, waypoints, lookahead_dist, self._nearest_idx
        )

        # 4) 조향각 계산
        steering = self._calc_steering(x, y, heading, target_point,
                                        lookahead_dist)

        # 5) 목표 속도 계산 (곡률 기반)
        target_speed = self._calc_speed(waypoints, self._nearest_idx,
                                         steering)

        # 6) 조향각 스무딩 (급격한 변화 억제)
        steering = self._smooth_steering(steering)

        self._prev_steering = steering

        return steering, target_speed

    def _calc_lookahead(self, speed: float) -> float:
        """속도 비례 lookahead 거리 계산"""
        ld = self.min_lookahead + self.lookahead_gain * speed
        return np.clip(ld, self.min_lookahead, self.max_lookahead)

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
        # 목표점까지의 벡터
        dx = target[0] - x
        dy = target[1] - y

        # 목표점의 절대 각도
        target_angle = np.arctan2(dy, dx)

        # 차량 기준 상대 각도
        alpha = _normalize_angle(target_angle - heading)

        # 실제 거리 (lookahead_dist 대신 실측 사용)
        actual_dist = np.sqrt(dx**2 + dy**2)
        if actual_dist < 1e-6:
            return 0.0

        # Pure Pursuit 공식
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha),
                              actual_dist)

        # 조향각 제한
        return np.clip(steering, -self.max_steering, self.max_steering)

    def _calc_speed(self,
                    waypoints: np.ndarray,
                    nearest_idx: int,
                    steering: float) -> float:
        """
        곡률 기반 속도 계산
        - 조향각이 크면 (커브) → 감속
        - 조향각이 작으면 (직선) → 가속
        """
        # 조향각 기반 감속
        abs_steering = abs(steering)
        steering_ratio = abs_steering / self.max_steering   # 0~1

        # 곡률도 추가 참고 (앞쪽 경로의 곡률)
        curvature = self._estimate_curvature(waypoints, nearest_idx)
        curvature_factor = 1.0 / (1.0 + self.curvature_gain * curvature)

        # 최종 속도: 곡률과 조향각 모두 고려
        speed = self.max_speed * curvature_factor * (1.0 - 0.5 * steering_ratio)

        return np.clip(speed, self.min_speed, self.max_speed)

    def _estimate_curvature(self,
                            waypoints: np.ndarray,
                            idx: int,
                            window: int = 10) -> float:
        """
        앞쪽 웨이포인트들의 곡률 추정
        세 점으로 외접원의 곡률을 계산
        """
        N = len(waypoints)
        if N < 3:
            return 0.0

        # 현재, 중간, 먼 점 선택
        i0 = idx
        i1 = (idx + window // 2) % N
        i2 = (idx + window) % N

        p0 = waypoints[i0]
        p1 = waypoints[i1]
        p2 = waypoints[i2]

        # 세 점으로 곡률 계산 (Menger curvature)
        # k = 4 * area / (|a| * |b| * |c|)
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)

        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0

        # 삼각형 넓이 (외적)
        area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                   (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2.0

        curvature = 4.0 * area / (a * b * c)
        return curvature

    def _smooth_steering(self, steering: float,
                         alpha: float = 0.3) -> float:
        """
        조향각 스무딩 (이전 값과 가중 평균)
        alpha가 작을수록 부드럽게 변화
        """
        smoothed = alpha * steering + (1 - alpha) * self._prev_steering
        return np.clip(smoothed, -self.max_steering, self.max_steering)

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

    # 시뮬레이션: 차량이 트랙을 따라가는지 확인
    x, y, heading = 5.0, 0.0, np.pi / 2   # 시작: (5,0), 위를 향함
    speed = 1.0
    dt = 0.01

    print(f'{"═"*55}')
    print(f'Pure Pursuit 테스트 (원형 트랙, 반지름 {radius}m)')
    print(f'{"═"*55}')

    errors = []
    for step in range(2000):
        steering, target_speed = controller.compute(x, y, heading, speed,
                                                     waypoints)

        # 간단한 자전거 모델로 차량 업데이트
        speed = target_speed
        x       += speed * np.cos(heading) * dt
        y       += speed * np.sin(heading) * dt
        heading += (speed / WHEELBASE) * np.tan(steering) * dt
        heading  = _normalize_angle(heading)

        # 트랙 중심과의 거리 오차
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