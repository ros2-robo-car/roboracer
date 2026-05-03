"""
eval_node.py

학습된 Hybrid SAC 모델 평가 코드.
SAC는 주행 라인과 Pure Pursuit 기준 속도 배율을 선택하고,
Pure Pursuit는 선택된 라인을 따라가기 위한 조향각과 기준 속도를 계산한다.

종료 조건:
- collision       : 충돌 발생
- env_done        : f1tenth_gym 환경에서 done=True 반환
- forward_done    : ForwardProgressTracker 기준 목표 lap 수 완료
- no_progress     : 일정 구간 동안 waypoint 진행이 부족하여 강제 종료
- max_steps       : EVAL_CONFIG['max_steps'] 도달

출력:
- STEP_DEBUG_INTERVAL마다 현재 waypoint, SAC 출력값, 현재 line, cmd_speed, 실제 속도, 조향값 출력
- 각 episode 종료 시 [END] 로그로 종료 이유 표시
- episode별 평균 속도, 라인 오차, 라인 전환 횟수, 진행률 표시
"""

import os
import sys
from collections import Counter

import gym
import f110_gym
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config as cfg

from config import (
    ENV_CONFIG,
    OBS_CONFIG,
    LINE_CONFIG,
    MODEL_CONFIG,
    REWARD_CONFIG,
    PURE_PURSUIT_CONFIG,
    SPEED_MIN,
    SPEED_MAX,
    MODEL_SAVE_PATH,
)

from sac_model import SAC, build_observation, get_obs_dim
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


# ── 평가 / 진행 설정 ─────────────────────────────────────────────────────────
EVAL_CONFIG = cfg.EVAL_CONFIG

N_EVAL_EPISODES = int(EVAL_CONFIG.get('episodes', 5))
MAX_EVAL_STEPS = int(EVAL_CONFIG.get('max_steps', 20000))

# eval_node에서는 학습 reward 설정과 별도로 평가 lap 수를 지정할 수 있다.
MAX_LAPS = int(
    EVAL_CONFIG.get(
        'max_laps',
        REWARD_CONFIG.get('max_laps', 2),
    )
)

MAX_FORWARD_WP_JUMP = REWARD_CONFIG.get('max_forward_wp_jump', 30)

# step debug 출력 간격
# 예: 100이면 100 step, 200 step, 300 step... 마다 출력
STEP_DEBUG_INTERVAL = int(EVAL_CONFIG.get('step_debug_interval', 100))

# SAC / Pure Pursuit가 선택하는 "목표 속도"의 최소값
# SPEED_MIN은 브레이크 명령 하한값으로만 사용한다.
TARGET_SPEED_MIN = REWARD_CONFIG.get('target_speed_min', 0.5)

# train_node와 동일하게 action[1]을 pp_speed 기준 배율로 해석할 때의 범위
SAC_SPEED_SCALE_RANGE = REWARD_CONFIG.get('sac_speed_scale_range', 0.2)

# no-progress 종료 조건
NO_PROGRESS_CHECK_INTERVAL = REWARD_CONFIG.get('no_progress_check_interval', 100)
NO_PROGRESS_MIN_DELTA = REWARD_CONFIG.get('no_progress_min_delta', 1.0)
NO_PROGRESS_PATIENCE = REWARD_CONFIG.get('no_progress_patience', 3)


# ── 브레이크 설정 ─────────────────────────────────────────────────────────────
BRAKE_GAIN = REWARD_CONFIG.get('brake_gain', 0.5)


def apply_brake(current_speed: float, target_speed: float) -> float:
    """
    현재 속도가 목표 속도보다 높으면 command speed를 낮춰 감속 유도.
    SPEED_MIN은 실제 환경에 넣는 command speed의 하한값.
    """
    if current_speed > target_speed:
        diff = current_speed - target_speed
        cmd_speed = target_speed - BRAKE_GAIN * diff
        cmd_speed = max(cmd_speed, SPEED_MIN)
        return float(cmd_speed)

    return float(target_speed)


# ── 전처리 ────────────────────────────────────────────────────────────────────
def preprocess_lidar(obs_raw: dict) -> np.ndarray:
    lidar = obs_raw['scans'][0].astype(np.float32)

    lidar = np.where(
        np.isfinite(lidar),
        lidar,
        OBS_CONFIG['lidar_range_max'],
    )

    lidar = np.clip(
        lidar,
        OBS_CONFIG['lidar_range_min'],
        OBS_CONFIG['lidar_range_max'],
    )

    lidar = (
        (lidar - OBS_CONFIG['lidar_range_min'])
        / (OBS_CONFIG['lidar_range_max'] - OBS_CONFIG['lidar_range_min'])
    )

    size = OBS_CONFIG['lidar_size']
    step = max(1, len(lidar) // size)
    lidar = lidar[::step][:size]

    if len(lidar) < size:
        lidar = np.pad(lidar, (0, size - len(lidar)), constant_values=1.0)

    return lidar.astype(np.float32)


def compute_three_point_curvature(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    세 waypoint 점을 이용해 곡률을 계산한다.
    거의 직선이면 0에 가깝고, 급커브일수록 값이 커진다.
    """
    a = float(np.linalg.norm(p1 - p0))
    b = float(np.linalg.norm(p2 - p1))
    c = float(np.linalg.norm(p2 - p0))

    denom = a * b * c
    if denom < 1e-6:
        return 0.0

    v1 = p1 - p0
    v2 = p2 - p0

    # 2D cross product 크기. 삼각형 넓이 기반 곡률 계산에 사용한다.
    cross = abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
    curvature = 2.0 * cross / denom

    if not np.isfinite(curvature):
        return 0.0

    return float(curvature)


def compute_line_lookahead_curvatures(
    obs_raw: dict,
    waypoints_lines: list,
    mode: str = 'max',
    max_curvature: float = 1.5,
) -> np.ndarray:
    """
    각 후보 line별 전방 곡률 feature를 계산한다.

    OBS_CONFIG['curvature_use_pp_window']가 True이면 Pure Pursuit의 속도 감속용
    window 설정을 그대로 사용한다.

    window = lookahead_window_base + 현재속도 * lookahead_window_speed_scale

    Returns:
        np.ndarray shape = (num_lines,)
        각 값은 0~1로 정규화된 곡률이다.
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    current_speed = abs(float(obs_raw['linear_vels_x'][0]))
    position = np.array([x, y], dtype=np.float32)

    if OBS_CONFIG.get('curvature_use_pp_window', True):
        lookahead_window = int(
            PURE_PURSUIT_CONFIG.get('lookahead_window_base', 5)
            + current_speed * PURE_PURSUIT_CONFIG.get('lookahead_window_speed_scale', 2)
        )
        sample_step = int(PURE_PURSUIT_CONFIG.get('curvature_sample_step', 2))
    else:
        lookahead_window = int(OBS_CONFIG.get('curvature_lookahead_window', 30))
        sample_step = int(OBS_CONFIG.get('curvature_sample_step', 2))

    lookahead_window = max(lookahead_window, 3)
    sample_step = max(sample_step, 1)
    max_curvature = max(float(max_curvature), 1e-6)

    line_curvatures = []

    for waypoints in waypoints_lines:
        n = len(waypoints)
        nearest_idx = get_nearest_waypoint_idx(position, waypoints)

        curvatures = []

        for offset in range(0, lookahead_window, sample_step):
            i0 = (nearest_idx + offset) % n
            i1 = (nearest_idx + offset + sample_step) % n
            i2 = (nearest_idx + offset + 2 * sample_step) % n

            curvature = compute_three_point_curvature(
                waypoints[i0],
                waypoints[i1],
                waypoints[i2],
            )
            curvatures.append(curvature)

        if not curvatures:
            line_curvature = 0.0
        elif mode == 'mean':
            line_curvature = float(np.mean(curvatures))
        else:
            # 선제 감속 목적에는 lookahead 구간의 최대 곡률이 더 직접적이다.
            line_curvature = float(np.max(curvatures))

        line_curvature = float(np.clip(line_curvature / max_curvature, 0.0, 1.0))
        line_curvatures.append(line_curvature)

    return np.array(line_curvatures, dtype=np.float32)


def preprocess_obs(
    obs_raw: dict,
    waypoints_lines: list,
    num_lines: int,
) -> np.ndarray:
    lidar = preprocess_lidar(obs_raw)

    position = np.array(
        [float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])],
        dtype=np.float32,
    )
    heading = float(obs_raw['poses_theta'][0])
    speed = float(obs_raw['linear_vels_x'][0])

    use_line_curvature = OBS_CONFIG.get('use_line_curvature', False)
    line_curvatures = None

    if use_line_curvature:
        line_curvatures = compute_line_lookahead_curvatures(
            obs_raw,
            waypoints_lines,
            mode=OBS_CONFIG.get('curvature_mode', 'max'),
            max_curvature=OBS_CONFIG.get('curvature_max_value', 1.5),
        )

    return build_observation(
        lidar,
        position,
        heading,
        speed,
        waypoints_lines,
        num_lines,
        line_curvatures=line_curvatures,
        use_line_curvature=use_line_curvature,
    ).astype(np.float32)


# ── Forward Progress Tracker ──────────────────────────────────────────────────
class ForwardProgressTracker:
    """
    lap_counts를 사용하지 않고 reference waypoint index 변화량으로
    전진 진행량을 누적한다.
    """

    def __init__(
        self,
        reference_waypoints: np.ndarray,
        max_laps: int = 2,
        max_forward_jump: int = 30,
    ):
        self.reference_waypoints = reference_waypoints
        self.n_waypoints = len(reference_waypoints)
        self.max_laps = max_laps
        self.total_waypoints = self.n_waypoints * max_laps
        self.max_forward_jump = int(max_forward_jump)

        self.prev_idx = None
        self.forward_progress = 0.0
        self.ignored_jump_count = 0

    def reset_from_obs(self, obs_raw: dict):
        x = float(obs_raw['poses_x'][0])
        y = float(obs_raw['poses_y'][0])
        position = np.array([x, y], dtype=np.float32)

        self.prev_idx = get_nearest_waypoint_idx(
            position,
            self.reference_waypoints,
        )

        self.forward_progress = 1.0
        self.ignored_jump_count = 0

    def update(self, obs_raw: dict) -> tuple:
        """
        Returns:
            progress_score, progress_pct, forward_done, progress_delta
        """
        x = float(obs_raw['poses_x'][0])
        y = float(obs_raw['poses_y'][0])
        position = np.array([x, y], dtype=np.float32)

        nearest_idx = get_nearest_waypoint_idx(
            position,
            self.reference_waypoints,
        )

        if self.prev_idx is None:
            self.prev_idx = nearest_idx

        prev_progress = self.forward_progress
        delta = nearest_idx - self.prev_idx

        if delta < -self.n_waypoints / 2:
            delta += self.n_waypoints
        elif delta > self.n_waypoints / 2:
            delta -= self.n_waypoints

        if 0 < delta <= self.max_forward_jump:
            self.forward_progress += delta
        elif delta > self.max_forward_jump:
            self.ignored_jump_count += 1

        self.prev_idx = nearest_idx

        progress_delta = max(0.0, self.forward_progress - prev_progress)

        progress_score = int(
            min(
                max(self.forward_progress, 0.0),
                self.total_waypoints,
            )
        )

        progress_pct = (
            progress_score / self.total_waypoints * 100.0
            if self.total_waypoints > 0
            else 0.0
        )

        forward_done = progress_score >= self.total_waypoints

        return progress_score, progress_pct, forward_done, progress_delta


# ── 평가 지표 ─────────────────────────────────────────────────────────────────
class EvalMetrics:
    def __init__(self, reference_waypoints: np.ndarray):
        self.speeds = []
        self.line_deviations = []
        self.line_switches = 0
        self.total_steps = 0
        self.laps_completed = 0
        self.prev_line_idx = None

        self.reference_waypoints = reference_waypoints
        self.reference_len = len(reference_waypoints)
        self.total_progress_target = self.reference_len * MAX_LAPS

        self.best_progress_score = 1
        self.best_progress_pct = (
            1 / self.total_progress_target * 100.0
            if self.total_progress_target > 0
            else 0.0
        )
        self.ignored_jump_count = 0

    def update(
        self,
        obs_raw: dict,
        actual_speed: float,
        line_idx: int,
        waypoints_lines: list,
        progress_score: int,
        progress_pct: float,
        ignored_jump_count: int,
    ):
        self.speeds.append(abs(float(actual_speed)))
        self.laps_completed = int(obs_raw['lap_counts'][0])

        x = float(obs_raw['poses_x'][0])
        y = float(obs_raw['poses_y'][0])
        position = np.array([x, y], dtype=np.float32)

        line_idx = int(np.clip(line_idx, 0, len(waypoints_lines) - 1))
        waypoints = waypoints_lines[line_idx]
        nearest_idx = get_nearest_waypoint_idx(position, waypoints)
        deviation = float(np.linalg.norm(waypoints[nearest_idx] - position))
        self.line_deviations.append(deviation)

        if progress_score > self.best_progress_score:
            self.best_progress_score = progress_score
            self.best_progress_pct = progress_pct

        self.ignored_jump_count = ignored_jump_count

        if self.prev_line_idx is not None and line_idx != self.prev_line_idx:
            self.line_switches += 1

        self.prev_line_idx = line_idx
        self.total_steps += 1

    def summary(self, episode: int, end_reason: str) -> dict:
        avg_speed = float(np.mean(self.speeds)) if self.speeds else 0.0
        avg_deviation = (
            float(np.mean(self.line_deviations))
            if self.line_deviations
            else 0.0
        )

        print(f'\n{"─" * 45}')
        print(f'에피소드 {episode + 1} 결과')
        print(f'{"─" * 45}')
        print(f'종료 이유     : {end_reason}')
        print(f'완주 lap      : {self.laps_completed}')
        print(f'평균 속도     : {avg_speed:.3f} m/s')
        print(f'평균 라인 오차: {avg_deviation:.4f} m')
        print(f'라인 전환 횟수: {self.line_switches}')
        print(f'총 스텝       : {self.total_steps}')
        print(
            f'wp 진행       : {self.best_progress_score}/'
            f'{self.total_progress_target} '
            f'({self.best_progress_pct:.1f}%)'
        )
        print(f'ignored jump  : {self.ignored_jump_count}')
        print(f'{"─" * 45}\n')

        return {
            'end_reason': end_reason,
            'laps_completed': self.laps_completed,
            'avg_speed': avg_speed,
            'avg_deviation': avg_deviation,
            'line_switches': self.line_switches,
            'total_steps': self.total_steps,
            'progress_score': self.best_progress_score,
            'progress_pct': self.best_progress_pct,
            'ignored_jump_count': self.ignored_jump_count,
        }


# ── 레이싱 라인 로드 ──────────────────────────────────────────────────────────
def load_racing_lines() -> list:
    csv_path = LINE_CONFIG['centerline_csv']

    if os.path.exists(csv_path):
        print(f'centerline CSV 로드: {csv_path}')
        wp = load_waypoints(
            centerline_path=csv_path,
            num_lines=LINE_CONFIG['num_lines'],
            line_spacing=LINE_CONFIG['line_spacing'],
            width_fraction=LINE_CONFIG.get('line_width_fraction', 0.60),
        )
    else:
        print('CSV 없음 → 맵 이미지에서 centerline 추출')
        wp = load_waypoints(
            map_path=LINE_CONFIG['map_path'],
            map_ext=LINE_CONFIG['map_ext'],
            num_lines=LINE_CONFIG['num_lines'],
            line_spacing=LINE_CONFIG['line_spacing'],
            width_fraction=LINE_CONFIG.get('line_width_fraction', 0.60),
        )

    print(f'라인 {len(wp["lines"])}개 로드 완료')
    return wp['lines']


def make_init_pose(waypoints_lines: list) -> np.ndarray:
    num_lines = len(waypoints_lines)

    start_line_idx = LINE_CONFIG.get('start_line_idx', None)
    if start_line_idx is None:
        start_line_idx = num_lines // 2

    start_wp_idx = LINE_CONFIG.get('start_wp_idx', 0)
    lookahead_idx = LINE_CONFIG.get('start_lookahead_idx', 5)

    reference_line = waypoints_lines[start_line_idx]
    n = len(reference_line)

    p0 = reference_line[start_wp_idx % n]
    p1 = reference_line[(start_wp_idx + lookahead_idx) % n]

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    heading = float(np.arctan2(dy, dx))

    init_poses = np.array([
        [float(p0[0]), float(p0[1]), heading]
    ])

    print(
        f'시작 pose 자동 설정 | '
        f'line: {start_line_idx}, '
        f'wp: {start_wp_idx}, '
        f'x: {p0[0]:.3f}, y: {p0[1]:.3f}, '
        f'theta: {heading:.3f}'
    )

    return init_poses


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(path: str) -> SAC:
    if not os.path.exists(path):
        raise FileNotFoundError(f'모델 파일 없음: {path}')

    checkpoint = torch.load(path, map_location='cpu')

    model_config = dict(MODEL_CONFIG)
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_config.update(checkpoint['model_config'])
        MODEL_CONFIG.update(checkpoint['model_config'])

    # checkpoint에 곡률 feature 사용 여부가 저장되어 있으면 평가 observation도 그 기준으로 맞춘다.
    if 'use_line_curvature' in model_config:
        OBS_CONFIG['use_line_curvature'] = bool(model_config['use_line_curvature'])

    num_lines = int(model_config.get('num_lines', LINE_CONFIG['num_lines']))

    saved_obs_dim = model_config.get('obs_dim', None)
    if saved_obs_dim is not None:
        obs_dim = int(saved_obs_dim)
    else:
        obs_dim = get_obs_dim(
            OBS_CONFIG['lidar_size'],
            num_lines,
            use_line_curvature=OBS_CONFIG.get('use_line_curvature', False),
        )

    model = SAC(
        obs_dim,
        model_config.get('action_dim', MODEL_CONFIG['action_dim']),
        model_config.get('hidden_dims', MODEL_CONFIG['hidden_dims']),
        num_lines=num_lines,
    )

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f'모델 로드 완료: {path}')
    print(
        f'obs_dim: {obs_dim} | '
        f'num_lines: {num_lines} | '
        f'use_line_curvature: {OBS_CONFIG.get("use_line_curvature", False)}'
    )

    return model


# ── line 상태 계산 ───────────────────────────────────────────────────────────
def get_current_line_idx(
    obs_raw: dict,
    waypoints_lines: list,
) -> int:
    """
    차량의 현재 위치가 가장 가까운 line index를 계산한다.

    sac_line      : SAC가 선택한 목표 line
    current_line  : 실제 차량 위치 기준으로 가장 가까운 line
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    position = np.array([x, y], dtype=np.float32)

    min_dist = float('inf')
    current_line_idx = 0

    for i, waypoints in enumerate(waypoints_lines):
        nearest_idx = get_nearest_waypoint_idx(position, waypoints)
        dist = float(np.linalg.norm(waypoints[nearest_idx] - position))

        if dist < min_dist:
            min_dist = dist
            current_line_idx = i

    return int(current_line_idx)


# ── action → 환경 제어 ────────────────────────────────────────────────────────
def action_to_env(
    action: np.ndarray,
    obs_raw: dict,
    model: SAC,
    waypoints_lines: list,
    controller: PurePursuitController,
):
    """
    train_node와 동일하게 action[1]을 절대 속도가 아니라
    Pure Pursuit가 계산한 pp_speed 기준 배율로 해석한다.

    speed_scale = 1 + SAC_SPEED_SCALE_RANGE * action[1]
    """
    line_idx = model.action_to_line_index(action)
    line_idx = int(np.clip(line_idx, 0, len(waypoints_lines) - 1))
    waypoints = waypoints_lines[line_idx]

    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    heading = float(obs_raw['poses_theta'][0])
    current_speed = float(obs_raw['linear_vels_x'][0])

    steering, pp_speed = controller.compute(
        x,
        y,
        heading,
        current_speed,
        waypoints,
    )

    speed_action = float(np.clip(action[1], -1.0, 1.0))
    speed_scale = 1.0 + SAC_SPEED_SCALE_RANGE * speed_action

    target_speed = pp_speed * speed_scale
    target_speed = float(np.clip(target_speed, TARGET_SPEED_MIN, SPEED_MAX))

    cmd_speed = apply_brake(current_speed, target_speed)

    return steering, cmd_speed, line_idx, target_speed, pp_speed


def print_step_debug(
    episode: int,
    step: int,
    progress_score: int,
    total_waypoints: int,
    line_idx: int,
    current_line_idx: int,
    action: np.ndarray,
    cmd_speed: float,
    actual_speed: float,
    steering: float,
):
    """
    STEP_DEBUG_INTERVAL마다 필요한 값만 출력한다.

    출력 항목:
    - 현재 waypoint 진행량
    - SAC line 출력
    - 현재 실제 차량 위치 기준 line
    - SAC speed 출력
    - 실제 env에 들어간 cmd_speed
    - 현재 자동차 속도
    - 바퀴 조향값
    """
    speed_action = float(np.clip(action[1], -1.0, 1.0))

    print(
        f'[STEP_DEBUG] '
        f'ep={episode + 1} | '
        f'step={step} | '
        f'wp={progress_score}/{total_waypoints} | '
        f'sac_line={line_idx} | '
        f'current_line={current_line_idx} | '
        f'sac_speed={speed_action:+.3f} | '
        f'cmd_spd={cmd_speed:.2f} | '
        f'actual_speed={actual_speed:.2f} | '
        f'steering={steering:+.3f}'
    )


def format_end_reason(
    end_reason: str,
    progress_score: int,
    total_waypoints: int,
    progress_pct: float,
    line_idx: int,
    actual_speed: float,
    cmd_speed: float,
) -> str:
    return (
        f'[END] reason={end_reason} | '
        f'wp={progress_score}/{total_waypoints} '
        f'({progress_pct:.1f}%) | '
        f'line={line_idx} | '
        f'actual_speed={actual_speed:.2f} | '
        f'cmd_speed={cmd_speed:.2f}'
    )


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print(
        f'\n평가 시작 | 모델: {MODEL_SAVE_PATH} | '
        f'에피소드: {N_EVAL_EPISODES} | '
        f'max_steps: {MAX_EVAL_STEPS} | '
        f'max_laps: {MAX_LAPS}\n'
    )
    print(f'SAC_SPEED_SCALE_RANGE: {SAC_SPEED_SCALE_RANGE}')
    print(f'STEP_DEBUG_INTERVAL: {STEP_DEBUG_INTERVAL}')
    print(f"line curvature feature: {OBS_CONFIG.get('use_line_curvature', False)}")
    print(f'브레이크 명령 하한 SPEED_MIN: {SPEED_MIN}')
    print(f'SAC 목표 속도 하한 TARGET_SPEED_MIN: {TARGET_SPEED_MIN}')
    print(f'속도 상한 SPEED_MAX: {SPEED_MAX}')
    print(f'브레이크 gain: {BRAKE_GAIN}\n')

    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

    waypoints_lines = load_racing_lines()
    num_lines = MODEL_CONFIG.get('num_lines', LINE_CONFIG['num_lines'])

    progress_reference_line = waypoints_lines[num_lines // 2]

    model = load_model(MODEL_SAVE_PATH)

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=TARGET_SPEED_MIN,
    )

    init_poses = make_init_pose(waypoints_lines)

    all_results = []

    for episode in range(N_EVAL_EPISODES):
        metrics = EvalMetrics(progress_reference_line)

        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        progress_tracker = ForwardProgressTracker(
            progress_reference_line,
            max_laps=MAX_LAPS,
            max_forward_jump=MAX_FORWARD_WP_JUMP,
        )
        progress_tracker.reset_from_obs(obs_raw)

        progress_score = 1
        progress_pct = (
            1 / progress_tracker.total_waypoints * 100.0
            if progress_tracker.total_waypoints > 0
            else 0.0
        )

        end_reason = 'max_steps'

        last_line_idx = -1
        last_actual_speed = 0.0
        last_cmd_speed = 0.0

        progress_window_sum = 0.0
        progress_window_steps = 0
        no_progress_bad_count = 0

        for step_in_ep in range(MAX_EVAL_STEPS):
            action = model.select_action(obs, training=False)

            steering, cmd_speed, line_idx, target_speed, pp_speed = action_to_env(
                action,
                obs_raw,
                model,
                waypoints_lines,
                controller,
            )

            env_action = np.array([[steering, cmd_speed]], dtype=np.float32)
            next_obs_raw, _, done, _ = env.step(env_action)
            #env.render()

            current_collision = bool(next_obs_raw['collisions'][0])
            actual_speed = abs(float(next_obs_raw['linear_vels_x'][0]))

            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            current_step = step_in_ep + 1

            # 100 step마다 현재 waypoint, SAC 출력, 현재 line, cmd_speed, 실제 속도, 조향값 출력
            if (
                STEP_DEBUG_INTERVAL > 0
                and current_step % STEP_DEBUG_INTERVAL == 0
            ):
                current_line_idx = get_current_line_idx(
                    next_obs_raw,
                    waypoints_lines,
                )

                print_step_debug(
                    episode=episode,
                    step=current_step,
                    progress_score=progress_score,
                    total_waypoints=progress_tracker.total_waypoints,
                    line_idx=line_idx,
                    current_line_idx=current_line_idx,
                    action=action,
                    cmd_speed=cmd_speed,
                    actual_speed=actual_speed,
                    steering=steering,
                )

            metrics.update(
                next_obs_raw,
                actual_speed,
                line_idx,
                waypoints_lines,
                progress_score,
                progress_pct,
                progress_tracker.ignored_jump_count,
            )

            last_line_idx = line_idx
            last_actual_speed = actual_speed
            last_cmd_speed = cmd_speed

            # no-progress 검사
            progress_window_sum += progress_delta
            progress_window_steps += 1

            no_progress_done = False
            if progress_window_steps >= NO_PROGRESS_CHECK_INTERVAL:
                if progress_window_sum < NO_PROGRESS_MIN_DELTA:
                    no_progress_bad_count += 1
                else:
                    no_progress_bad_count = 0

                progress_window_sum = 0.0
                progress_window_steps = 0

                if no_progress_bad_count >= NO_PROGRESS_PATIENCE:
                    no_progress_done = True

            # 다음 obs 준비
            obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            obs_raw = next_obs_raw

            # 종료 이유 판정
            if current_collision:
                end_reason = 'collision'
                break

            if no_progress_done:
                end_reason = 'no_progress'
                break

            if forward_done:
                end_reason = 'forward_done'
                break

            if done:
                end_reason = 'env_done'
                break

        print(
            format_end_reason(
                end_reason=end_reason,
                progress_score=progress_score,
                total_waypoints=progress_tracker.total_waypoints,
                progress_pct=progress_pct,
                line_idx=last_line_idx,
                actual_speed=last_actual_speed,
                cmd_speed=last_cmd_speed,
            )
        )

        all_results.append(metrics.summary(episode, end_reason))

    print(f'\n{"═" * 45}')
    print(f'전체 {N_EVAL_EPISODES} 에피소드 평균')
    print(f'{"═" * 45}')

    if all_results:
        end_reason_counts = Counter([r['end_reason'] for r in all_results])

        print('종료 이유 분포:')
        for reason, count in end_reason_counts.items():
            print(f'  {reason}: {count}')

        print(
            f'평균 완주 lap  : '
            f'{np.mean([r["laps_completed"] for r in all_results]):.2f}'
        )
        print(
            f'평균 속도      : '
            f'{np.mean([r["avg_speed"] for r in all_results]):.3f} m/s'
        )
        print(
            f'평균 라인 오차 : '
            f'{np.mean([r["avg_deviation"] for r in all_results]):.4f}'
        )
        print(
            f'평균 라인 전환 : '
            f'{np.mean([r["line_switches"] for r in all_results]):.2f}'
        )
        print(
            f'평균 스텝 수   : '
            f'{np.mean([r["total_steps"] for r in all_results]):.2f}'
        )
        print(
            f'평균 wp 진행률: '
            f'{np.mean([r["progress_pct"] for r in all_results]):.1f}%'
        )
        print(
            f'최대 wp 진행률: '
            f'{np.max([r["progress_pct"] for r in all_results]):.1f}%'
        )
        print(
            f'평균 ignored jump: '
            f'{np.mean([r["ignored_jump_count"] for r in all_results]):.2f}'
        )

    print(f'{"═" * 45}\n')

    env.close()


if __name__ == '__main__':
    main()