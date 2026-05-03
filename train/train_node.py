"""
train_node_stable_split_entropy.py

Hybrid SAC 안정화 버전.

"""

import os
import sys
import random
from collections import deque

import gym
import f110_gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import (
    ENV_CONFIG,
    OBS_CONFIG,
    LINE_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    REWARD_CONFIG,
    PURE_PURSUIT_CONFIG,
    SPEED_MIN,
    SPEED_MAX,
    MODEL_SAVE_PATH,
)

from sac_model import SAC, build_observation, get_obs_dim, encode_action
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


# ── 브레이크 설정 ─────────────────────────────────────────────────────────────
BRAKE_GAIN = REWARD_CONFIG.get('brake_gain', 0.5)

# SPEED_MIN은 환경에 넣는 command speed의 하한값으로 사용한다.
# SAC/Pure Pursuit가 선택하는 목표 속도의 하한은 TARGET_SPEED_MIN으로 분리한다.
TARGET_SPEED_MIN = REWARD_CONFIG.get('target_speed_min', 0.5)

# action[1]을 pp_speed 기준 배율로 해석할 때의 범위.
# 예: 0.2이면 action[1] = -1 → 0.8배, 0 → 1.0배, +1 → 1.2배
SAC_SPEED_SCALE_RANGE = REWARD_CONFIG.get('sac_speed_scale_range', 0.2)

# warmup buffer에 넣는 speed scale action의 랜덤 범위.
# 예: 0.2이면 warmup action[1]은 [-0.2, 0.2]에서 샘플링.
WARMUP_SPEED_ACTION_RANGE = REWARD_CONFIG.get('warmup_speed_action_range', 0.2)

# 학습 중 탐색 noise. 기존 0.05보다 보수적으로 0.02 권장.
SPEED_ACTION_NOISE_STD = TRAIN_CONFIG.get(
    'speed_action_noise_std',
    REWARD_CONFIG.get('speed_action_noise_std', 0.02),
)


# ── 체크포인트 설정 ───────────────────────────────────────────────────────────
NUM_CHECKPOINTS = REWARD_CONFIG['num_checkpoints']
CHECKPOINT_ARRIVAL_REWARD = REWARD_CONFIG['checkpoint_arrival']
SPEED_REWARD_SCALE = REWARD_CONFIG['speed_reward_scale']
BASELINE_STEPS = REWARD_CONFIG.get('baseline_steps', 500)
WARMUP_BASELINE_MIN_SAMPLES = REWARD_CONFIG.get('warmup_baseline_min_samples', 1)
WARMUP_BASELINE_MULTIPLIER = REWARD_CONFIG.get('warmup_baseline_multiplier', 1.0)


# ── 충돌 curriculum 설정 ─────────────────────────────────────────────────────
COLLISION_PENALTY_START = REWARD_CONFIG['collision_penalty_start']
COLLISION_PENALTY_END = REWARD_CONFIG['collision_penalty_end']
COLLISION_CURRICULUM_EPISODES = REWARD_CONFIG['collision_curriculum_episodes']


# ── Forward progress 설정 ────────────────────────────────────────────────────
MAX_LAPS = REWARD_CONFIG.get('max_laps', 2)
MAX_FORWARD_WP_JUMP = REWARD_CONFIG.get('max_forward_wp_jump', 30)
WAYPOINT_PROGRESS_REWARD = REWARD_CONFIG.get('waypoint_progress_reward', 1.0)


# ── No progress penalty 설정 ─────────────────────────────────────────────────
NO_PROGRESS_CHECK_INTERVAL = REWARD_CONFIG.get('no_progress_check_interval', 50)
NO_PROGRESS_MIN_DELTA = REWARD_CONFIG.get('no_progress_min_delta', 1.0)
NO_PROGRESS_PENALTY = REWARD_CONFIG.get('no_progress_penalty', -1.0)
NO_PROGRESS_PATIENCE = REWARD_CONFIG.get('no_progress_patience', 4)
NO_PROGRESS_TERMINAL_PENALTY = REWARD_CONFIG.get('no_progress_terminal_penalty', -50.0)


# ── Timeout penalty 설정 ──────────────────────────────────────────────────────
TIMEOUT_PENALTY_SCALE = REWARD_CONFIG.get('timeout_penalty_scale', -200.0)
TIMEOUT_FIXED_PENALTY = REWARD_CONFIG.get('timeout_fixed_penalty', -50.0)


# ── Steering penalty 설정 ────────────────────────────────────────────────────
STEER_SPEED_THRESHOLD = REWARD_CONFIG.get('steer_speed_threshold', 8.0)
STEER_DEADZONE = REWARD_CONFIG.get('steer_deadzone', 0.25)
STEER_PENALTY = REWARD_CONFIG.get('steer_penalty', 0.2)


# ── 학습 안정화 설정 ─────────────────────────────────────────────────────────
REWARD_CLAMP_MIN = REWARD_CONFIG.get('reward_clamp_min', -500.0)
REWARD_CLAMP_MAX = REWARD_CONFIG.get('reward_clamp_max', 200.0)
INVALID_OBS_PENALTY_SCALE = REWARD_CONFIG.get('invalid_obs_penalty_scale', 2.0)
GRAD_CLIP_NORM = TRAIN_CONFIG.get('grad_clip_norm', 1.0)


def get_collision_penalty(episode: int) -> float:
    """에피소드에 따라 충돌 페널티를 점진적으로 키운다."""
    if episode >= COLLISION_CURRICULUM_EPISODES:
        return COLLISION_PENALTY_END

    ratio = episode / COLLISION_CURRICULUM_EPISODES
    penalty = COLLISION_PENALTY_START + ratio * (
        COLLISION_PENALTY_END - COLLISION_PENALTY_START
    )
    return float(penalty)


def build_checkpoint_indices(n_waypoints: int, num_checkpoints: int = None) -> list:
    if num_checkpoints is None:
        num_checkpoints = NUM_CHECKPOINTS

    step = max(n_waypoints // num_checkpoints, 1)
    return [(i + 1) * step % n_waypoints for i in range(num_checkpoints)]


class CheckpointTracker:
    """체크포인트 통과를 추적하고, 구간 소요 스텝 수를 계산한다."""

    def __init__(self, n_waypoints: int):
        self.checkpoint_indices = build_checkpoint_indices(n_waypoints)
        self.n_waypoints = n_waypoints
        self.next_checkpoint = 0
        self.segment_steps = 0
        self.checkpoints_passed = 0

    def reset(self):
        self.next_checkpoint = 0
        self.segment_steps = 0
        self.checkpoints_passed = 0

    def tick(self):
        self.segment_steps += 1

    def check(self, nearest_idx: int) -> tuple:
        """
        Returns:
            passed, segment_steps, is_last_checkpoint, passed_checkpoint_idx
        """
        if self.next_checkpoint >= len(self.checkpoint_indices):
            return False, 0, False, -1

        target_idx = self.checkpoint_indices[self.next_checkpoint]
        threshold = max(self.n_waypoints // 20, 3)

        forward_dist = (nearest_idx - target_idx) % self.n_waypoints
        passed = forward_dist <= threshold and forward_dist > 0

        if not passed:
            return False, 0, False, -1

        steps = self.segment_steps
        self.segment_steps = 0

        passed_checkpoint_idx = self.next_checkpoint
        self.checkpoints_passed += 1
        is_last = self.next_checkpoint == len(self.checkpoint_indices) - 1
        self.next_checkpoint += 1

        return True, steps, is_last, passed_checkpoint_idx


class WarmupCheckpointBaseline:
    """checkpoint 구간별 warmup segment_steps 평균을 저장한다."""

    def __init__(
        self,
        num_checkpoints: int,
        fallback_steps: float,
        min_samples: int = 1,
        multiplier: float = 1.0,
    ):
        self.num_checkpoints = int(num_checkpoints)
        self.fallback_steps = float(fallback_steps)
        self.min_samples = int(min_samples)
        self.multiplier = float(multiplier)
        self.samples = [[] for _ in range(self.num_checkpoints)]

    def add(self, checkpoint_idx: int, segment_steps: int):
        if checkpoint_idx is None:
            return

        checkpoint_idx = int(checkpoint_idx)
        if checkpoint_idx < 0 or checkpoint_idx >= self.num_checkpoints:
            return

        if segment_steps is None or segment_steps <= 0:
            return

        self.samples[checkpoint_idx].append(float(segment_steps))

    def ready(self, checkpoint_idx: int) -> bool:
        checkpoint_idx = int(checkpoint_idx)
        if checkpoint_idx < 0 or checkpoint_idx >= self.num_checkpoints:
            return False
        return len(self.samples[checkpoint_idx]) >= self.min_samples

    def get(self, checkpoint_idx: int) -> float:
        checkpoint_idx = int(checkpoint_idx)
        if checkpoint_idx < 0 or checkpoint_idx >= self.num_checkpoints:
            return self.fallback_steps * self.multiplier
        if self.ready(checkpoint_idx):
            return float(np.mean(self.samples[checkpoint_idx])) * self.multiplier
        return self.fallback_steps * self.multiplier

    def count(self, checkpoint_idx: int) -> int:
        checkpoint_idx = int(checkpoint_idx)
        if checkpoint_idx < 0 or checkpoint_idx >= self.num_checkpoints:
            return 0
        return len(self.samples[checkpoint_idx])

    def ready_count(self) -> int:
        return sum(1 for i in range(self.num_checkpoints) if self.ready(i))

    def global_mean(self) -> float:
        all_samples = []
        for item in self.samples:
            all_samples.extend(item)
        if not all_samples:
            return self.fallback_steps
        return float(np.mean(all_samples))

    def compact_summary(self) -> str:
        return (
            f'ready={self.ready_count()}/{self.num_checkpoints}, '
            f'global_mean={self.global_mean():.1f}, '
            f'fallback={self.fallback_steps:.1f}, '
            f'mult={self.multiplier:.2f}'
        )

    def detail_summary(self) -> str:
        parts = []
        for i in range(self.num_checkpoints):
            if self.ready(i):
                parts.append(f'cp{i}:{self.get(i):.1f}({self.count(i)})')
            else:
                parts.append(f'cp{i}:fallback({self.count(i)})')
        return ' | '.join(parts)


class ForwardProgressTracker:
    """reference waypoint index 변화량으로 전진 진행량을 누적한다."""

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
        self.prev_idx = get_nearest_waypoint_idx(position, self.reference_waypoints)
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

        nearest_idx = get_nearest_waypoint_idx(position, self.reference_waypoints)
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
            min(max(self.forward_progress, 0.0), self.total_waypoints)
        )
        progress_pct = (
            progress_score / self.total_waypoints * 100.0
            if self.total_waypoints > 0
            else 0.0
        )
        forward_done = progress_score >= self.total_waypoints

        return progress_score, progress_pct, forward_done, progress_delta


# ── 브레이크 유틸리티 ─────────────────────────────────────────────────────────
def apply_brake(current_speed: float, target_speed: float) -> float:
    """현재 속도가 목표속도보다 높으면 command speed를 낮춰 감속한다."""
    if current_speed > target_speed:
        diff = current_speed - target_speed
        cmd_speed = target_speed - BRAKE_GAIN * diff
        cmd_speed = max(cmd_speed, SPEED_MIN)
        return float(cmd_speed)
    return float(target_speed)


def compute_steer_change_penalty(
    speed_value: float,
    current_steering: float,
    prev_steering: float,
) -> float:
    """고속에서 steering을 크게 꺾는 것에 패널티를 준다."""
    speed_over = max(0.0, speed_value - STEER_SPEED_THRESHOLD)
    if speed_over <= 0.0:
        return 0.0

    steer_over = max(0.0, abs(current_steering) - STEER_DEADZONE)
    if steer_over <= 0.0:
        return 0.0

    return float(STEER_PENALTY * steer_over * speed_over)


# ── 웨이포인트 로드 ───────────────────────────────────────────────────────────
def load_racing_lines() -> dict:
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

    print(f'라인 {len(wp["lines"])}개 생성 완료 (점 수: {len(wp["lines"][0])})')
    return wp


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

    init_poses = np.array([[float(p0[0]), float(p0[1]), heading]])

    print(
        f'시작 pose 자동 설정 | '
        f'line: {start_line_idx}, '
        f'wp: {start_wp_idx}, '
        f'x: {p0[0]:.3f}, y: {p0[1]:.3f}, '
        f'theta: {heading:.3f}'
    )

    return init_poses


# ── 전처리 ────────────────────────────────────────────────────────────────────
def preprocess_lidar(obs_raw: dict) -> np.ndarray:
    lidar = obs_raw['scans'][0].astype(np.float32)
    lidar = np.where(np.isfinite(lidar), lidar, OBS_CONFIG['lidar_range_max'])
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

    곡률이 클수록 해당 구간이 급하게 휘는 것으로 해석한다.
    거의 직선이면 0에 가까운 값이 나온다.
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
    normalize: bool = True,
    max_curvature: float = 1.5,
) -> np.ndarray:
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

    line_curvatures = []

    for waypoints in waypoints_lines:
        n = len(waypoints)
        nearest_idx = get_nearest_waypoint_idx(position, waypoints)

        curvatures = []

        for offset in range(0, lookahead_window, sample_step):
            i0 = (nearest_idx + offset) % n
            i1 = (nearest_idx + offset + sample_step) % n
            i2 = (nearest_idx + offset + 2 * sample_step) % n

            p0 = waypoints[i0]
            p1 = waypoints[i1]
            p2 = waypoints[i2]

            curvature = compute_three_point_curvature(p0, p1, p2)
            curvatures.append(curvature)

        if not curvatures:
            line_curvature = 0.0
        elif mode == 'mean':
            line_curvature = float(np.mean(curvatures))
        else:
            line_curvature = float(np.max(curvatures))

        if normalize:
            line_curvature = np.clip(
                line_curvature / max_curvature,
                0.0,
                1.0,
            )

        line_curvatures.append(line_curvature)

    return np.array(line_curvatures, dtype=np.float32)


def preprocess_obs(obs_raw: dict, waypoints_lines: list, num_lines: int) -> np.ndarray:
    lidar = preprocess_lidar(obs_raw)
    position = np.array(
        [float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])],
        dtype=np.float32,
    )
    heading = float(obs_raw['poses_theta'][0])
    speed = float(obs_raw['linear_vels_x'][0])

    base_obs = build_observation(
        lidar,
        position,
        heading,
        speed,
        waypoints_lines,
        num_lines,
    ).astype(np.float32)

    if OBS_CONFIG.get('use_line_curvature', False):
        line_curvatures = compute_line_lookahead_curvatures(
            obs_raw,
            waypoints_lines,
            mode=OBS_CONFIG.get('curvature_mode', 'max'),
            normalize=True,
            max_curvature=OBS_CONFIG.get('curvature_max_value', 1.5),
        )

        return np.concatenate([base_obs, line_curvatures]).astype(np.float32)

    return base_obs.astype(np.float32)


# ── action → 환경 제어 ────────────────────────────────────────────────────────
def action_to_env(
    action: np.ndarray,
    obs_raw: dict,
    model: SAC,
    waypoints_lines: list,
    controller: PurePursuitController,
) -> np.ndarray:
    """
    action[1]을 절대 속도가 아니라 pp_speed 기준 배율로 해석한다.

    speed_scale = 1 + SAC_SPEED_SCALE_RANGE * action[1]

    예:
        SAC_SPEED_SCALE_RANGE = 0.2
        action[1] = -1.0 → pp_speed * 0.8
        action[1] =  0.0 → pp_speed * 1.0
        action[1] = +1.0 → pp_speed * 1.2
    """
    line_idx = model.action_to_line_index(action)
    waypoints = waypoints_lines[line_idx]

    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    heading = float(obs_raw['poses_theta'][0])
    current_speed = float(obs_raw['linear_vels_x'][0])

    steering, pp_speed = controller.compute(x, y, heading, current_speed, waypoints)

    speed_action = float(np.clip(action[1], -1.0, 1.0))
    speed_scale = 1.0 + SAC_SPEED_SCALE_RANGE * speed_action
    target_speed = pp_speed * speed_scale
    target_speed = float(np.clip(target_speed, TARGET_SPEED_MIN, SPEED_MAX))

    cmd_speed = apply_brake(current_speed, target_speed)
    return np.array([steering, cmd_speed], dtype=np.float32)


# ── Reward ────────────────────────────────────────────────────────────────────
def compute_reward(
    obs_raw: dict,
    action: np.ndarray,
    model: SAC,
    waypoints_lines: list,
    checkpoint_tracker: CheckpointTracker,
    episode: int,
    baseline_provider: WarmupCheckpointBaseline = None,
    use_speed_reward: bool = True,
):
    """
    Returns:
        reward, line_idx, nearest_idx, checkpoint_passed, segment_steps, checkpoint_idx

    use_speed_reward=False이면 checkpoint 도착 보상만 주고,
    baseline 대비 speed reward는 주지 않는다. warmup 중에는 False 권장.
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    collision = bool(obs_raw['collisions'][0])

    line_idx = model.action_to_line_index(action)
    waypoints = waypoints_lines[line_idx]

    position = np.array([x, y], dtype=np.float32)
    nearest_idx = get_nearest_waypoint_idx(position, waypoints)

    if collision:
        penalty = get_collision_penalty(episode)
        return penalty, line_idx, nearest_idx, False, 0, -1

    checkpoint_tracker.tick()
    passed, segment_steps, _, checkpoint_idx = checkpoint_tracker.check(nearest_idx)

    checkpoint_reward = 0.0
    if passed:
        checkpoint_reward = CHECKPOINT_ARRIVAL_REWARD

        if use_speed_reward:
            if baseline_provider is not None:
                current_baseline_steps = baseline_provider.get(checkpoint_idx)
            else:
                current_baseline_steps = BASELINE_STEPS

            current_baseline_steps = max(float(current_baseline_steps), 1.0)

            time_ratio = max(0.0, 1.0 - segment_steps / current_baseline_steps)
            checkpoint_reward += SPEED_REWARD_SCALE * time_ratio

    return checkpoint_reward, line_idx, nearest_idx, passed, segment_steps, checkpoint_idx


# ── Warmup (Pure Pursuit 기반) ────────────────────────────────────────────────
def make_warmup_action(
    obs_raw: dict,
    waypoints_lines: list,
    num_lines: int,
    controller: PurePursuitController,
) -> tuple:
    """
    Hybrid SAC용 warmup action 생성.

    새 speed action 의미:
        action[1] = 0.0 → pp_speed 그대로

    안정성을 위해 warmup에서는 [-WARMUP_SPEED_ACTION_RANGE, +WARMUP_SPEED_ACTION_RANGE]
    안에서 작은 랜덤 speed action을 넣고, 실제 env_action도 같은 배율을 반영한다.
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    heading = float(obs_raw['poses_theta'][0])
    current_speed = float(obs_raw['linear_vels_x'][0])
    position = np.array([x, y], dtype=np.float32)

    min_dist = float('inf')
    best_line = num_lines // 2

    for i, wp in enumerate(waypoints_lines):
        idx = get_nearest_waypoint_idx(position, wp)
        dist = float(np.linalg.norm(wp[idx] - position))
        if dist < min_dist:
            min_dist = dist
            best_line = i

    if np.random.rand() < 0.05:
        line_idx = np.random.randint(0, num_lines)
    else:
        line_idx = best_line

    waypoints = waypoints_lines[line_idx]
    steering, pp_speed = controller.compute(x, y, heading, current_speed, waypoints)

    speed_val = float(np.random.uniform(-WARMUP_SPEED_ACTION_RANGE, WARMUP_SPEED_ACTION_RANGE))
    speed_val = float(np.clip(speed_val, -1.0, 1.0))
    speed_scale = 1.0 + SAC_SPEED_SCALE_RANGE * speed_val

    target_speed = pp_speed * speed_scale
    target_speed = float(np.clip(target_speed, TARGET_SPEED_MIN, SPEED_MAX))
    cmd_speed = apply_brake(current_speed, target_speed)

    env_action = np.array([steering, cmd_speed], dtype=np.float32)
    sac_action = np.array([float(line_idx), speed_val], dtype=np.float32)

    return sac_action, env_action


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def is_valid_obs(obs: np.ndarray) -> bool:
    return bool(np.isfinite(obs).all())


def is_valid_transition(obs, action, reward, next_obs) -> bool:
    return (
        is_valid_obs(obs)
        and is_valid_obs(next_obs)
        and np.isfinite(action).all()
        and np.isfinite(reward)
    )


def build_model(obs_dim: int) -> SAC:
    return SAC(
        obs_dim,
        MODEL_CONFIG['action_dim'],
        MODEL_CONFIG['hidden_dims'],
        MODEL_CONFIG['num_lines'],
    )


def format_lap_times(lap_times: list) -> str:
    if not lap_times:
        return '-'
    return ', '.join([f'{t:.2f}s' for t in lap_times])


# ── 리플레이 버퍼 ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def push(self, obs, action, reward, next_obs, done):
        """action: [line_idx(정수), speed_val(-1~1)]"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(np.array(done)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ── Trainer ───────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, obs_dim: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(
            f'device: {self.device} | '
            f'obs_dim: {obs_dim} | '
            f'num_lines: {MODEL_CONFIG["num_lines"]}'
        )

        self.model = build_model(obs_dim).to(self.device)
        self.buffer = ReplayBuffer(TRAIN_CONFIG['buffer_size'])
        self.checkpoint_baselines = None
        self._init_optimizers()

    def _init_optimizers(self):
        self.actor_opt = optim.Adam(
            self.model.actor.parameters(),
            lr=TRAIN_CONFIG['lr_actor'],
        )
        self.critic1_opt = optim.Adam(
            self.model.critic1.parameters(),
            lr=TRAIN_CONFIG['lr_critic'],
        )
        self.critic2_opt = optim.Adam(
            self.model.critic2.parameters(),
            lr=TRAIN_CONFIG['lr_critic'],
        )

        # 기존 entropy 설계는 그대로 유지한다.
        continuous_entropy = -1.0 * 0.5
        discrete_entropy = -np.log(1.0 / MODEL_CONFIG['num_lines']) * 0.5

        # 기존 self.target_entropy = continuous_entropy + discrete_entropy 였던 값을
        # line/speed로 분리해서 사용한다.
        self.target_entropy_speed = float(continuous_entropy)
        self.target_entropy_line = float(discrete_entropy)
        self.target_entropy = self.target_entropy_speed + self.target_entropy_line

        self.log_alpha_line = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_speed = torch.zeros(1, requires_grad=True, device=self.device)

        self.alpha_line = self.log_alpha_line.exp()
        self.alpha_speed = self.log_alpha_speed.exp()

        self.alpha_line_opt = optim.Adam(
            [self.log_alpha_line],
            lr=TRAIN_CONFIG['lr_alpha'],
        )
        self.alpha_speed_opt = optim.Adam(
            [self.log_alpha_speed],
            lr=TRAIN_CONFIG['lr_alpha'],
        )

    def _unpack_actor_sample(self, sample_output):
        """
        actor.sample() 결과에서 line/speed log_prob를 분리한다.

        actor.sample()은 다음 형식이어야 한다.
            line_idx, speed, total_log_prob, info = actor.sample(obs)

        info에는 반드시 다음 key가 있어야 한다.
            info['line_log_prob']
            info['speed_log_prob']
        """
        if not isinstance(sample_output, tuple) or len(sample_output) != 4:
            raise RuntimeError(
                'actor.sample()은 (line_idx, speed, total_log_prob, info) '
                '형식으로 4개 값을 반환해야 합니다.'
            )

        line_idx, speed, total_log_prob, info = sample_output

        if not isinstance(info, dict):
            raise RuntimeError(
                'actor.sample()의 4번째 반환값이 dict가 아닙니다. '
                'sac_model.py에서 line_log_prob와 speed_log_prob를 info dict로 반환하도록 수정해야 합니다.'
            )

        if 'line_log_prob' not in info or 'speed_log_prob' not in info:
            raise RuntimeError(
                'actor.sample() info에 line_log_prob 또는 speed_log_prob가 없습니다.'
            )

        line_log_prob = info['line_log_prob']
        speed_log_prob = info['speed_log_prob']

        if line_log_prob.dim() == 1:
            line_log_prob = line_log_prob.unsqueeze(1)
        if speed_log_prob.dim() == 1:
            speed_log_prob = speed_log_prob.unsqueeze(1)
        if total_log_prob.dim() == 1:
            total_log_prob = total_log_prob.unsqueeze(1)

        return line_idx, speed, total_log_prob, line_log_prob, speed_log_prob

    def update(self):
        if len(self.buffer) < TRAIN_CONFIG['batch_size']:
            return

        obs, action, reward, next_obs, done = self.buffer.sample(TRAIN_CONFIG['batch_size'])

        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).clamp(REWARD_CLAMP_MIN, REWARD_CLAMP_MAX)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        num_lines = self.model.num_lines
        gamma = TRAIN_CONFIG['gamma']
        tau = TRAIN_CONFIG['tau']

        buf_line_idx = action[:, 0].long()
        buf_speed = action[:, 1].unsqueeze(1)
        buf_action_encoded = encode_action(buf_line_idx, buf_speed, num_lines, self.device)

        # ── Critic 학습 ──
        with torch.no_grad():
            (
                next_line_idx,
                next_speed,
                _,
                next_line_log_prob,
                next_speed_log_prob,
            ) = self._unpack_actor_sample(self.model.actor.sample(next_obs))

            next_action_encoded = encode_action(
                next_line_idx,
                next_speed,
                num_lines,
                self.device,
            )
            target_q1 = self.model.target_critic1(next_obs, next_action_encoded)
            target_q2 = self.model.target_critic2(next_obs, next_action_encoded)

            next_entropy_term = (
                self.alpha_line * next_line_log_prob
                + self.alpha_speed * next_speed_log_prob
            )

            target_q = torch.min(target_q1, target_q2) - next_entropy_term
            target_q = reward + (1.0 - done) * gamma * target_q

        critic1_loss = F.mse_loss(self.model.critic1(obs, buf_action_encoded), target_q)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic1.parameters(), GRAD_CLIP_NORM)
        self.critic1_opt.step()

        critic2_loss = F.mse_loss(self.model.critic2(obs, buf_action_encoded), target_q)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic2.parameters(), GRAD_CLIP_NORM)
        self.critic2_opt.step()

        # ── Actor 학습 ──
        (
            new_line_idx,
            new_speed,
            _,
            line_log_prob,
            speed_log_prob,
        ) = self._unpack_actor_sample(self.model.actor.sample(obs))

        new_action_encoded = encode_action(new_line_idx, new_speed, num_lines, self.device)
        q1_new = self.model.critic1(obs, new_action_encoded)
        q2_new = self.model.critic2(obs, new_action_encoded)
        q_new = torch.min(q1_new, q2_new)

        # actor update에서는 alpha를 상수처럼 사용한다.
        entropy_term = (
            self.alpha_line.detach() * line_log_prob
            + self.alpha_speed.detach() * speed_log_prob
        )
        actor_loss = (entropy_term - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), GRAD_CLIP_NORM)
        self.actor_opt.step()

        # ── Alpha 학습: line entropy ──
        alpha_line_loss = -(
            self.log_alpha_line
            * (line_log_prob + self.target_entropy_line).detach()
        ).mean()

        self.alpha_line_opt.zero_grad()
        alpha_line_loss.backward()
        self.alpha_line_opt.step()
        self.alpha_line = self.log_alpha_line.exp()

        # ── Alpha 학습: speed entropy ──
        alpha_speed_loss = -(
            self.log_alpha_speed
            * (speed_log_prob + self.target_entropy_speed).detach()
        ).mean()

        self.alpha_speed_opt.zero_grad()
        alpha_speed_loss.backward()
        self.alpha_speed_opt.step()
        self.alpha_speed = self.log_alpha_speed.exp()

        # ── Target Critic 소프트 업데이트 ──
        for critic, target_critic in [
            (self.model.critic1, self.model.target_critic1),
            (self.model.critic2, self.model.target_critic2),
        ]:
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def evaluate(
        self,
        env,
        waypoints_lines: list,
        controller: PurePursuitController,
        init_poses: np.ndarray,
        n_episodes: int = 3,
    ) -> float:
        total_reward = 0.0
        num_lines = MODEL_CONFIG['num_lines']
        reference_line = waypoints_lines[num_lines // 2]
        n_wp = len(reference_line)

        for _ in range(n_episodes):
            obs_raw, _, _, _ = env.reset(poses=init_poses)
            obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

            progress_tracker = ForwardProgressTracker(
                reference_line,
                max_laps=MAX_LAPS,
                max_forward_jump=MAX_FORWARD_WP_JUMP,
            )
            progress_tracker.reset_from_obs(obs_raw)
            eval_checkpoint = CheckpointTracker(n_wp)

            progress_window_sum = 0.0
            progress_window_steps = 0
            no_progress_bad_count = 0
            prev_steering = None

            for step_in_ep in range(TRAIN_CONFIG['max_steps']):
                if not is_valid_obs(obs):
                    print('[WARN][eval] invalid obs before select_action. terminate episode.')
                    break

                action = self.model.select_action(obs, training=False)
                env_action = action_to_env(action, obs_raw, self.model, waypoints_lines, controller)
                next_obs_raw, _, done, _ = env.step(np.array([env_action]))

                current_collision = bool(next_obs_raw['collisions'][0])
                speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))
                next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)

                if not is_valid_obs(next_obs):
                    invalid_penalty = get_collision_penalty(COLLISION_CURRICULUM_EPISODES) * INVALID_OBS_PENALTY_SCALE
                    total_reward += float(invalid_penalty)
                    print('[WARN][eval] invalid next_obs. apply penalty and terminate episode.')
                    break

                progress_score, progress_pct, forward_done, progress_delta = progress_tracker.update(next_obs_raw)
                reward, _, _, _, _, _ = compute_reward(
                    next_obs_raw,
                    action,
                    self.model,
                    waypoints_lines,
                    eval_checkpoint,
                    episode=COLLISION_CURRICULUM_EPISODES,
                    baseline_provider=self.checkpoint_baselines,
                    use_speed_reward=True,
                )

                current_steering = float(env_action[0])
                reward -= compute_steer_change_penalty(speed_value, current_steering, prev_steering)
                prev_steering = current_steering

                no_progress_done = False
                if not current_collision:
                    if progress_delta > 0.0:
                        reward += WAYPOINT_PROGRESS_REWARD * progress_delta

                    progress_window_sum += progress_delta
                    progress_window_steps += 1

                    if progress_window_steps >= NO_PROGRESS_CHECK_INTERVAL:
                        if progress_window_sum < NO_PROGRESS_MIN_DELTA:
                            reward += NO_PROGRESS_PENALTY
                            no_progress_bad_count += 1
                        else:
                            no_progress_bad_count = 0

                        progress_window_sum = 0.0
                        progress_window_steps = 0

                    if no_progress_bad_count >= NO_PROGRESS_PATIENCE:
                        reward += NO_PROGRESS_TERMINAL_PENALTY
                        no_progress_done = True

                timeout_done = step_in_ep == TRAIN_CONFIG['max_steps'] - 1
                if timeout_done and not forward_done and not current_collision and not no_progress_done:
                    timeout_penalty = (
                        TIMEOUT_FIXED_PENALTY
                        + TIMEOUT_PENALTY_SCALE * (1.0 - progress_pct / 100.0)
                    )
                    reward += timeout_penalty

                obs = next_obs
                obs_raw = next_obs_raw
                total_reward += float(reward)

                if done or forward_done or current_collision or no_progress_done or timeout_done:
                    break

        return total_reward / max(n_episodes, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                'model_state': self.model.state_dict(),
                'log_alpha_line': self.log_alpha_line.detach().cpu(),
                'log_alpha_speed': self.log_alpha_speed.detach().cpu(),
                'model_config': {
                    'action_dim': MODEL_CONFIG['action_dim'],
                    'hidden_dims': MODEL_CONFIG['hidden_dims'],
                    'num_lines': MODEL_CONFIG['num_lines'],
                    'use_line_curvature': OBS_CONFIG.get('use_line_curvature', False),
                    'obs_dim': get_obs_dim(
                        OBS_CONFIG['lidar_size'],
                        MODEL_CONFIG['num_lines'],
                        use_line_curvature=OBS_CONFIG.get('use_line_curvature', False),
                    ),
                },
            },
            path,
        )
        print(f'모델 저장: {path}')

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])

            if 'log_alpha_line' in checkpoint:
                self.log_alpha_line.data.copy_(checkpoint['log_alpha_line'].to(self.device))
                self.alpha_line = self.log_alpha_line.exp()

            if 'log_alpha_speed' in checkpoint:
                self.log_alpha_speed.data.copy_(checkpoint['log_alpha_speed'].to(self.device))
                self.alpha_speed = self.log_alpha_speed.exp()
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise RuntimeError(f'지원하지 않는 모델 파일 형식입니다: {type(checkpoint)}')
        self.model.to(self.device)
        self.model.eval()
        print(f'모델 로드: {path}')


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────
def main():
    num_lines = MODEL_CONFIG['num_lines']
    obs_dim = get_obs_dim(
        OBS_CONFIG['lidar_size'],
        num_lines,
        use_line_curvature=OBS_CONFIG.get('use_line_curvature', False),
    )
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

    wp = load_racing_lines()
    waypoints_lines = wp['lines']
    progress_reference_line = waypoints_lines[num_lines // 2]
    n_waypoints = len(progress_reference_line)

    controller = PurePursuitController(max_speed=SPEED_MAX, min_speed=TARGET_SPEED_MIN)
    init_poses = make_init_pose(waypoints_lines)

    trainer = Trainer(obs_dim)
    best_reward = -float('inf')
    total_steps = 0

    warmup_baseline = WarmupCheckpointBaseline(
        num_checkpoints=NUM_CHECKPOINTS,
        fallback_steps=BASELINE_STEPS,
        min_samples=WARMUP_BASELINE_MIN_SAMPLES,
        multiplier=WARMUP_BASELINE_MULTIPLIER,
    )
    trainer.checkpoint_baselines = warmup_baseline

    cp_indices = build_checkpoint_indices(n_waypoints)
    print(f'\n체크포인트 {NUM_CHECKPOINTS}개 설정: {cp_indices}')
    print(f'총 waypoint 수: {n_waypoints}')
    print(
        f'\n학습 시작 | map: {ENV_CONFIG["map"]} | '
        f'obs_dim: {obs_dim} | lines: {num_lines}'
    )
    print(f'브레이크 명령 하한 SPEED_MIN: {SPEED_MIN}')
    print(f'SAC 목표 속도 하한 TARGET_SPEED_MIN: {TARGET_SPEED_MIN}')
    print(f'속도 상한 SPEED_MAX: {SPEED_MAX}')
    print(f'브레이크 gain: {BRAKE_GAIN}')

    for episode in range(TRAIN_CONFIG['max_episodes']):
        # episode 시작 시점에 warmup 여부를 고정한다.
        # episode 중간에서 warmup_steps를 넘더라도 그 episode는 warmup으로 끝낸다.
        is_warmup_episode = total_steps < TRAIN_CONFIG['warmup_steps']

        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        episode_reward = 0.0
        speeds = []
        episode_step = 0
        lap_times = []
        last_lap_step = 0
        next_lap_progress = n_waypoints

        progress_tracker = ForwardProgressTracker(
            progress_reference_line,
            max_laps=MAX_LAPS,
            max_forward_jump=MAX_FORWARD_WP_JUMP,
        )
        progress_tracker.reset_from_obs(obs_raw)
        progress_score = 1
        progress_pct = progress_score / (n_waypoints * MAX_LAPS) * 100.0 if n_waypoints > 0 else 0.0

        checkpoint_tracker = CheckpointTracker(n_waypoints)
        collisions = 0
        last_line_idx = -1
        progress_window_sum = 0.0
        progress_window_steps = 0
        no_progress_bad_count = 0
        no_progress_done_count = 0
        prev_steering = None

        for step_in_ep in range(TRAIN_CONFIG['max_steps']):
            if is_warmup_episode:
                action, env_action = make_warmup_action(
                    obs_raw,
                    waypoints_lines,
                    num_lines,
                    controller,
                )
            else:
                if not is_valid_obs(obs):
                    print('[WARN] invalid obs before select_action. terminate episode.')
                    break

                action = trainer.model.select_action(obs, training=True)
                if SPEED_ACTION_NOISE_STD > 0.0:
                    speed_noise = np.random.normal(0.0, SPEED_ACTION_NOISE_STD)
                    action[1] = float(np.clip(action[1] + speed_noise, -1.0, 1.0))

                env_action = action_to_env(
                    action,
                    obs_raw,
                    trainer.model,
                    waypoints_lines,
                    controller,
                )

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))
            episode_step += 1

            current_collision = bool(next_obs_raw['collisions'][0])
            if current_collision:
                collisions += 1

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            if not is_valid_obs(next_obs):
                reward = get_collision_penalty(episode) * INVALID_OBS_PENALTY_SCALE
                if is_valid_obs(obs) and np.isfinite(action).all() and np.isfinite(reward):
                    trainer.buffer.push(obs, action, reward, obs, 1.0)
                episode_reward += float(reward)
                total_steps += 1
                print('[WARN] invalid next_obs after env.step. apply penalty and terminate episode.')
                break

            progress_score, progress_pct, forward_done, progress_delta = progress_tracker.update(next_obs_raw)

            while progress_score >= next_lap_progress and len(lap_times) < MAX_LAPS:
                lap_steps = episode_step - last_lap_step
                lap_time = lap_steps * ENV_CONFIG['timestep']
                lap_times.append(lap_time)
                last_lap_step = episode_step
                next_lap_progress += n_waypoints

            speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))
            if np.isfinite(speed_value):
                speeds.append(speed_value)

            reward, line_idx, _, checkpoint_passed, segment_steps, checkpoint_idx = compute_reward(
                next_obs_raw,
                action,
                trainer.model,
                waypoints_lines,
                checkpoint_tracker,
                episode=episode,
                baseline_provider=warmup_baseline,
                use_speed_reward=not is_warmup_episode,
            )
            last_line_idx = line_idx

            # warmup 중에는 speed reward 없이 baseline sample만 저장한다.
            if is_warmup_episode and checkpoint_passed:
                warmup_baseline.add(checkpoint_idx, segment_steps)

            current_steering = float(env_action[0])
            reward -= compute_steer_change_penalty(speed_value, current_steering, prev_steering)
            prev_steering = current_steering

            if current_collision:
                print(
                    f'[CRASH] '
                    f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%) | '
                    f'speed={speed_value:.2f} | '
                    f'line={line_idx} | '
                    f'action=[line={int(action[0])}, spd_scale={action[1]:.3f}] | '
                    f'env_action={np.round(env_action, 3)}'
                )

            no_progress_done = False
            if not current_collision:
                if progress_delta > 0.0:
                    reward += WAYPOINT_PROGRESS_REWARD * progress_delta

                progress_window_sum += progress_delta
                progress_window_steps += 1

                if progress_window_steps >= NO_PROGRESS_CHECK_INTERVAL:
                    if progress_window_sum < NO_PROGRESS_MIN_DELTA:
                        reward += NO_PROGRESS_PENALTY
                        no_progress_bad_count += 1
                    else:
                        no_progress_bad_count = 0

                    progress_window_sum = 0.0
                    progress_window_steps = 0

                if no_progress_bad_count >= NO_PROGRESS_PATIENCE:
                    reward += NO_PROGRESS_TERMINAL_PENALTY
                    no_progress_done = True
                    no_progress_done_count += 1
                    print(
                        f'[NO_PROGRESS] terminate | '
                        f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                        f'({progress_pct:.1f}%) | '
                        f'bad_count={no_progress_bad_count} | '
                        f'penalty={NO_PROGRESS_TERMINAL_PENALTY}'
                    )

            timeout_done = step_in_ep == TRAIN_CONFIG['max_steps'] - 1
            if timeout_done and not forward_done and not current_collision and not no_progress_done:
                timeout_penalty = (
                    TIMEOUT_FIXED_PENALTY
                    + TIMEOUT_PENALTY_SCALE * (1.0 - progress_pct / 100.0)
                )
                reward += timeout_penalty
                print(
                    f'[TIMEOUT] terminate | '
                    f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%) | '
                    f'penalty={timeout_penalty:.1f}'
                )

            terminal = bool(done or forward_done or current_collision or no_progress_done or timeout_done)

            if is_valid_transition(obs, action, reward, next_obs):
                trainer.buffer.push(obs, action, reward, next_obs, float(terminal))
            else:
                print('[WARN] invalid transition skipped.')

            episode_reward += float(reward)
            total_steps += 1

            # warmup episode에서는 update하지 않는다.
            # episode 시작 시점부터 train episode였을 때만 update한다.
            if not is_warmup_episode:
                trainer.update()

            obs = next_obs
            obs_raw = next_obs_raw

            if terminal:
                break

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        lap_time_str = format_lap_times(lap_times)
        phase = 'warmup' if is_warmup_episode else 'train'

        print(
            f'[{phase:6s}] '
            f'ep {episode:4d} | '
            f'reward: {episode_reward:8.1f} | '
            f'wp: {progress_score}/{n_waypoints * MAX_LAPS} '
            f'({progress_pct:.1f}%) | '
            f'speed: {avg_speed:.2f} | '
            f'line: {last_line_idx} | '
            f'crash: {collisions} | '
            f'no_prog_bad: {no_progress_bad_count} | '
            f'lap_time: {lap_time_str} | '
            f'ep_steps: {episode_step} | '
            f'total_steps: {total_steps}'
        )

        # warmup이 끝난 뒤부터만 eval/save한다.
        if (
            total_steps >= TRAIN_CONFIG['warmup_steps']
            and episode > 0
            and episode % TRAIN_CONFIG['eval_interval'] == 0
        ):
            eval_reward = trainer.evaluate(env, waypoints_lines, controller, init_poses)
            print(f'  [EVAL] reward: {eval_reward:.1f} (best: {best_reward:.1f})')

            if eval_reward > best_reward:
                best_reward = eval_reward
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(
                    {
                        'model_state': trainer.model.state_dict(),
                        'log_alpha_line': trainer.log_alpha_line.detach().cpu(),
                        'log_alpha_speed': trainer.log_alpha_speed.detach().cpu(),
                        'model_config': {
                            'action_dim': MODEL_CONFIG['action_dim'],
                            'hidden_dims': MODEL_CONFIG['hidden_dims'],
                            'num_lines': MODEL_CONFIG['num_lines'],
                            'use_line_curvature': OBS_CONFIG.get('use_line_curvature', False),
                            'obs_dim': obs_dim,
                        },
                    },
                    MODEL_SAVE_PATH,
                )
                print(f'  모델 저장: {MODEL_SAVE_PATH}')

    env.close()
    print('\n학습 완료')


if __name__ == '__main__':
    main()
