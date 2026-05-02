"""
train_node.py (Hybrid SAC 버전)

SAC가 주행 라인(이산)과 목표 속도(연속)를 선택하고,
Pure Pursuit가 선택된 라인을 추종하는 구조.

reward 설계:
- waypoint 전진마다 dense reward (waypoint_progress_reward)
- 일정 step 구간 동안 waypoint 진행이 없으면 no_progress_penalty
- no-progress 구간이 연속으로 누적되면 episode 종료 + terminal penalty
- max_steps까지 갔는데 완주 못 하면 timeout penalty
- 체크포인트 통과 시 도착 보상 + checkpoint 구간별 warmup baseline 대비 빠른 도착 보상
- 충돌 페널티: curriculum 방식으로 에피소드에 따라 점진적 증가
- 고속 상태에서 steering을 크게 꺾으면 패널티
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
    SPEED_MIN,
    SPEED_MAX,
    MODEL_SAVE_PATH,
)

from sac_model import SAC, build_observation, get_obs_dim, encode_action
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


# ── 브레이크 설정 ─────────────────────────────────────────────────────────────
BRAKE_GAIN = 0.5

# SAC / Pure Pursuit가 선택하는 "목표 속도"의 최소값
# SPEED_MIN은 브레이크 명령 하한값으로만 사용한다.
TARGET_SPEED_MIN = REWARD_CONFIG.get('target_speed_min', 0.5)


# ── 체크포인트 설정 ───────────────────────────────────────────────────────────
NUM_CHECKPOINTS = REWARD_CONFIG['num_checkpoints']
CHECKPOINT_ARRIVAL_REWARD = REWARD_CONFIG['checkpoint_arrival']
SPEED_REWARD_SCALE = REWARD_CONFIG['speed_reward_scale']

# checkpoint별 warmup baseline이 준비되기 전까지 사용할 fallback 값
BASELINE_STEPS = REWARD_CONFIG.get('baseline_steps', 500)

# checkpoint별 sample이 몇 개 이상 쌓여야 해당 구간 baseline으로 인정할지
# warmup이 1바퀴 정도면 1 추천, 여러 바퀴면 2~3 추천
WARMUP_BASELINE_MIN_SAMPLES = REWARD_CONFIG.get(
    'warmup_baseline_min_samples',
    1,
)


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
NO_PROGRESS_TERMINAL_PENALTY = REWARD_CONFIG.get(
    'no_progress_terminal_penalty',
    -50.0,
)


# ── Timeout penalty 설정 ──────────────────────────────────────────────────────
# max_steps까지 갔는데 완주하지 못한 경우 부여
# progress_pct가 낮을수록 더 큰 페널티
TIMEOUT_PENALTY_SCALE = REWARD_CONFIG.get('timeout_penalty_scale', -200.0)


# ── Steering penalty 설정 ────────────────────────────────────────────────────
STEER_SPEED_THRESHOLD = REWARD_CONFIG.get(
    'steer_speed_threshold',
    8.0,
)
STEER_DEADZONE = REWARD_CONFIG.get(
    'steer_deadzone',
    0.25,
)
STEER_PENALTY = REWARD_CONFIG.get(
    'steer_penalty',
    0.2,
)


# ── 학습 안정화 설정 ─────────────────────────────────────────────────────────
REWARD_CLAMP_MIN = REWARD_CONFIG.get('reward_clamp_min', -500.0)
REWARD_CLAMP_MAX = REWARD_CONFIG.get('reward_clamp_max', 200.0)
INVALID_OBS_PENALTY_SCALE = REWARD_CONFIG.get('invalid_obs_penalty_scale', 2.0)
GRAD_CLIP_NORM = TRAIN_CONFIG.get('grad_clip_norm', 1.0)


def get_collision_penalty(episode: int) -> float:
    """
    에피소드에 따라 충돌 페널티를 점진적으로 키운다.
    초반에는 가볍게 → 후반에는 강하게.
    """
    if episode >= COLLISION_CURRICULUM_EPISODES:
        return COLLISION_PENALTY_END

    ratio = episode / COLLISION_CURRICULUM_EPISODES
    penalty = COLLISION_PENALTY_START + ratio * (
        COLLISION_PENALTY_END - COLLISION_PENALTY_START
    )
    return penalty


def build_checkpoint_indices(n_waypoints: int, num_checkpoints: int = None) -> list:
    if num_checkpoints is None:
        num_checkpoints = NUM_CHECKPOINTS

    step = n_waypoints // num_checkpoints
    indices = [(i + 1) * step % n_waypoints for i in range(num_checkpoints)]
    return indices


class CheckpointTracker:
    """
    체크포인트 통과를 추적하고, 구간 소요 스텝 수를 계산한다.
    """
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
        is_last = (self.next_checkpoint == len(self.checkpoint_indices) - 1)
        self.next_checkpoint += 1

        return True, steps, is_last, passed_checkpoint_idx


class WarmupCheckpointBaseline:
    """
    checkpoint 구간별 warmup segment_steps 평균을 저장한다.

    예:
        checkpoint 0 baseline = warmup에서 checkpoint 0까지 걸린 평균 step
        checkpoint 1 baseline = warmup에서 checkpoint 1까지 걸린 평균 step
        ...
        checkpoint 9 baseline = warmup에서 checkpoint 9까지 걸린 평균 step

    SAC가 해당 checkpoint 구간에서 baseline보다 빠르면 speed reward를 받는다.
    """
    def __init__(
        self,
        num_checkpoints: int,
        fallback_steps: float,
        min_samples: int = 1,
    ):
        self.num_checkpoints = int(num_checkpoints)
        self.fallback_steps = float(fallback_steps)
        self.min_samples = int(min_samples)
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
            return self.fallback_steps

        if self.ready(checkpoint_idx):
            return float(np.mean(self.samples[checkpoint_idx]))

        return self.fallback_steps

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

        if len(all_samples) == 0:
            return self.fallback_steps

        return float(np.mean(all_samples))

    def compact_summary(self) -> str:
        return (
            f'ready={self.ready_count()}/{self.num_checkpoints}, '
            f'global_mean={self.global_mean():.1f}, '
            f'fallback={self.fallback_steps:.1f}'
        )

    def detail_summary(self) -> str:
        parts = []
        for i in range(self.num_checkpoints):
            if self.ready(i):
                parts.append(
                    f'cp{i}:{self.get(i):.1f}({self.count(i)})'
                )
            else:
                parts.append(
                    f'cp{i}:fallback({self.count(i)})'
                )

        return ' | '.join(parts)


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


# ── 브레이크 유틸리티 ─────────────────────────────────────────────────────────
def apply_brake(current_speed: float, target_speed: float) -> float:
    """
    브레이크 명령은 SPEED_MIN까지 허용한다.
    여기서 SPEED_MIN은 환경에 넣는 command speed의 하한값이다.
    """
    if current_speed > target_speed:
        diff = current_speed - target_speed
        cmd_speed = target_speed - BRAKE_GAIN * diff
        cmd_speed = max(cmd_speed, SPEED_MIN)
        return cmd_speed

    return target_speed


def compute_steer_change_penalty(
    speed_value: float,
    current_steering: float,
    prev_steering: float,
) -> float:
    """
    고속에서 steering을 크게 꺾는 것에 패널티를 준다.

    penalty =
        steer_penalty
        * max(abs(steering) - steer_deadzone, 0)
        * max(speed - speed_threshold, 0)
    """
    speed_over = max(0.0, speed_value - STEER_SPEED_THRESHOLD)
    if speed_over <= 0.0:
        return 0.0

    steer_over = max(0.0, abs(current_steering) - STEER_DEADZONE)
    if steer_over <= 0.0:
        return 0.0

    penalty = STEER_PENALTY * steer_over * speed_over
    return float(penalty)


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

    return build_observation(
        lidar,
        position,
        heading,
        speed,
        waypoints_lines,
        num_lines,
    ).astype(np.float32)


# ── action → 환경 제어 ────────────────────────────────────────────────────────
def action_to_env(
    action: np.ndarray,
    obs_raw: dict,
    model: SAC,
    waypoints_lines: list,
    controller: PurePursuitController,
) -> np.ndarray:
    line_idx = model.action_to_line_index(action)
    waypoints = waypoints_lines[line_idx]

    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    heading = float(obs_raw['poses_theta'][0])
    current_speed = float(obs_raw['linear_vels_x'][0])

    steering, pp_speed = controller.compute(
        x, y, heading, current_speed, waypoints,
    )

    # SAC가 선택하는 것은 "목표 속도"이므로 음수로 두지 않는다.
    # 실제 브레이크 명령은 apply_brake()에서 SPEED_MIN까지 허용한다.
    sac_speed = model.action_to_speed(action, TARGET_SPEED_MIN, SPEED_MAX)
    target_speed = min(sac_speed, pp_speed)

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
):
    """
    reward 구성:
    - 충돌: curriculum 기반 페널티
    - 체크포인트 통과: 도착 보상 + checkpoint별 warmup baseline 대비 빠른 도착 보상
    - waypoint 진행 보상/무진행 페널티는 메인 루프에서 처리
    - steering 패널티도 메인 루프에서 처리

    Returns:
        reward, line_idx, nearest_idx, checkpoint_passed, segment_steps, checkpoint_idx
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
        if baseline_provider is not None:
            current_baseline_steps = baseline_provider.get(checkpoint_idx)
        else:
            current_baseline_steps = BASELINE_STEPS

        current_baseline_steps = max(float(current_baseline_steps), 1.0)

        time_ratio = max(
            0.0,
            1.0 - segment_steps / current_baseline_steps,
        )

        checkpoint_reward = (
            CHECKPOINT_ARRIVAL_REWARD
            + SPEED_REWARD_SCALE * time_ratio
        )

    reward = checkpoint_reward

    return reward, line_idx, nearest_idx, passed, segment_steps, checkpoint_idx


# ── Warmup (Pure Pursuit 기반) ────────────────────────────────────────────────
def make_warmup_action(
    obs_raw: dict,
    waypoints_lines: list,
    num_lines: int,
    controller: PurePursuitController,
) -> tuple:
    """
    Hybrid SAC용 warmup action 생성.
    action = [line_idx(정수), speed_val(-1~1)]
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

    # warmup은 안정적인 주행 데이터를 넣기 위해 가까운 라인 위주로 선택
    if np.random.rand() < 0.05:
        line_idx = np.random.randint(0, num_lines)
    else:
        line_idx = best_line

    waypoints = waypoints_lines[line_idx]
    steering, pp_speed = controller.compute(x, y, heading, current_speed, waypoints)

    cmd_speed = apply_brake(current_speed, pp_speed)
    env_action = np.array([steering, cmd_speed], dtype=np.float32)

    # warmup buffer에 들어갈 speed action도 SAC 목표 속도 범위와 맞춘다.
    speed_val = (
        (pp_speed - TARGET_SPEED_MIN)
        / (SPEED_MAX - TARGET_SPEED_MIN)
    ) * 2.0 - 1.0
    speed_val = float(np.clip(speed_val, -1.0, 1.0))

    # Hybrid SAC: line_idx는 정수, speed_val은 연속값
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
        """
        action: [line_idx(정수), speed_val(-1~1)]
        """
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

        # main에서 WarmupCheckpointBaseline 객체를 넣어준다.
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

        # target entropy: 이산(라인) + 연속(속도) 합산
        continuous_entropy = -1.0 * 0.5
        discrete_entropy = -np.log(1.0 / MODEL_CONFIG['num_lines']) * 0.5
        self.target_entropy = continuous_entropy + discrete_entropy

        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=self.device,
        )
        self.alpha = self.log_alpha.exp()

        self.alpha_opt = optim.Adam(
            [self.log_alpha],
            lr=TRAIN_CONFIG['lr_alpha'],
        )

    def update(self):
        if len(self.buffer) < TRAIN_CONFIG['batch_size']:
            return

        obs, action, reward, next_obs, done = self.buffer.sample(
            TRAIN_CONFIG['batch_size']
        )

        obs = obs.to(self.device)
        action = action.to(self.device)         # (batch, 2): [line_idx, speed_val]
        reward = reward.to(self.device).clamp(REWARD_CLAMP_MIN, REWARD_CLAMP_MAX)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        num_lines = self.model.num_lines
        gamma = TRAIN_CONFIG['gamma']
        tau = TRAIN_CONFIG['tau']

        # buffer의 action에서 line_idx, speed 분리
        buf_line_idx = action[:, 0].long()      # (batch,)
        buf_speed = action[:, 1].unsqueeze(1)   # (batch, 1)

        # Critic에 넣을 action 인코딩: [onehot(line), speed]
        buf_action_encoded = encode_action(
            buf_line_idx, buf_speed, num_lines, self.device
        )

        # ── Critic 학습 ──
        with torch.no_grad():
            next_line_idx, next_speed, next_log_prob, _ = (
                self.model.actor.sample(next_obs)
            )

            next_action_encoded = encode_action(
                next_line_idx, next_speed, num_lines, self.device
            )

            target_q1 = self.model.target_critic1(next_obs, next_action_encoded)
            target_q2 = self.model.target_critic2(next_obs, next_action_encoded)
            target_q = (
                torch.min(target_q1, target_q2)
                - self.alpha * next_log_prob
            )

            target_q = reward + (1.0 - done) * gamma * target_q

        critic1_loss = F.mse_loss(
            self.model.critic1(obs, buf_action_encoded),
            target_q,
        )
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.critic1.parameters(),
            GRAD_CLIP_NORM,
        )
        self.critic1_opt.step()

        critic2_loss = F.mse_loss(
            self.model.critic2(obs, buf_action_encoded),
            target_q,
        )
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.critic2.parameters(),
            GRAD_CLIP_NORM,
        )
        self.critic2_opt.step()

        # ── Actor 학습 ──
        new_line_idx, new_speed, log_prob, _ = self.model.actor.sample(obs)

        new_action_encoded = encode_action(
            new_line_idx,
            new_speed,
            num_lines,
            self.device,
        )

        q1_new = self.model.critic1(obs, new_action_encoded)
        q2_new = self.model.critic2(obs, new_action_encoded)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.actor.parameters(),
            GRAD_CLIP_NORM,
        )
        self.actor_opt.step()

        # ── Alpha 학습 ──
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()

        # ── Target Critic 소프트 업데이트 ──
        for critic, target_critic in [
            (self.model.critic1, self.model.target_critic1),
            (self.model.critic2, self.model.target_critic2),
        ]:
            for param, target_param in zip(
                critic.parameters(),
                target_critic.parameters(),
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

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

                env_action = action_to_env(
                    action,
                    obs_raw,
                    self.model,
                    waypoints_lines,
                    controller,
                )

                next_obs_raw, _, done, _ = env.step(np.array([env_action]))

                current_collision = bool(next_obs_raw['collisions'][0])
                speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))

                next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)

                if not is_valid_obs(next_obs):
                    invalid_penalty = (
                        get_collision_penalty(COLLISION_CURRICULUM_EPISODES)
                        * INVALID_OBS_PENALTY_SCALE
                    )
                    total_reward += float(invalid_penalty)

                    print('[WARN][eval] invalid next_obs. apply penalty and terminate episode.')
                    break

                progress_score, progress_pct, forward_done, progress_delta = (
                    progress_tracker.update(next_obs_raw)
                )

                reward, _, _, _, _, _ = compute_reward(
                    next_obs_raw,
                    action,
                    self.model,
                    waypoints_lines,
                    eval_checkpoint,
                    episode=COLLISION_CURRICULUM_EPISODES,
                    baseline_provider=self.checkpoint_baselines,
                )

                current_steering = float(env_action[0])
                steer_change_penalty = compute_steer_change_penalty(
                    speed_value,
                    current_steering,
                    prev_steering,
                )
                reward -= steer_change_penalty
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

                if (
                    timeout_done
                    and not forward_done
                    and not current_collision
                    and not no_progress_done
                ):
                    timeout_penalty = TIMEOUT_PENALTY_SCALE * (
                        1.0 - progress_pct / 100.0
                    )
                    reward += timeout_penalty

                obs = next_obs
                obs_raw = next_obs_raw
                total_reward += float(reward)

                if done or forward_done or current_collision or no_progress_done or timeout_done:
                    break

        return total_reward / max(n_episodes, 1)


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────
def main():
    num_lines = MODEL_CONFIG['num_lines']
    obs_dim = get_obs_dim(OBS_CONFIG['lidar_size'], num_lines)

    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

    wp = load_racing_lines()
    waypoints_lines = wp['lines']

    progress_reference_line = waypoints_lines[num_lines // 2]
    n_waypoints = len(progress_reference_line)

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=TARGET_SPEED_MIN,
    )

    init_poses = make_init_pose(waypoints_lines)

    trainer = Trainer(obs_dim)
    best_reward = -float('inf')
    total_steps = 0

    warmup_baseline = WarmupCheckpointBaseline(
        num_checkpoints=NUM_CHECKPOINTS,
        fallback_steps=BASELINE_STEPS,
        min_samples=WARMUP_BASELINE_MIN_SAMPLES,
    )
    trainer.checkpoint_baselines = warmup_baseline

    cp_indices = build_checkpoint_indices(n_waypoints)
    print(f'\n체크포인트 {NUM_CHECKPOINTS}개 설정: {cp_indices}')
    print(f'총 waypoint 수: {n_waypoints}')

    print(
        f'\n학습 시작 | '
        f'map: {ENV_CONFIG["map"]} | '
        f'obs_dim: {obs_dim} | lines: {num_lines}'
    )
    print(f'브레이크 명령 하한 SPEED_MIN: {SPEED_MIN}')
    print(f'SAC 목표 속도 하한 TARGET_SPEED_MIN: {TARGET_SPEED_MIN}')
    print(f'속도 상한 SPEED_MAX: {SPEED_MAX}')
    print(f'브레이크 gain: {BRAKE_GAIN}')

    for episode in range(TRAIN_CONFIG['max_episodes']):
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        episode_reward = 0.0
        speeds = []

        # episode 내부 step / lap time 기록용
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
        progress_pct = (
            progress_score / (n_waypoints * MAX_LAPS) * 100.0
            if n_waypoints > 0
            else 0.0
        )

        checkpoint_tracker = CheckpointTracker(n_waypoints)
        collisions = 0
        last_line_idx = -1

        progress_window_sum = 0.0
        progress_window_steps = 0
        no_progress_bad_count = 0
        no_progress_done_count = 0
        prev_steering = None

        for step_in_ep in range(TRAIN_CONFIG['max_steps']):
            if total_steps < TRAIN_CONFIG['warmup_steps']:
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

                # Hybrid SAC: 라인은 이산이므로 noise 불필요
                # 속도에만 noise 추가
                speed_noise = np.random.normal(0, 0.05)
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
                invalid_penalty = (
                    get_collision_penalty(episode)
                    * INVALID_OBS_PENALTY_SCALE
                )

                reward = invalid_penalty

                if (
                    is_valid_obs(obs)
                    and np.isfinite(action).all()
                    and np.isfinite(reward)
                ):
                    trainer.buffer.push(
                        obs,
                        action,
                        reward,
                        obs,
                        1.0,
                    )

                episode_reward += float(reward)
                total_steps += 1

                print('[WARN] invalid next_obs after env.step. apply penalty and terminate episode.')
                break

            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            # lap time 기록
            while (
                progress_score >= next_lap_progress
                and len(lap_times) < MAX_LAPS
            ):
                lap_steps = episode_step - last_lap_step
                lap_time = lap_steps * ENV_CONFIG['timestep']
                lap_times.append(lap_time)

                last_lap_step = episode_step
                next_lap_progress += n_waypoints

            speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))
            if np.isfinite(speed_value):
                speeds.append(speed_value)

            (
                reward,
                line_idx,
                _,
                checkpoint_passed,
                segment_steps,
                checkpoint_idx,
            ) = compute_reward(
                next_obs_raw,
                action,
                trainer.model,
                waypoints_lines,
                checkpoint_tracker,
                episode=episode,
                baseline_provider=warmup_baseline,
            )
            last_line_idx = line_idx

            # warmup 중에는 checkpoint별 segment step을 baseline sample로 저장
            if total_steps < TRAIN_CONFIG['warmup_steps'] and checkpoint_passed:
                warmup_baseline.add(checkpoint_idx, segment_steps)

            current_steering = float(env_action[0])
            steer_change_penalty = compute_steer_change_penalty(
                speed_value,
                current_steering,
                prev_steering,
            )
            reward -= steer_change_penalty
            prev_steering = current_steering

            if current_collision:
                print(
                    f'[CRASH] '
                    f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%) | '
                    f'speed={speed_value:.2f} | '
                    f'line={line_idx} | '
                    f'action=[line={int(action[0])}, spd={action[1]:.3f}] | '
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

            if (
                timeout_done
                and not forward_done
                and not current_collision
                and not no_progress_done
            ):
                timeout_penalty = TIMEOUT_PENALTY_SCALE * (
                    1.0 - progress_pct / 100.0
                )
                reward += timeout_penalty

                print(
                    f'[TIMEOUT] terminate | '
                    f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%) | '
                    f'penalty={timeout_penalty:.1f}'
                )

            terminal = bool(
                done
                or forward_done
                or current_collision
                or no_progress_done
                or timeout_done
            )

            if is_valid_transition(obs, action, reward, next_obs):
                trainer.buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    float(terminal),
                )
            else:
                print('[WARN] invalid transition skipped.')

            episode_reward += float(reward)
            total_steps += 1

            if total_steps >= TRAIN_CONFIG['warmup_steps']:
                trainer.update()

            obs = next_obs
            obs_raw = next_obs_raw

            if terminal:
                break

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        lap_time_str = format_lap_times(lap_times)

        print(
            f'ep {episode:4d} | '
            f'reward: {episode_reward:8.1f} | '
            f'wp: {progress_score}/{n_waypoints * MAX_LAPS} '
            f'({progress_pct:.1f}%) | '
            f'speed: {avg_speed:.2f} | '
            f'line: {last_line_idx} | '
            f'crash: {collisions} | '
            f'no_prog_bad: {no_progress_bad_count} | '
            f'no_prog_done: {no_progress_done_count} | '
            f'lap_time: {lap_time_str} | '
            f'ep_steps: {episode_step} | '
            f'total_steps: {total_steps}'
        )

        if (
            episode > 0
            and episode % TRAIN_CONFIG['eval_interval'] == 0
        ):
            eval_reward = trainer.evaluate(
                env,
                waypoints_lines,
                controller,
                init_poses,
            )

            print(f'  [EVAL] reward: {eval_reward:.1f} (best: {best_reward:.1f})')

            if eval_reward > best_reward:
                best_reward = eval_reward
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

                torch.save(
                    {
                        'model_state': trainer.model.state_dict(),
                        'model_config': {
                            'action_dim': MODEL_CONFIG['action_dim'],
                            'hidden_dims': MODEL_CONFIG['hidden_dims'],
                            'num_lines': MODEL_CONFIG['num_lines'],
                        },
                    },
                    MODEL_SAVE_PATH,
                )
                print(f'  모델 저장: {MODEL_SAVE_PATH}')

    env.close()
    print('\n학습 완료')


if __name__ == '__main__':
    main()