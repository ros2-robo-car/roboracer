"""
train_node.py

SAC가 주행 라인과 목표 속도를 선택하고,
Pure Pursuit가 선택된 라인을 추종하는 구조.

reward 설계:
- waypoint를 10개 체크포인트로 나눠서 통과 시에만 보상
- 체크포인트 보상 = 도착 보상 + 구간 소요시간 보상 (빠를수록 높음)
- 체크포인트 사이(매 스텝)에는 reward = 0
- 충돌 페널티 = -1000
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

from sac_model import SAC, build_observation, get_obs_dim
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


# ── 브레이크 설정 ─────────────────────────────────────────────────────────────
BRAKE_GAIN = 1.0


# ── 체크포인트 설정 (config.py에서 로드) ──────────────────────────────────────
NUM_CHECKPOINTS           = REWARD_CONFIG['num_checkpoints']
CHECKPOINT_ARRIVAL_REWARD = REWARD_CONFIG['checkpoint_arrival']
SPEED_REWARD_SCALE        = REWARD_CONFIG['speed_reward_scale']
COLLISION_PENALTY         = REWARD_CONFIG['collision_penalty']
LAP_COMPLETION_REWARD     = REWARD_CONFIG['lap_completion_reward']
BASELINE_STEPS            = REWARD_CONFIG.get('baseline_steps', 500)


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
        """매 스텝마다 호출하여 구간 스텝 수를 증가시킨다."""
        self.segment_steps += 1

    def check(self, nearest_idx: int) -> tuple:
        """
        현재 nearest_idx가 다음 체크포인트를 통과했는지 확인한다.

        Returns:
            (passed, segment_steps, is_lap_complete)
            - passed: 체크포인트를 통과했으면 True
            - segment_steps: 구간 소요 스텝 수 (통과 시에만 유효)
            - is_lap_complete: 마지막 체크포인트(한 바퀴)를 통과했으면 True
        """
        if self.next_checkpoint >= len(self.checkpoint_indices):
            return False, 0, False

        target_idx = self.checkpoint_indices[self.next_checkpoint]
        threshold = max(self.n_waypoints // 20, 3)

        forward_dist = (nearest_idx - target_idx) % self.n_waypoints
        passed = forward_dist <= threshold and forward_dist > 0

        if not passed:
            return False, 0, False

        steps = self.segment_steps
        self.segment_steps = 0

        self.checkpoints_passed += 1
        is_last = (self.next_checkpoint == len(self.checkpoint_indices) - 1)
        self.next_checkpoint += 1

        return True, steps, is_last


# ── 브레이크 유틸리티 ─────────────────────────────────────────────────────────
def apply_brake(current_speed: float, target_speed: float) -> float:
    """
    현재 속도가 목표보다 높으면 더 낮은 desired velocity를 반환하여
    내부 PID가 감속하도록 유도한다.
    """
    if current_speed > target_speed:
        diff = current_speed - target_speed
        cmd_speed = target_speed - BRAKE_GAIN * diff
        cmd_speed = max(cmd_speed, -SPEED_MAX)
        return cmd_speed
    else:
        return target_speed


# ── 웨이포인트 로드 ───────────────────────────────────────────────────────────
def load_racing_lines() -> dict:
    csv_path = LINE_CONFIG['centerline_csv']

    if os.path.exists(csv_path):
        print(f'centerline CSV 로드: {csv_path}')
        wp = load_waypoints(
            centerline_path=csv_path,
            num_lines=LINE_CONFIG['num_lines'],
            line_spacing=LINE_CONFIG['line_spacing'],
        )
    else:
        print('CSV 없음 → 맵 이미지에서 centerline 추출')
        wp = load_waypoints(
            map_path=LINE_CONFIG['map_path'],
            map_ext=LINE_CONFIG['map_ext'],
            num_lines=LINE_CONFIG['num_lines'],
            line_spacing=LINE_CONFIG['line_spacing'],
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

    sac_speed = model.action_to_speed(action, SPEED_MIN, SPEED_MAX)
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
):
    """
    체크포인트 기반 reward (시간 기반).

    - 매 스텝: reward = 0 (스텝 카운트만 증가)
    - 체크포인트 통과 시: 도착 보상 + 소요시간 보상 (빠를수록 높음)
    - 한 바퀴 완주 시: 추가 보너스
    - 충돌 시: 큰 페널티
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    collision = bool(obs_raw['collisions'][0])

    line_idx = model.action_to_line_index(action)
    waypoints = waypoints_lines[line_idx]

    position = np.array([x, y], dtype=np.float32)
    nearest_idx = get_nearest_waypoint_idx(position, waypoints)

    if collision:
        return COLLISION_PENALTY, line_idx, nearest_idx

    checkpoint_tracker.tick()

    passed, segment_steps, is_lap_complete = checkpoint_tracker.check(nearest_idx)

    if not passed:
        return 0.0, line_idx, nearest_idx

    # 빠를수록 높은 보상: 스텝이 적을수록 time_ratio가 높음
    time_ratio = max(0.0, 1.0 - segment_steps / BASELINE_STEPS)

    reward = CHECKPOINT_ARRIVAL_REWARD
    reward += SPEED_REWARD_SCALE * time_ratio

    if is_lap_complete:
        reward += LAP_COMPLETION_REWARD

    return reward, line_idx, nearest_idx


# ── Warmup (Pure Pursuit 기반) ────────────────────────────────────────────────
def make_warmup_action(
    obs_raw: dict,
    waypoints_lines: list,
    num_lines: int,
    controller: PurePursuitController,
) -> tuple:
    """
    Returns:
        (sac_action, env_action)
        - sac_action: replay buffer 저장용 [-1, 1]
        - env_action: 실제 env.step용 [steering, speed] (브레이크 적용됨)
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    heading = float(obs_raw['poses_theta'][0])
    current_speed = float(obs_raw['linear_vels_x'][0])
    position = np.array([x, y], dtype=np.float32)

    # 가장 가까운 라인 찾기
    min_dist = float('inf')
    best_line = num_lines // 2
    for i, wp in enumerate(waypoints_lines):
        idx = get_nearest_waypoint_idx(position, wp)
        dist = float(np.linalg.norm(wp[idx] - position))
        if dist < min_dist:
            min_dist = dist
            best_line = i

    # 약간의 탐색: 80%는 최적 라인, 20%는 랜덤 라인
    if np.random.rand() < 0.2:
        line_idx = np.random.randint(0, num_lines)
    else:
        line_idx = best_line

    # Pure Pursuit으로 steering, speed 계산
    waypoints = waypoints_lines[line_idx]
    steering, pp_speed = controller.compute(x, y, heading, current_speed, waypoints)

    # 브레이크 적용
    cmd_speed = apply_brake(current_speed, pp_speed)

    # env action (브레이크 적용된 값)
    env_action = np.array([steering, cmd_speed], dtype=np.float32)

    # SAC action (replay buffer용, 브레이크 전 목표속도 기준)
    if num_lines > 1:
        line_val = (line_idx / (num_lines - 1)) * 2.0 - 1.0
    else:
        line_val = 0.0

    speed_val = ((pp_speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)) * 2.0 - 1.0
    speed_val = float(np.clip(speed_val, -1.0, 1.0))
    sac_action = np.array([line_val, speed_val], dtype=np.float32)

    return sac_action, env_action


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def is_valid_obs(obs: np.ndarray) -> bool:
    return bool(np.isfinite(obs).all())


def build_model(obs_dim: int) -> SAC:
    return SAC(
        obs_dim,
        MODEL_CONFIG['action_dim'],
        MODEL_CONFIG['hidden_dims'],
        MODEL_CONFIG['num_lines'],
    )


def get_wp_progress(obs_raw: dict, reference_waypoints: np.ndarray):
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    lap_count = int(obs_raw['lap_counts'][0])

    position = np.array([x, y], dtype=np.float32)
    nearest_idx = get_nearest_waypoint_idx(position, reference_waypoints)
    n_waypoints = len(reference_waypoints)

    progress_pct = (nearest_idx + 1) / n_waypoints * 100.0
    progress_score = lap_count * n_waypoints + nearest_idx

    return nearest_idx, progress_pct, progress_score


# ── 리플레이 버퍼 ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def push(self, obs, action, reward, next_obs, done):
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

        self.target_entropy = -MODEL_CONFIG['action_dim'] * 0.5

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
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        gamma = TRAIN_CONFIG['gamma']
        tau = TRAIN_CONFIG['tau']

        with torch.no_grad():
            next_action, next_log_prob = self.model.actor.sample(next_obs)

            target_q1 = self.model.target_critic1(next_obs, next_action)
            target_q2 = self.model.target_critic2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob

            target_q = reward + (1.0 - done) * gamma * target_q

        critic1_loss = F.mse_loss(self.model.critic1(obs, action), target_q)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        critic2_loss = F.mse_loss(self.model.critic2(obs, action), target_q)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        new_action, log_prob = self.model.actor.sample(obs)

        q1_new = self.model.critic1(obs, new_action)
        q2_new = self.model.critic2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()

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

        for _ in range(n_episodes):
            obs_raw, _, _, _ = env.reset(poses=init_poses)
            obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

            n_wp = len(waypoints_lines[num_lines // 2])
            eval_tracker = CheckpointTracker(n_wp)

            for _ in range(TRAIN_CONFIG['max_steps']):
                action = self.model.select_action(obs, training=False)

                env_action = action_to_env(
                    action, obs_raw, self.model, waypoints_lines, controller,
                )

                next_obs_raw, _, done, _ = env.step(np.array([env_action]))

                reward, _, _ = compute_reward(
                    next_obs_raw, action, self.model,
                    waypoints_lines, eval_tracker,
                )

                obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
                obs_raw = next_obs_raw
                total_reward += float(reward)

                if done or int(next_obs_raw['lap_counts'][0]) >= 2:
                    break

        return total_reward / n_episodes

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                'model_state': self.model.state_dict(),
                'model_config': MODEL_CONFIG,
                'obs_config': OBS_CONFIG,
                'line_config': LINE_CONFIG,
            },
            path,
        )

        print(f'모델 저장: {path}')

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f'모델 로드: {path}')


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

    num_lines = MODEL_CONFIG['num_lines']

    wp = load_racing_lines()
    waypoints_lines = wp['lines']

    progress_reference_line = waypoints_lines[num_lines // 2]
    n_waypoints = len(progress_reference_line)

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=SPEED_MIN,
    )

    obs_dim = get_obs_dim(OBS_CONFIG['lidar_size'], num_lines)
    trainer = Trainer(obs_dim)

    best_reward = -float('inf')
    total_steps = 0

    init_poses = make_init_pose(waypoints_lines)

    cp_indices = build_checkpoint_indices(n_waypoints)
    print(f'\n체크포인트 {NUM_CHECKPOINTS}개 설정: {cp_indices}')
    print(f'총 waypoint 수: {n_waypoints}')

    print(
        f'\n학습 시작 | map: {ENV_CONFIG["map"]} | '
        f'obs_dim: {obs_dim} | lines: {num_lines}'
    )
    print(f'속도 범위: {SPEED_MIN}~{SPEED_MAX} m/s')
    print(f'브레이크 gain: {BRAKE_GAIN}')
    print(f'baseline_steps: {BASELINE_STEPS}')
    print(f'reward: 체크포인트 도착={CHECKPOINT_ARRIVAL_REWARD}, '
          f'시간 스케일={SPEED_REWARD_SCALE}, '
          f'충돌={COLLISION_PENALTY}, '
          f'완주 보너스={LAP_COMPLETION_REWARD}\n')

    for episode in range(TRAIN_CONFIG['max_episodes']):
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        episode_reward = 0.0
        speeds = []

        best_progress_score = -1
        best_progress_idx = 0
        best_progress_pct = 0.0

        checkpoint_tracker = CheckpointTracker(n_waypoints)

        for _ in range(TRAIN_CONFIG['max_steps']):
            if total_steps < TRAIN_CONFIG['warmup_steps']:
                action, env_action = make_warmup_action(
                    obs_raw, waypoints_lines, num_lines, controller,
                )
            else:
                action = trainer.model.select_action(obs, training=True)
                action = action + np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
                env_action = action_to_env(
                    action, obs_raw, trainer.model, waypoints_lines, controller,
                )

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))
            #############
            #env.render()

            progress_idx, progress_pct, progress_score = get_wp_progress(
                next_obs_raw, progress_reference_line,
            )

            if progress_score > best_progress_score:
                best_progress_score = progress_score
                best_progress_idx = progress_idx
                best_progress_pct = progress_pct

            speeds.append(abs(float(next_obs_raw['linear_vels_x'][0])))

            reward, prev_line_idx, _ = compute_reward(
                next_obs_raw, action, trainer.model,
                waypoints_lines, checkpoint_tracker,
            )

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)

            if is_valid_obs(obs) and is_valid_obs(next_obs):
                trainer.buffer.push(
                    obs, action, reward, next_obs, float(done),
                )

            obs = next_obs
            obs_raw = next_obs_raw
            episode_reward += float(reward)
            total_steps += 1

            if total_steps >= TRAIN_CONFIG['warmup_steps']:
                trainer.update()

            if done or int(next_obs_raw['lap_counts'][0]) >= 2:
                break

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        lap_count = int(obs_raw['lap_counts'][0])

        print(
            f'ep {episode:5d} | '
            f'reward: {episode_reward:8.2f} | '
            f'lap: {lap_count} | '
            f'cp: {checkpoint_tracker.checkpoints_passed}/{NUM_CHECKPOINTS} | '
            f'speed: {avg_speed:.2f} | '
            f'line: {prev_line_idx} | '
            f'steps: {total_steps} | '
            f'wp: {best_progress_idx + 1}/{n_waypoints} '
            f'({best_progress_pct:.1f}%)'
        )

        if episode % TRAIN_CONFIG['eval_interval'] == 0 and episode > 0:
            eval_reward = trainer.evaluate(
                env, waypoints_lines, controller, init_poses,
            )

            print(f'[평가] ep {episode} | eval reward: {eval_reward:.2f}')

            if eval_reward >= best_reward:
                best_reward = eval_reward
                trainer.save(MODEL_SAVE_PATH)

    env.close()
    print('\n학습 완료')


if __name__ == '__main__':
    main()