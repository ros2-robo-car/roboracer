"""
train_node.py (라인 선택 버전)
──────────────────────────────
SAC가 라인을 선택하고, Pure Pursuit이 선택된 라인을 추종하는 구조.
학습 설정은 config.py에서 관리.
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
    SPEED_MIN,
    SPEED_MAX,
    MODEL_SAVE_PATH,
)

from sac_model import SAC, build_observation, get_obs_dim
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


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


def preprocess_obs(obs_raw: dict, waypoints_lines: list, num_lines: int) -> np.ndarray:
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
    """
    SAC action:
    action[0] → 라인 선택
    action[1] → 목표 속도

    실제 env action:
    [steering, speed]
    """
    line_idx = model.action_to_line_index(action)
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

    sac_speed = model.action_to_speed(action, SPEED_MIN, SPEED_MAX)

    # SAC가 고른 속도와 Pure Pursuit가 곡률 기반으로 제한한 속도 중 더 작은 값 사용
    target_speed = min(sac_speed, pp_speed)

    return np.array([steering, target_speed], dtype=np.float32)


# ── Reward ────────────────────────────────────────────────────────────────────
def compute_reward(
    obs_raw: dict,
    action: np.ndarray,
    model: SAC,
    waypoints_lines: list,
    prev_pos,
    prev_lap: int,
    prev_line_idx: int,
    prev_nearest_idx: int,
):
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    speed = abs(float(obs_raw['linear_vels_x'][0]))
    collision = bool(obs_raw['collisions'][0])
    lap_count = int(obs_raw['lap_counts'][0])

    line_idx = model.action_to_line_index(action)
    waypoints = waypoints_lines[line_idx]

    position = np.array([x, y], dtype=np.float32)
    nearest_idx = get_nearest_waypoint_idx(position, waypoints)
    nearest_dist = float(np.linalg.norm(waypoints[nearest_idx] - position))

    if collision:
        return -10.0, (x, y), lap_count, line_idx, nearest_idx

    n_waypoints = len(waypoints)

    # 진행도 보상
    progress = (nearest_idx - prev_nearest_idx) % n_waypoints

    # 역방향으로 크게 튄 경우 보상 제거
    if progress > n_waypoints // 2:
        progress = 0

    progress_reward = progress * 1.0

    # 속도 보상
    speed_reward = (speed / SPEED_MAX) * 0.5

    # 랩 완료 보상
    lap_reward = 100.0 if lap_count > prev_lap else 0.0

    # 선택한 라인에서 멀어지면 페널티
    distance_penalty = -0.5 * nearest_dist

    # 라인을 너무 자주 바꾸면 페널티
    line_change_penalty = -0.3 if line_idx != prev_line_idx else 0.0

    reward = (
        progress_reward
        + speed_reward
        + lap_reward
        + distance_penalty
        + line_change_penalty
    )

    return reward, (x, y), lap_count, line_idx, nearest_idx


# ── Warmup ────────────────────────────────────────────────────────────────────
def simple_policy(obs_raw: dict) -> np.ndarray:
    """
    초기 warmup 동안 사용할 단순 주행 정책.
    가장 멀리 뚫린 방향으로 조향하고, 전방 거리에 따라 속도 결정.
    """
    scan = np.array(obs_raw['scans'][0], dtype=np.float32)
    scan = np.where(np.isfinite(scan), scan, 0.0)

    n = len(scan)
    front_scan = scan[n // 4: 3 * n // 4]

    max_idx = int(np.argmax(front_scan))
    mid = len(front_scan) // 2

    steer = float(np.clip((max_idx - mid) / mid * 0.4, -0.4, 0.4))

    front_dist = float(np.mean(front_scan[max(0, mid - 10):mid + 10]))
    speed = float(np.clip(front_dist * 0.8, 1.0, 5.0))

    return np.array([steer, speed], dtype=np.float32)


def warmup_action_to_sac(
    obs_raw: dict,
    waypoints_lines: list,
    num_lines: int,
) -> np.ndarray:
    """
    simple_policy로 실제 env action을 만들되,
    replay buffer에는 SAC action 형식인 [line_select, speed_select]로 저장.
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    position = np.array([x, y], dtype=np.float32)

    min_dist = float('inf')
    best_line = num_lines // 2

    for i, wp in enumerate(waypoints_lines):
        idx = get_nearest_waypoint_idx(position, wp)
        dist = float(np.linalg.norm(wp[idx] - position))

        if dist < min_dist:
            min_dist = dist
            best_line = i

    # line index → -1~1
    if num_lines > 1:
        line_val = (best_line / (num_lines - 1)) * 2.0 - 1.0
    else:
        line_val = 0.0

    env_action = simple_policy(obs_raw)

    # speed → -1~1
    speed_val = ((env_action[1] - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)) * 2.0 - 1.0
    speed_val = float(np.clip(speed_val, -1.0, 1.0))

    return np.array([line_val, speed_val], dtype=np.float32)


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

        # Critic target 계산
        with torch.no_grad():
            next_action, next_log_prob = self.model.actor.sample(next_obs)

            target_q1 = self.model.target_critic1(next_obs, next_action)
            target_q2 = self.model.target_critic2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob

            target_q = reward + (1.0 - done) * gamma * target_q

        # Critic 1 업데이트
        critic1_loss = F.mse_loss(self.model.critic1(obs, action), target_q)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        # Critic 2 업데이트
        critic2_loss = F.mse_loss(self.model.critic2(obs, action), target_q)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # Actor 업데이트
        new_action, log_prob = self.model.actor.sample(obs)

        q1_new = self.model.critic1(obs, new_action)
        q2_new = self.model.critic2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha 업데이트
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()

        # Target critic soft update
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
        n_episodes: int = 3,
    ) -> float:
        total_reward = 0.0
        init_poses = np.array([[0.0, 0.0, np.pi / 2]])
        num_lines = MODEL_CONFIG['num_lines']

        for _ in range(n_episodes):
            obs_raw, _, _, _ = env.reset(poses=init_poses)
            obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

            prev_pos = (
                float(obs_raw['poses_x'][0]),
                float(obs_raw['poses_y'][0]),
            )
            prev_lap = int(obs_raw['lap_counts'][0])
            prev_line_idx = num_lines // 2
            prev_nearest_idx = 0

            for _ in range(TRAIN_CONFIG['max_steps']):
                action = self.model.select_action(obs, training=False)

                env_action = action_to_env(
                    action,
                    obs_raw,
                    self.model,
                    waypoints_lines,
                    controller,
                )

                next_obs_raw, _, done, _ = env.step(np.array([env_action]))

                reward, prev_pos, prev_lap, prev_line_idx, prev_nearest_idx = (
                    compute_reward(
                        next_obs_raw,
                        action,
                        self.model,
                        waypoints_lines,
                        prev_pos,
                        prev_lap,
                        prev_line_idx,
                        prev_nearest_idx,
                    )
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

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=SPEED_MIN,
    )

    obs_dim = get_obs_dim(OBS_CONFIG['lidar_size'], num_lines)
    trainer = Trainer(obs_dim)

    best_reward = -float('inf')
    total_steps = 0

    init_poses = np.array([[0.0, 0.0, np.pi / 2]])

    print(
        f'\n학습 시작 | map: {ENV_CONFIG["map"]} | '
        f'obs_dim: {obs_dim} | lines: {num_lines}'
    )
    print(f'속도 범위: {SPEED_MIN}~{SPEED_MAX} m/s\n')

    for episode in range(TRAIN_CONFIG['max_episodes']):
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        episode_reward = 0.0
        collisions = 0
        speeds = []

        prev_pos = (
            float(obs_raw['poses_x'][0]),
            float(obs_raw['poses_y'][0]),
        )
        prev_lap = int(obs_raw['lap_counts'][0])
        prev_line_idx = num_lines // 2
        prev_nearest_idx = 0

        for _ in range(TRAIN_CONFIG['max_steps']):
            if total_steps < TRAIN_CONFIG['warmup_steps']:
                action = warmup_action_to_sac(
                    obs_raw,
                    waypoints_lines,
                    num_lines,
                )
                env_action = simple_policy(obs_raw)
            else:
                action = trainer.model.select_action(obs, training=True)

                # 탐색 노이즈 추가
                action = action + np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, -1.0, 1.0)

                env_action = action_to_env(
                    action,
                    obs_raw,
                    trainer.model,
                    waypoints_lines,
                    controller,
                )

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))

            if bool(next_obs_raw['collisions'][0]):
                collisions += 1

            speeds.append(abs(float(next_obs_raw['linear_vels_x'][0])))

            reward, prev_pos, prev_lap, prev_line_idx, prev_nearest_idx = (
                compute_reward(
                    next_obs_raw,
                    action,
                    trainer.model,
                    waypoints_lines,
                    prev_pos,
                    prev_lap,
                    prev_line_idx,
                    prev_nearest_idx,
                )
            )

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)

            if is_valid_obs(obs) and is_valid_obs(next_obs):
                trainer.buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    float(done),
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

        print(
            f'ep {episode:5d} | '
            f'reward: {episode_reward:8.2f} | '
            f'lap: {prev_lap} | '
            f'speed: {avg_speed:.2f} | '
            f'crash: {collisions} | '
            f'line: {prev_line_idx} | '
            f'steps: {total_steps}'
        )

        if episode % TRAIN_CONFIG['eval_interval'] == 0 and episode > 0:
            eval_reward = trainer.evaluate(env, waypoints_lines, controller)

            print(f'[평가] ep {episode} | eval reward: {eval_reward:.2f}')

            if eval_reward >= best_reward:
                best_reward = eval_reward
                trainer.save(MODEL_SAVE_PATH)

    env.close()
    print('\n학습 완료')


if __name__ == '__main__':
    main()
