"""
train_node.py (라인 선택 버전)
──────────────────────────────
SAC가 라인을 선택하고, Pure Pursuit이 추종하는 구조로 학습
설정은 config.py에서 관리
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import gym
import f110_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
<<<<<<< Updated upstream
from sac_model import SAC


# ── Config ────────────────────────────────────────────────────────────────────
ENV_CONFIG = {
    'map'       : os.path.expanduser('~/f1tenth_gym/gym/f110_gym/envs/maps/vegas'),
    'map_ext'   : '.png',
    'num_agents': 1,
    'timestep'  : 0.01,
    'ego_idx'   : 0,
    'params'    : {
        'mu'      : 1.0489,
        'C_Sf'    : 4.718,
        'C_Sr'    : 5.4562,
        'lf'      : 0.15875,
        'lr'      : 0.17145,
        'h'       : 0.074,
        'm'       : 3.74,
        'I'       : 0.04712,
        's_min'   : -0.4189,
        's_max'   : 0.4189,
        'sv_min'  : -3.2,
        'sv_max'  : 3.2,
        'v_switch': 7.319,
        'a_max'   : 9.51,
        'v_min'   : 0.1,
        'v_max'   : 20.0,
        'width'   : 0.31,
        'length'  : 0.58,
    }
}

OBS_CONFIG = {
    'lidar_size'     : 108,
    'lidar_range_min': 0.1,
    'lidar_range_max': 10.0,
}

MODEL_CONFIG = {
    'type'       : 'SAC',
    'hidden_dims': [1024, 512, 1024, 1024, 512, 256],
    'action_dim' : 2,
}

TRAIN_CONFIG = {
    'buffer_size'  : 100000,
    'batch_size'   : 256,
    'gamma'        : 0.99,
    'tau'          : 0.005,
    'lr_actor'     : 3e-4,
    'lr_critic'    : 3e-4,
    'lr_alpha'     : 1e-3,
    'max_episodes' : 1000,
    'max_steps'    : 2000,
    'eval_interval': 50,
    'warmup_steps' : 2000,
}

# SAC action(-1~1) → f110_gym 실제 제어값 변환 범위
ACTION_SPEED_MIN = 0.1    # m/s
ACTION_SPEED_MAX = 20.0   # m/s
ACTION_STEER_MAX = 0.4189 # rad

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__),
                               '../models/sac_model.pth')


# ── action 변환 함수 ──────────────────────────────────────────────────────────
def action_to_env(action: np.ndarray) -> np.ndarray:
    """
    SAC 출력 (-1~1) → f110_gym 입력 [steering_angle, speed]
    f110_gym은 실제 물리값(rad, m/s)을 직접 받음
    """
    steering = float(action[0]) * ACTION_STEER_MAX
    speed_norm = (float(action[1]) + 1.0) / 2.0
    speed = ACTION_SPEED_MIN + speed_norm * (ACTION_SPEED_MAX - ACTION_SPEED_MIN)
    return np.array([steering, speed], dtype=np.float32)


# ── 커스텀 reward 함수 ────────────────────────────────────────────────────────
def compute_reward(next_obs: dict, action: np.ndarray) -> float:
    """
    커스텀 reward 함수
    - 충돌: -100.0 페널티
    - 속도: 빠를수록 높은 보상
    - 조향: 급조향 페널티 (바퀴 떨림 억제)
    """
    speed     = float(next_obs['linear_vels_x'][0])
    collision = bool(next_obs['collisions'][0])

    if collision:
        return -100.0

    # 속도 보상: 빠를수록 높은 reward (0~1)
    speed_reward = speed / ACTION_SPEED_MAX

    # 조향 안정성: 급격한 조향 페널티
    #steer_penalty = -0.1 * abs(float(action[0]))

    return speed_reward #+ steer_penalty


# ── 상태 유효성 검사 ──────────────────────────────────────────────────────────
def is_valid_obs(obs: np.ndarray) -> bool:
    """nan/inf가 포함된 obs는 버퍼에 넣지 않음"""
    return bool(np.isfinite(obs).all())


# ── 모델 팩토리 ───────────────────────────────────────────────────────────────
def build_model(obs_dim: int) -> torch.nn.Module:
    model_type  = MODEL_CONFIG['type']
    action_dim  = MODEL_CONFIG['action_dim']
    hidden_dims = MODEL_CONFIG['hidden_dims']

    if model_type == 'SAC':
        return SAC(obs_dim, action_dim, hidden_dims)
    else:
        raise ValueError(f'지원하지 않는 모델 타입: {model_type}')


# ── 전처리 함수 ───────────────────────────────────────────────────────────────
def preprocess_obs(obs: dict) -> np.ndarray:
    lidar = obs['scans'][0].astype(np.float32)
    lidar = np.where(np.isfinite(lidar), lidar, OBS_CONFIG['lidar_range_max'])
    lidar = np.clip(lidar,
                    OBS_CONFIG['lidar_range_min'],
                    OBS_CONFIG['lidar_range_max'])
    lidar = ((lidar - OBS_CONFIG['lidar_range_min']) /
             (OBS_CONFIG['lidar_range_max'] - OBS_CONFIG['lidar_range_min']))

    size  = OBS_CONFIG['lidar_size']
    step  = max(1, len(lidar) // size)
    lidar = lidar[::step][:size]

    if len(lidar) < size:
        lidar = np.pad(lidar, (0, size - len(lidar)), constant_values=1.0)

    return lidar


def get_obs_dim() -> int:
    return OBS_CONFIG['lidar_size']
=======
from config import (
    ENV_CONFIG, OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG, TRAIN_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH,
)
from sac_model import SAC, build_observation, get_obs_dim
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


# ── 웨이포인트 로드 ───────────────────────────────────────────────────────────
def load_racing_lines() -> dict:
    csv_path = LINE_CONFIG['centerline_csv']
    if os.path.exists(csv_path):
        print(f'centerline CSV 로드: {csv_path}')
        wp = load_waypoints(centerline_path=csv_path,
                            num_lines=LINE_CONFIG['num_lines'],
                            line_spacing=LINE_CONFIG['line_spacing'])
    else:
        print(f'CSV 없음 → 맵 이미지에서 centerline 추출')
        wp = load_waypoints(map_path=LINE_CONFIG['map_path'],
                            map_ext=LINE_CONFIG['map_ext'],
                            num_lines=LINE_CONFIG['num_lines'],
                            line_spacing=LINE_CONFIG['line_spacing'])
    print(f'라인 {len(wp["lines"])}개 생성 완료 (점 수: {len(wp["lines"][0])})')
    return wp


# ── 전처리 ────────────────────────────────────────────────────────────────────
def preprocess_lidar(obs_raw: dict) -> np.ndarray:
    lidar = obs_raw['scans'][0].astype(np.float32)
    lidar = np.where(np.isfinite(lidar), lidar, OBS_CONFIG['lidar_range_max'])
    lidar = np.clip(lidar, OBS_CONFIG['lidar_range_min'], OBS_CONFIG['lidar_range_max'])
    lidar = ((lidar - OBS_CONFIG['lidar_range_min']) /
             (OBS_CONFIG['lidar_range_max'] - OBS_CONFIG['lidar_range_min']))
    size = OBS_CONFIG['lidar_size']
    step = max(1, len(lidar) // size)
    lidar = lidar[::step][:size]
    if len(lidar) < size:
        lidar = np.pad(lidar, (0, size - len(lidar)), constant_values=1.0)
    return lidar


def preprocess_obs(obs_raw, waypoints_lines, num_lines):
    lidar = preprocess_lidar(obs_raw)
    position = np.array([float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])])
    heading = float(obs_raw['poses_theta'][0])
    speed = float(obs_raw['linear_vels_x'][0])
    return build_observation(lidar, position, heading, speed, waypoints_lines, num_lines)


# ── action → 환경 제어 ────────────────────────────────────────────────────────
def action_to_env(action, obs_raw, model, waypoints_lines, controller):
    line_idx = model.action_to_line_index(action)
    waypoints = waypoints_lines[line_idx]
    x, y = float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])
    heading = float(obs_raw['poses_theta'][0])
    speed = float(obs_raw['linear_vels_x'][0])
    steering, pp_speed = controller.compute(x, y, heading, speed, waypoints)
    sac_speed = model.action_to_speed(action, SPEED_MIN, SPEED_MAX)
    return np.array([steering, min(sac_speed, pp_speed)], dtype=np.float32)


# ── Reward ────────────────────────────────────────────────────────────────────
def compute_reward(obs_raw, action, model, waypoints_lines,
                   prev_pos, prev_lap, prev_line_idx, prev_nearest_idx):
    x, y = float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])
    speed = float(obs_raw['linear_vels_x'][0])
    collision = bool(obs_raw['collisions'][0])
    lap_count = int(obs_raw['lap_counts'][0])

    line_idx = model.action_to_line_index(action)
    waypoints = waypoints_lines[line_idx]
    position = np.array([x, y])
    nearest_idx = get_nearest_waypoint_idx(position, waypoints)
    nearest_dist = np.linalg.norm(waypoints[nearest_idx] - position)

    if collision:
        return -10.0, (x, y), lap_count, line_idx, nearest_idx

    N = len(waypoints)
    progress = (nearest_idx - prev_nearest_idx) % N
    if progress > N // 2:
        progress = 0

    reward = (progress * 1.0
              + (abs(speed) / SPEED_MAX) * 0.5
              + (100.0 if lap_count > prev_lap else 0.0)
              - 0.5 * nearest_dist
              + (-0.3 if line_idx != prev_line_idx else 0.0))

    return reward, (x, y), lap_count, line_idx, nearest_idx


# ── Warmup ────────────────────────────────────────────────────────────────────
def simple_policy(obs_raw):
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


def warmup_action_to_sac(obs_raw, waypoints_lines, num_lines):
    x, y = float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])
    position = np.array([x, y])
    min_dist, best_line = float('inf'), num_lines // 2
    for i, wp in enumerate(waypoints_lines):
        idx = get_nearest_waypoint_idx(position, wp)
        dist = np.linalg.norm(wp[idx] - position)
        if dist < min_dist:
            min_dist, best_line = dist, i
    line_val = (best_line / (num_lines - 1)) * 2.0 - 1.0
    env_action = simple_policy(obs_raw)
    speed_val = float(np.clip(((env_action[1] - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)) * 2.0 - 1.0, -1.0, 1.0))
    return np.array([line_val, speed_val], dtype=np.float32)


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def is_valid_obs(obs):
    return bool(np.isfinite(obs).all())

def build_model(obs_dim):
    return SAC(obs_dim, MODEL_CONFIG['action_dim'],
               MODEL_CONFIG['hidden_dims'], MODEL_CONFIG['num_lines'])
>>>>>>> Stashed changes


# ── 리플레이 버퍼 ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (torch.FloatTensor(np.array(obs)),
                torch.FloatTensor(np.array(action)),
                torch.FloatTensor(np.array(reward)).unsqueeze(1),
                torch.FloatTensor(np.array(next_obs)),
                torch.FloatTensor(np.array(done)).unsqueeze(1))
    def __len__(self):
        return len(self.buffer)


# ── Trainer ───────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, obs_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device} | obs_dim: {obs_dim} | num_lines: {MODEL_CONFIG["num_lines"]}')
        self.model = build_model(obs_dim).to(self.device)
        self.buffer = ReplayBuffer(TRAIN_CONFIG['buffer_size'])
        self._init_optimizers()

    def _init_optimizers(self):
        self.actor_opt = optim.Adam(self.model.actor.parameters(), lr=TRAIN_CONFIG['lr_actor'])
        self.critic1_opt = optim.Adam(self.model.critic1.parameters(), lr=TRAIN_CONFIG['lr_critic'])
        self.critic2_opt = optim.Adam(self.model.critic2.parameters(), lr=TRAIN_CONFIG['lr_critic'])
        self.target_entropy = -MODEL_CONFIG['action_dim'] * 0.5
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_opt = optim.Adam([self.log_alpha], lr=TRAIN_CONFIG['lr_alpha'])

    def update(self):
        if len(self.buffer) < TRAIN_CONFIG['batch_size']:
            return
        obs, action, reward, next_obs, done = self.buffer.sample(TRAIN_CONFIG['batch_size'])
        obs, action, reward = obs.to(self.device), action.to(self.device), reward.to(self.device)
        next_obs, done = next_obs.to(self.device), done.to(self.device)
        gamma, tau = TRAIN_CONFIG['gamma'], TRAIN_CONFIG['tau']

        with torch.no_grad():
            na, nlp = self.model.actor.sample(next_obs)
            tq = torch.min(self.model.target_critic1(next_obs, na),
                           self.model.target_critic2(next_obs, na)) - self.alpha * nlp
            tq = reward + (1 - done) * gamma * tq

        for opt, critic in [(self.critic1_opt, self.model.critic1), (self.critic2_opt, self.model.critic2)]:
            loss = F.mse_loss(critic(obs, action), tq)
            opt.zero_grad(); loss.backward(); opt.step()

        new_a, lp = self.model.actor.sample(obs)
        actor_loss = (self.alpha * lp - torch.min(self.model.critic1(obs, new_a), self.model.critic2(obs, new_a))).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        for c, tc in [(self.model.critic1, self.model.target_critic1), (self.model.critic2, self.model.target_critic2)]:
            for p, tp in zip(c.parameters(), tc.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

<<<<<<< Updated upstream
    def evaluate(self, env, n_episodes: int = 5) -> float:
        total_reward = 0.0
        init_poses   = np.array([[0.0, 0.0, np.pi / 2]])
        for _ in range(n_episodes):
            obs, _, _, _ = env.reset(poses=init_poses)
            obs          = preprocess_obs(obs)
            done         = False
            for _ in range(TRAIN_CONFIG['max_steps']):
                action     = self.model.select_action(obs, training=False)
                env_action = action_to_env(action)
                next_obs, reward, done, _ = env.step(np.array([env_action]))

                # 커스텀 reward 적용
                reward = compute_reward(next_obs, action)

                obs          = preprocess_obs(next_obs)
                total_reward += reward
                if done:
=======
    def evaluate(self, env, waypoints_lines, controller, n_episodes=3):
        total_reward, init_poses = 0.0, np.array([[0.0, 0.0, np.pi / 2]])
        num_lines = MODEL_CONFIG['num_lines']
        for _ in range(n_episodes):
            obs_raw, _, _, _ = env.reset(poses=init_poses)
            obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)
            prev_pos = (float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0]))
            prev_lap, prev_line, prev_near = 0, num_lines // 2, 0
            for _ in range(TRAIN_CONFIG['max_steps']):
                action = self.model.select_action(obs, training=False)
                env_action = action_to_env(action, obs_raw, self.model, waypoints_lines, controller)
                next_obs_raw, _, done, _ = env.step(np.array([env_action]))
                reward, prev_pos, prev_lap, prev_line, prev_near = compute_reward(
                    next_obs_raw, action, self.model, waypoints_lines, prev_pos, prev_lap, prev_line, prev_near)
                obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
                obs_raw = next_obs_raw
                total_reward += reward
                if done or int(next_obs_raw['lap_counts'][0]) >= 2:
>>>>>>> Stashed changes
                    break
        return total_reward / n_episodes

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state': self.model.state_dict(), 'model_config': MODEL_CONFIG,
                     'obs_config': OBS_CONFIG, 'line_config': LINE_CONFIG}, path)
        print(f'모델 저장: {path}')

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        print(f'모델 로드: {path}')


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)
    num_lines = MODEL_CONFIG['num_lines']
    wp = load_racing_lines()
    waypoints_lines = wp['lines']
    controller = PurePursuitController(max_speed=SPEED_MAX, min_speed=SPEED_MIN)
    obs_dim = get_obs_dim(OBS_CONFIG['lidar_size'], num_lines)
    trainer = Trainer(obs_dim)
    best_reward, total_steps = -float('inf'), 0
    init_poses = np.array([[0.0, 0.0, np.pi / 2]])

<<<<<<< Updated upstream
    obs_dim     = get_obs_dim()
    trainer     = Trainer(obs_dim)
    best_reward = -float('inf')
    total_steps = 0

    init_poses  = np.array([[0.0, 0.0, np.pi / 2]])

    print(f'\n학습 시작 | map: {ENV_CONFIG["map"]} | obs_dim: {obs_dim}')
    print(f'속도 범위: {ACTION_SPEED_MIN}~{ACTION_SPEED_MAX} m/s | '
          f'조향 범위: ±{ACTION_STEER_MAX} rad\n')

    for episode in range(TRAIN_CONFIG['max_episodes']):

        obs, _, _, _ = env.reset(poses=init_poses)
        obs          = preprocess_obs(obs)
        episode_reward = 0.0
=======
    print(f'\n학습 시작 | map: {ENV_CONFIG["map"]} | obs_dim: {obs_dim} | lines: {num_lines}\n')

    for episode in range(TRAIN_CONFIG['max_episodes']):
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)
        ep_reward, prev_lap = 0.0, 0
        prev_pos = (float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0]))
        prev_line_idx, prev_nearest, collisions, speeds = num_lines // 2, 0, 0, []
>>>>>>> Stashed changes

        for _ in range(TRAIN_CONFIG['max_steps']):
            if total_steps < TRAIN_CONFIG['warmup_steps']:
<<<<<<< Updated upstream
                action = np.array([
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0),
                ], dtype=np.float32)
            else:
                action = trainer.model.select_action(obs, training=True)
                # 탐색 노이즈 추가
                action = action + np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, -1.0, 1.0)

            # SAC(-1~1) → 실제 물리값으로 변환 후 환경에 전달
            env_action = action_to_env(action)
            next_obs, reward, done, _ = env.step(np.array([env_action]))

            # 커스텀 reward 적용
            reward = compute_reward(next_obs, action)

            next_obs_processed = preprocess_obs(next_obs)

            # nan/inf가 섞인 transition은 버퍼에서 제외
            if is_valid_obs(obs) and is_valid_obs(next_obs_processed):
                trainer.buffer.push(obs, action, reward, next_obs_processed, float(done))

            obs             = next_obs_processed
            episode_reward += float(reward)
            total_steps    += 1

            if total_steps >= TRAIN_CONFIG['warmup_steps']:
                trainer.update()

            if done:
                break

        print(f'episode {episode:5d} | '
              f'reward: {episode_reward:8.2f} | '
              f'steps: {total_steps}')
=======
                action = warmup_action_to_sac(obs_raw, waypoints_lines, num_lines)
                env_action = simple_policy(obs_raw)
            else:
                action = trainer.model.select_action(obs, training=True)
                action = np.clip(action + np.random.normal(0, 0.1, size=action.shape), -1.0, 1.0)
                env_action = action_to_env(action, obs_raw, trainer.model, waypoints_lines, controller)

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))
            if bool(next_obs_raw['collisions'][0]): collisions += 1
            speeds.append(abs(float(next_obs_raw['linear_vels_x'][0])))

            reward, prev_pos, prev_lap, prev_line_idx, prev_nearest = compute_reward(
                next_obs_raw, action, trainer.model, waypoints_lines, prev_pos, prev_lap, prev_line_idx, prev_nearest)

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            if is_valid_obs(obs) and is_valid_obs(next_obs):
                trainer.buffer.push(obs, action, reward, next_obs, float(done))

            obs, obs_raw, ep_reward, total_steps = next_obs, next_obs_raw, ep_reward + float(reward), total_steps + 1
            if total_steps >= TRAIN_CONFIG['warmup_steps']:
                trainer.update()
            if done or int(next_obs_raw['lap_counts'][0]) >= 2:
                break

        print(f'ep {episode:5d} | reward: {ep_reward:8.2f} | lap: {prev_lap} | '
              f'speed: {np.mean(speeds) if speeds else 0:.2f} | crash: {collisions} | '
              f'line: {prev_line_idx} | steps: {total_steps}')
>>>>>>> Stashed changes

        if episode % TRAIN_CONFIG['eval_interval'] == 0 and episode > 0:
            ev = trainer.evaluate(env, waypoints_lines, controller)
            print(f'[평가] ep {episode} | eval reward: {ev:.2f}')
            if ev >= best_reward:
                best_reward = ev
                trainer.save(MODEL_SAVE_PATH)

    env.close()
    print('\n학습 완료')

if __name__ == '__main__':
    main()