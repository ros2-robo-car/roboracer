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
        'v_max'   : 20.0,        # [수정 1] 수치 안정성을 위해 실제 제어 속도는 별도 스케일링
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
    'type'      : 'SAC',
    'hidden_dim': 256,
    'action_dim': 2,
}

TRAIN_CONFIG = {
    'buffer_size'  : 10000,
    'batch_size'   : 64,
    'gamma'        : 0.99,
    'tau'          : 0.005,
    'lr_actor'     : 3e-4,
    'lr_critic'    : 3e-4,
    'lr_alpha'     : 3e-4,
    'max_episodes' : 200,
    'max_steps'    : 2000,
    'eval_interval': 50,
    'warmup_steps' : 500,
}

# [수정 2] SAC action(-1~1) → f110_gym 실제 제어값 변환 범위
# v_max를 20.0으로 두되, 실제 학습 중엔 안전한 범위만 사용
ACTION_SPEED_MIN = 0.1    # m/s
ACTION_SPEED_MAX = 10.0    # m/s  ← 수치 폭발 방지를 위해 낮게 설정
ACTION_STEER_MAX = 0.4189 # rad

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__),
                               '../models/sac_model.pth')


# ── action 변환 함수 ──────────────────────────────────────────────────────────
def action_to_env(action: np.ndarray) -> np.ndarray:
    """
    SAC 출력 (-1~1) → f110_gym 입력 [steering_angle, speed]
    f110_gym은 실제 물리값(rad, m/s)을 직접 받음
    """
    steering = float(action[0]) * ACTION_STEER_MAX          # -0.4189 ~ 0.4189 rad
    speed_norm = (float(action[1]) + 1.0) / 2.0             # 0 ~ 1
    speed = ACTION_SPEED_MIN + speed_norm * (ACTION_SPEED_MAX - ACTION_SPEED_MIN)
    return np.array([steering, speed], dtype=np.float32)


# ── 상태 유효성 검사 ──────────────────────────────────────────────────────────
def is_valid_obs(obs: np.ndarray) -> bool:
    """nan/inf가 포함된 obs는 버퍼에 넣지 않음"""
    return bool(np.isfinite(obs).all())


# ── 모델 팩토리 ───────────────────────────────────────────────────────────────
def build_model(obs_dim: int) -> torch.nn.Module:
    model_type = MODEL_CONFIG['type']
    action_dim = MODEL_CONFIG['action_dim']
    hidden_dim = MODEL_CONFIG['hidden_dim']

    if model_type == 'SAC':
        return SAC(obs_dim, action_dim, hidden_dim)
    else:
        raise ValueError(f'지원하지 않는 모델 타입: {model_type}')


# ── 전처리 함수 ───────────────────────────────────────────────────────────────
def preprocess_obs(obs: dict) -> np.ndarray:
    lidar = obs['scans'][0].astype(np.float32)
    lidar = np.where(np.isfinite(lidar), lidar, OBS_CONFIG['lidar_range_max'])  # [수정 3] nan/inf → max로 대체
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


# ── 학습 클래스 ───────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, obs_dim: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device    : {self.device}')
        print(f'model     : {MODEL_CONFIG["type"]}')
        print(f'obs_dim   : {obs_dim}')
        print(f'action_dim: {MODEL_CONFIG["action_dim"]}')

        self.model  = build_model(obs_dim).to(self.device)
        self.buffer = ReplayBuffer(TRAIN_CONFIG['buffer_size'])

        if MODEL_CONFIG['type'] == 'SAC':
            self._init_sac_optimizers()

    def _init_sac_optimizers(self):
        self.actor_optimizer   = optim.Adam(
            self.model.actor.parameters(),   lr=TRAIN_CONFIG['lr_actor'])
        self.critic1_optimizer = optim.Adam(
            self.model.critic1.parameters(), lr=TRAIN_CONFIG['lr_critic'])
        self.critic2_optimizer = optim.Adam(
            self.model.critic2.parameters(), lr=TRAIN_CONFIG['lr_critic'])

        self.target_entropy  = -MODEL_CONFIG['action_dim']
        self.log_alpha       = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha           = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=TRAIN_CONFIG['lr_alpha'])

    def update(self):
        if len(self.buffer) < TRAIN_CONFIG['batch_size']:
            return

        obs, action, reward, next_obs, done = self.buffer.sample(
            TRAIN_CONFIG['batch_size'])
        obs      = obs.to(self.device)
        action   = action.to(self.device)
        reward   = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done     = done.to(self.device)

        gamma = TRAIN_CONFIG['gamma']
        tau   = TRAIN_CONFIG['tau']

        with torch.no_grad():
            next_action, next_log_prob = self.model.actor.sample(next_obs)
            target_q1 = self.model.target_critic1(next_obs, next_action)
            target_q2 = self.model.target_critic2(next_obs, next_action)
            target_q  = (torch.min(target_q1, target_q2)
                         - self.alpha * next_log_prob)
            target_q  = reward + (1 - done) * gamma * target_q

        for optimizer, critic in [
            (self.critic1_optimizer, self.model.critic1),
            (self.critic2_optimizer, self.model.critic2),
        ]:
            loss = F.mse_loss(critic(obs, action), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        new_action, log_prob = self.model.actor.sample(obs)
        q1         = self.model.critic1(obs, new_action)
        q2         = self.model.critic2(obs, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha *
                       (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        for critic, target in [
            (self.model.critic1, self.model.target_critic1),
            (self.model.critic2, self.model.target_critic2),
        ]:
            for p, tp in zip(critic.parameters(), target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def evaluate(self, env, n_episodes: int = 5) -> float:
        total_reward = 0.0
        init_poses   = np.array([[0.0, 0.0, np.pi / 2]])
        for _ in range(n_episodes):
            obs, _, _, _ = env.reset(poses=init_poses)
            obs          = preprocess_obs(obs)
            done         = False
            for _ in range(TRAIN_CONFIG['max_steps']):
                action     = self.model.select_action(obs, training=False)
                env_action = action_to_env(action)                        # [수정 2 적용]
                obs, reward, done, _ = env.step(np.array([env_action]))
                obs          = preprocess_obs(obs)
                total_reward += reward
                if done:
                    break
        return total_reward / n_episodes

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state' : self.model.state_dict(),
            'model_config': MODEL_CONFIG,
            'obs_config'  : OBS_CONFIG,
        }, path)
        print(f'모델 저장: {path}')

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f'모델 로드: {path}')


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────
def main():
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

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

        for _ in range(TRAIN_CONFIG['max_steps']):

            if total_steps < TRAIN_CONFIG['warmup_steps']:
                action = np.array([
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0),
                ], dtype=np.float32)
            else:
                action = trainer.model.select_action(obs, training=True)

            # [수정 2 적용] SAC(-1~1) → 실제 물리값으로 변환 후 환경에 전달
            env_action = action_to_env(action)
            next_obs, reward, done, _ = env.step(np.array([env_action]))
            next_obs_processed = preprocess_obs(next_obs)

            # [수정 3 적용] nan/inf가 섞인 transition은 버퍼에서 제외
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

        if episode % TRAIN_CONFIG['eval_interval'] == 0 and episode > 0:
            eval_reward = trainer.evaluate(env)
            print(f'[평가] episode {episode} | eval reward: {eval_reward:.2f}')

            if eval_reward >= best_reward:
                best_reward = eval_reward
                trainer.save(MODEL_SAVE_PATH)
                print(f'최고 성능 갱신: {best_reward:.2f}')

    env.close()
    print('\n학습 완료')


if __name__ == '__main__':
    main()