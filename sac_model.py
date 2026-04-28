"""
sac_model.py (라인 선택 버전)
─────────────────────────────
SAC (Soft Actor-Critic) 모델

변경 사항 (기존 대비):
  - Actor 출력: [조향각, 속도] → [라인 선택(-1~1), 목표 속도(-1~1)]
  - 라인 선택값은 외부에서 라인 인덱스로 변환
  - obs에 라인 관련 정보가 추가됨을 전제
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# 행동 범위 설정
LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON     = 1e-6


class Actor(nn.Module):
    """
    상태(observation)를 입력받아 행동(action)을 출력하는 신경망
    출력: [라인 선택, 목표 속도] (연속적인 값, -1~1)
    """
    def __init__(self, obs_dim, action_dim=2,
                 hidden_dims=[1024, 512, 1024, 1024, 512, 256]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)

        # 평균과 표준편차를 따로 출력 (확률적 행동 선택)
        self.mean_layer    = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        x = self.net(state)

        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        """
        학습 중 행동 샘플링 (탐색용)
        reparameterization trick 사용
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 정규분포에서 샘플링
        normal = Normal(mean, std)
        x = normal.rsample()
        x = torch.clamp(x, -6.0, 6.0)

        # tanh로 -1~1 범위로 압축
        action = torch.tanh(x)

        # log probability 계산 (학습에 필요)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state):
        """
        실제 주행 시 행동 선택 (추론용)
        """
        mean, _ = self.forward(state)
        action  = torch.tanh(mean)
        return action


class Critic(nn.Module):
    """
    상태(observation) + 행동(action)을 입력받아 Q값을 출력하는 신경망
    Q값: 해당 행동이 얼마나 좋은지 나타내는 점수
    """
    def __init__(self, obs_dim, action_dim=2,
                 hidden_dims=[1024, 512, 1024, 1024, 512, 256]):
        super().__init__()

        layers = []
        in_dim = obs_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SAC(nn.Module):
    """
    SAC (Soft Actor-Critic) 전체 모델
    Actor 1개 + Critic 2개로 구성

    action 해석:
        action[0]: 라인 선택 (-1~1 → 라인 인덱스로 변환)
        action[1]: 목표 속도 (-1~1 → 실제 속도로 변환)
    """
    def __init__(self, obs_dim, action_dim=2,
                 hidden_dims=[1024, 512, 1024, 1024, 512, 256],
                 num_lines=5):
        super().__init__()

        self.num_lines  = num_lines
        self.action_dim = action_dim

        self.actor   = Actor(obs_dim, action_dim, hidden_dims)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dims)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dims)

        # Target Critic (안정적인 학습을 위해 사용)
        self.target_critic1 = Critic(obs_dim, action_dim, hidden_dims)
        self.target_critic2 = Critic(obs_dim, action_dim, hidden_dims)

        # Target Critic을 Critic과 동일하게 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, state, training=True):
        """
        행동 선택

        training=True  → 탐색 (학습 중)
        training=False → 최적 행동 (실제 주행)

        반환: np.ndarray [line_select, speed_select]
              둘 다 -1~1 범위
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            device = next(self.parameters()).device
            state = state.to(device)
            if training:
                action, _ = self.actor.sample(state)
            else:
                action = self.actor.get_action(state)
        return action.squeeze(0).cpu().numpy()

    def action_to_line_index(self, action) -> int:
        """
        연속 action[0] (-1~1) → 라인 인덱스 (0 ~ num_lines-1)

        예시 (num_lines=5):
            -1.0 ~ -0.6 → 0 (가장 안쪽)
            -0.6 ~ -0.2 → 1
            -0.2 ~  0.2 → 2 (중앙)
             0.2 ~  0.6 → 3
             0.6 ~  1.0 → 4 (가장 바깥)
        """
        raw = float(action[0]) if hasattr(action, '__len__') else float(action)
        # -1~1 → 0~1
        normalized = (raw + 1.0) / 2.0
        # 0~1 → 0~num_lines-1
        idx = int(normalized * self.num_lines)
        return min(max(idx, 0), self.num_lines - 1)

    def action_to_speed(self, action,
                        min_speed: float = 0.5,
                        max_speed: float = 3.0) -> float:
        """
        연속 action[1] (-1~1) → 실제 속도 (min_speed ~ max_speed)
        """
        raw = float(action[1]) if hasattr(action, '__len__') else float(action)
        normalized = (raw + 1.0) / 2.0
        return min_speed + normalized * (max_speed - min_speed)


# ══════════════════════════════════════════════════════════════════════════════
# 관측(observation) 구성 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def build_observation(lidar: np.ndarray,
                      position: np.ndarray,
                      heading: float,
                      speed: float,
                      waypoints_lines: list,
                      num_lines: int = 5,
                      nearest_points: int = 3) -> np.ndarray:
    """
    SAC 입력용 관측 벡터 구성

    구성:
        - LiDAR 데이터 (108차원)
        - 차량 속도 (1차원)
        - 각 라인별 가장 가까운 점까지의 거리 (num_lines차원)
        - 각 라인별 가장 가까운 점의 상대 각도 (num_lines차원)

    Args:
        lidar:           전처리된 LiDAR (108,)
        position:        차량 위치 [x, y]
        heading:         차량 방향 (rad)
        speed:           현재 속도 (m/s)
        waypoints_lines: 라인별 웨이포인트 리스트
        num_lines:       라인 수
        nearest_points:  참조할 가까운 점 수

    Returns:
        obs: np.ndarray (108 + 1 + num_lines * 2,)
    """
    obs_parts = [lidar]

    # 속도 정규화 (0~20 m/s → 0~1)
    obs_parts.append(np.array([speed / 20.0], dtype=np.float32))

    # 각 라인별 거리 및 각도
    for line_idx in range(num_lines):
        if line_idx < len(waypoints_lines):
            wp = waypoints_lines[line_idx]
            diffs = wp - position[:2]
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]

            # 상대 각도 계산
            dx = wp[nearest_idx, 0] - position[0]
            dy = wp[nearest_idx, 1] - position[1]
            abs_angle = np.arctan2(dy, dx)
            rel_angle = abs_angle - heading
            # 정규화 (-pi ~ pi → -1 ~ 1)
            rel_angle = ((rel_angle + np.pi) % (2 * np.pi) - np.pi) / np.pi

            # 거리 정규화 (0~10m → 0~1)
            nearest_dist = min(nearest_dist / 10.0, 1.0)

            obs_parts.append(np.array([nearest_dist, rel_angle],
                                       dtype=np.float32))
        else:
            obs_parts.append(np.array([1.0, 0.0], dtype=np.float32))

    return np.concatenate(obs_parts)


def get_obs_dim(lidar_size: int = 108, num_lines: int = 5) -> int:
    """
    관측 벡터 차원 계산
    = LiDAR + 속도 + (거리 + 각도) × 라인 수
    """
    return lidar_size + 1 + num_lines * 2