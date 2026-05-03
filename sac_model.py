"""
sac_model.py
──────────────────────────────
SAC (Soft Actor-Critic) 모델

Hybrid SAC 구조:
- line 선택: 이산 action
- speed 제어: 연속 action

곡률 feature를 사용할 경우 observation 뒤에 line별 lookahead curvature가 추가된다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

# 행동 범위 설정
LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6


class HybridActor(nn.Module):
    """
    Hybrid Actor: 이산(라인 선택) + 연속(속도 배율)

    출력:
      - line_logits: (num_lines,) → softmax → Categorical 분포
      - speed_mean, speed_log_std: (1,) → Normal 분포 → tanh
    """
    def __init__(self, obs_dim, num_lines=5,
                 hidden_dims=[1024, 512, 1024, 1024, 512, 256]):
        super().__init__()
        self.num_lines = num_lines

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)

        # 이산: 라인 선택 (num_lines개 logits)
        self.line_logits_layer = nn.Linear(hidden_dims[-1], num_lines)

        # 연속: pp_speed 배율 조절용 speed action (mean, log_std 각 1차원)
        self.speed_mean_layer = nn.Linear(hidden_dims[-1], 1)
        self.speed_log_std_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state):
        x = self.net(state)

        line_logits = self.line_logits_layer(x)
        line_logits = torch.clamp(line_logits, -10, 10)

        speed_mean = self.speed_mean_layer(x)
        speed_mean = torch.clamp(speed_mean, -5, 5)

        speed_log_std = self.speed_log_std_layer(x)
        speed_log_std = torch.clamp(speed_log_std, LOG_STD_MIN, LOG_STD_MAX)

        return line_logits, speed_mean, speed_log_std

    def sample(self, state):
        """
        학습 중 행동 샘플링.

        Returns:
            line_idx:       (batch,)           선택된 라인 인덱스
            speed_action:   (batch, 1)         tanh 적용된 속도 배율 action (-1~1)
            log_prob:       (batch, 1)         전체 log probability
            line_probs:     (batch, num_lines) 라인 선택 확률
        """
        line_logits, speed_mean, speed_log_std = self.forward(state)

        # ── 이산: 라인 선택 ──
        line_probs = F.softmax(line_logits, dim=-1)
        line_dist = Categorical(line_probs)
        line_idx = line_dist.sample()
        line_log_prob = line_dist.log_prob(line_idx)

        # ── 연속: 속도 배율 action ──
        speed_std = speed_log_std.exp()
        speed_dist = Normal(speed_mean, speed_std)
        speed_raw = speed_dist.rsample()
        speed_raw = torch.clamp(speed_raw, -6.0, 6.0)
        speed_action = torch.tanh(speed_raw)

        speed_log_prob = speed_dist.log_prob(speed_raw)
        speed_log_prob -= torch.log(1 - speed_action.pow(2) + EPSILON)
        speed_log_prob = speed_log_prob.sum(dim=-1)

        total_log_prob = (line_log_prob + speed_log_prob).unsqueeze(-1)

        return line_idx, speed_action, total_log_prob, line_probs

    def get_action(self, state):
        """
        평가/실제 주행 시 행동 선택.
        line은 argmax, speed는 mean의 tanh를 사용한다.
        """
        line_logits, speed_mean, _ = self.forward(state)

        line_idx = torch.argmax(line_logits, dim=-1)
        speed_action = torch.tanh(speed_mean)

        return line_idx, speed_action

    def evaluate_actions(self, state, line_idx, speed_action):
        """
        주어진 state에서 특정 action의 log_prob를 계산한다.
        """
        line_logits, speed_mean, speed_log_std = self.forward(state)

        line_probs = F.softmax(line_logits, dim=-1)
        line_dist = Categorical(line_probs)
        line_log_prob = line_dist.log_prob(line_idx)

        speed_action_clamped = torch.clamp(
            speed_action,
            -1.0 + EPSILON,
            1.0 - EPSILON,
        )
        speed_raw = torch.atanh(speed_action_clamped)

        speed_std = speed_log_std.exp()
        speed_dist = Normal(speed_mean, speed_std)
        speed_log_prob = speed_dist.log_prob(speed_raw)
        speed_log_prob -= torch.log(1 - speed_action.pow(2) + EPSILON)
        speed_log_prob = speed_log_prob.sum(dim=-1)

        total_log_prob = (line_log_prob + speed_log_prob).unsqueeze(-1)

        line_entropy = line_dist.entropy()
        speed_entropy = speed_dist.entropy().sum(dim=-1)

        return total_log_prob, line_probs, line_entropy + speed_entropy


class Critic(nn.Module):
    """
    상태(observation) + 행동을 입력받아 Q값을 출력하는 신경망.

    action 입력: [line_onehot(num_lines), speed_action(1)]
    """
    def __init__(self, obs_dim, num_lines=5,
                 hidden_dims=[1024, 512, 1024, 1024, 512, 256]):
        super().__init__()

        action_input_dim = num_lines + 1

        layers = []
        in_dim = obs_dim + action_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action_encoded):
        x = torch.cat([state, action_encoded], dim=-1)
        return self.net(x)


def encode_action(line_idx, speed_action, num_lines, device):
    """
    line_idx (batch,) + speed_action (batch, 1) → (batch, num_lines+1)
    Critic에 넣기 위한 action 인코딩.
    """
    batch_size = line_idx.shape[0]
    line_onehot = torch.zeros(batch_size, num_lines, device=device)
    line_onehot.scatter_(1, line_idx.long().unsqueeze(1), 1.0)

    if speed_action.dim() == 1:
        speed_action = speed_action.unsqueeze(1)

    return torch.cat([line_onehot, speed_action], dim=-1)


class SAC(nn.Module):
    """
    Hybrid SAC.

    action 해석:
        action[0]: 라인 인덱스
        action[1]: pp_speed 기준 배율 조절값 (-1~1)
    """
    def __init__(self, obs_dim, action_dim=2,
                 hidden_dims=[1024, 512, 1024, 1024, 512, 256],
                 num_lines=5):
        super().__init__()

        self.num_lines = num_lines
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.actor = HybridActor(obs_dim, num_lines, hidden_dims)
        self.critic1 = Critic(obs_dim, num_lines, hidden_dims)
        self.critic2 = Critic(obs_dim, num_lines, hidden_dims)

        self.target_critic1 = Critic(obs_dim, num_lines, hidden_dims)
        self.target_critic2 = Critic(obs_dim, num_lines, hidden_dims)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, state, training=True):
        """
        행동 선택.

        training=True  → 탐색 샘플링
        training=False → 결정적 선택

        반환:
            np.ndarray [line_idx, speed_action]
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            device = next(self.parameters()).device
            state_t = state_t.to(device)

            if training:
                line_idx, speed_action, _, _ = self.actor.sample(state_t)
            else:
                line_idx, speed_action = self.actor.get_action(state_t)

            line_idx_np = line_idx.squeeze(0).cpu().numpy()
            speed_np = speed_action.squeeze(0).cpu().numpy()

        return np.array([float(line_idx_np), float(speed_np)], dtype=np.float32)

    def action_to_line_index(self, action) -> int:
        idx = int(round(float(action[0]) if hasattr(action, '__len__') else float(action)))
        return min(max(idx, 0), self.num_lines - 1)

    def action_to_speed(self, action,
                        min_speed: float = 0.5,
                        max_speed: float = 3.0) -> float:
        """
        legacy utility.
        현재 train/eval에서는 action[1]을 절대 속도가 아니라 pp_speed 배율로 해석한다.
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
                      nearest_points: int = 3,
                      line_curvatures: np.ndarray = None,
                      use_line_curvature: bool = False) -> np.ndarray:
    """
    SAC 입력용 관측 벡터 구성.

    기본 구성:
        - LiDAR 데이터 (lidar_size)
        - 차량 속도 (1)
        - 각 line별 가장 가까운 점까지의 거리 (num_lines)
        - 각 line별 가장 가까운 점의 상대 각도 (num_lines)

    use_line_curvature=True이면 뒤에 추가:
        - 각 line별 lookahead curvature (num_lines)

    Returns:
        obs: np.ndarray
    """
    obs_parts = [lidar.astype(np.float32)]

    # 속도 정규화. 환경 v_max와 별개로 기존 학습 기준을 유지한다.
    obs_parts.append(np.array([speed / 20.0], dtype=np.float32))

    for line_idx in range(num_lines):
        if line_idx < len(waypoints_lines):
            wp = waypoints_lines[line_idx]
            diffs = wp - position[:2]
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_idx])

            dx = float(wp[nearest_idx, 0] - position[0])
            dy = float(wp[nearest_idx, 1] - position[1])
            abs_angle = np.arctan2(dy, dx)
            rel_angle = abs_angle - heading
            rel_angle = ((rel_angle + np.pi) % (2 * np.pi) - np.pi) / np.pi

            nearest_dist = min(nearest_dist / 10.0, 1.0)

            obs_parts.append(np.array([nearest_dist, rel_angle], dtype=np.float32))
        else:
            obs_parts.append(np.array([1.0, 0.0], dtype=np.float32))

    if use_line_curvature:
        if line_curvatures is None:
            line_curvatures = np.zeros(num_lines, dtype=np.float32)
        else:
            line_curvatures = np.asarray(line_curvatures, dtype=np.float32)

            if len(line_curvatures) < num_lines:
                line_curvatures = np.pad(
                    line_curvatures,
                    (0, num_lines - len(line_curvatures)),
                    constant_values=0.0,
                )
            elif len(line_curvatures) > num_lines:
                line_curvatures = line_curvatures[:num_lines]

        obs_parts.append(line_curvatures.astype(np.float32))

    return np.concatenate(obs_parts).astype(np.float32)


def get_obs_dim(
    lidar_size: int = 108,
    num_lines: int = 5,
    use_line_curvature: bool = False,
) -> int:
    """
    관측 벡터 차원 계산.

    기본:
        LiDAR + 속도 + (거리 + 각도) × line 수

    curvature 사용 시:
        + line별 curvature × line 수
    """
    obs_dim = lidar_size + 1 + num_lines * 2

    if use_line_curvature:
        obs_dim += num_lines

    return obs_dim
