import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 행동 범위 설정
LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON     = 1e-6


class Actor(nn.Module):
    """
    상태(observation)를 입력받아 행동(action)을 출력하는 신경망
    출력: 조향각, 속도 (연속적인 값)
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=[1024, 512, 1024, 1024, 512, 256]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)

        # 평균과 표준편차를 따로 출력 (확률적 행동 선택)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
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
        x = torch.clamp(x, -6.0, 6.0)  # 수치 폭발 방지

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
    def __init__(self, obs_dim, action_dim, hidden_dims=[1024, 512, 1024, 1024, 512, 256]):
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
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=[1024, 512, 1024, 1024, 512, 256]):
        super().__init__()

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