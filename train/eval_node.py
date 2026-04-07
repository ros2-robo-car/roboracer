import os
import sys
import numpy as np
import torch
import gym
import f110_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sac_model import SAC
from train_node import (
    ENV_CONFIG, OBS_CONFIG, MODEL_CONFIG,
    preprocess_obs, get_obs_dim, build_model
)


# ── 설정 ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '../models/sac_model.pth')
N_EPISODES  = 10       # 평가할 에피소드 수
MAX_STEPS   = 1000    # 에피소드당 최대 스텝 수


# ── 평가 지표 계산 ────────────────────────────────────────────────────────────
class EvalMetrics:
    def __init__(self):
        self.collisions      = 0      # 충돌 횟수
        self.speeds          = []     # 매 스텝 속도
        self.steering_angles = []     # 매 스텝 조향각 (바퀴 떨림 계산용)
        self.laps_completed  = 0      # 완주 횟수
        self.total_steps     = 0      # 총 스텝 수

    def update(self, obs, action, done):
        # 속도 기록
        speed = obs['linear_vels_x'][0]
        self.speeds.append(abs(float(speed)))

        # 조향각 기록
        self.steering_angles.append(float(action[0]))

        # 충돌 기록
        if obs['collisions'][0]:
            self.collisions += 1

        self.total_steps += 1

    def summary(self, episode: int):
        avg_speed    = np.mean(self.speeds) if self.speeds else 0.0

        # 바퀴 떨림 = 조향각 변화량의 평균
        if len(self.steering_angles) > 1:
            steering_diff = np.abs(np.diff(self.steering_angles))
            wheel_tremor  = float(np.mean(steering_diff))
        else:
            wheel_tremor  = 0.0

        print(f'\n{"─"*40}')
        print(f'에피소드 {episode + 1} 결과')
        print(f'{"─"*40}')
        print(f'충돌 횟수    : {self.collisions} 회')
        print(f'평균 속도    : {avg_speed:.3f} m/s')
        print(f'바퀴 떨림    : {wheel_tremor:.4f} rad/step')
        print(f'총 스텝      : {self.total_steps}')
        print(f'{"─"*40}\n')

        return {
            'collisions'  : self.collisions,
            'avg_speed'   : avg_speed,
            'wheel_tremor': wheel_tremor,
            'total_steps' : self.total_steps,
        }


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(path: str) -> torch.nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f'모델 파일 없음: {path}')

    checkpoint = torch.load(path, map_location='cpu')

    # 저장된 config로 모델 재구성
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        saved_config = checkpoint['model_config']
        MODEL_CONFIG.update(saved_config)
        print(f'저장된 config 로드: {saved_config}')

    obs_dim = get_obs_dim()
    model   = build_model(obs_dim)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f'모델 로드 완료: {path}')
    return model


# ── 메인 평가 루프 ────────────────────────────────────────────────────────────
def main():
    print('\n평가 시작')
    print(f'모델 경로: {MODEL_PATH}')
    print(f'에피소드 수: {N_EPISODES}\n')

    # 환경 초기화
    env        = gym.make('f110_gym:f110-v0', **ENV_CONFIG)
    model      = load_model(MODEL_PATH)
    init_poses = np.array([[0.0, 0.0, np.pi / 2]])

    # 전체 에피소드 결과 저장
    all_results = []

    for episode in range(N_EPISODES):
        metrics          = EvalMetrics()
        obs, _, _, _     = env.reset(poses=init_poses)
        processed_obs    = preprocess_obs(obs)
        done             = False

        for _ in range(MAX_STEPS):
            # 추론 (training=False → 최적 행동)
            action = model.select_action(processed_obs, training=False)

            next_obs, reward, done, _ = env.step(np.array([action]))

            # 평가 지표 업데이트 (원본 obs 사용)
            metrics.update(next_obs, action, done)

            processed_obs = preprocess_obs(next_obs)

            if done:
                break

        result = metrics.summary(episode)
        all_results.append(result)

    # 전체 에피소드 평균
    print(f'\n{"═"*40}')
    print(f'전체 {N_EPISODES} 에피소드 평균')
    print(f'{"═"*40}')
    print(f'평균 충돌 횟수 : {np.mean([r["collisions"]   for r in all_results]):.1f} 회')
    print(f'평균 속도      : {np.mean([r["avg_speed"]     for r in all_results]):.3f} m/s')
    print(f'평균 바퀴 떨림 : {np.mean([r["wheel_tremor"]  for r in all_results]):.4f} rad/step')
    print(f'{"═"*40}\n')

    env.close()


if __name__ == '__main__':
    main()