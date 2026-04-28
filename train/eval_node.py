"""
eval_node.py (라인 선택 버전)
─────────────────────────────
설정은 config.py에서 관리
"""

import os
import sys
import numpy as np
import torch
import gym
import f110_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
<<<<<<< Updated upstream
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
=======
from config import (
    ENV_CONFIG, OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH,
    EVAL_EPISODES, EVAL_MAX_STEPS,
)
from sac_model import SAC, build_observation, get_obs_dim
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


# ── 전처리 ────────────────────────────────────────────────────────────────────
def preprocess_lidar(obs_raw):
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


# ── 평가 지표 ─────────────────────────────────────────────────────────────────
class EvalMetrics:
    def __init__(self):
        self.collisions = 0
        self.speeds = []
        self.steering_angles = []
        self.line_deviations = []
        self.line_switches = 0
        self.total_steps = 0
        self.prev_line_idx = None

    def update(self, obs_raw, steering, speed, line_idx, waypoints_lines):
        self.speeds.append(abs(float(speed)))
        self.steering_angles.append(float(steering))
        if obs_raw['collisions'][0]:
            self.collisions += 1
        x, y = float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])
        wp = waypoints_lines[line_idx]
        nearest_idx = get_nearest_waypoint_idx(np.array([x, y]), wp)
        self.line_deviations.append(np.linalg.norm(wp[nearest_idx] - np.array([x, y])))
        if self.prev_line_idx is not None and line_idx != self.prev_line_idx:
            self.line_switches += 1
        self.prev_line_idx = line_idx
        self.total_steps += 1

    def summary(self, episode):
        avg_speed = np.mean(self.speeds) if self.speeds else 0.0
        wheel_tremor = float(np.mean(np.abs(np.diff(self.steering_angles)))) if len(self.steering_angles) > 1 else 0.0
        avg_deviation = np.mean(self.line_deviations) if self.line_deviations else 0.0
>>>>>>> Stashed changes

        print(f'\n{"─"*40}')
        print(f'에피소드 {episode + 1} 결과')
<<<<<<< Updated upstream
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
=======
        print(f'{"─"*45}')
        print(f'충돌: {self.collisions} | 속도: {avg_speed:.3f} m/s | '
              f'떨림: {wheel_tremor:.4f} | 이탈: {avg_deviation:.4f}m | '
              f'전환: {self.line_switches} | 스텝: {self.total_steps}')

        return {
            'collisions': self.collisions, 'avg_speed': avg_speed,
            'wheel_tremor': wheel_tremor, 'avg_deviation': avg_deviation,
            'line_switches': self.line_switches, 'total_steps': self.total_steps,
>>>>>>> Stashed changes
        }


# ── 레이싱 라인 로드 ──────────────────────────────────────────────────────────
def load_racing_lines():
    csv_path = LINE_CONFIG['centerline_csv']
    if os.path.exists(csv_path):
        wp = load_waypoints(centerline_path=csv_path,
                            num_lines=LINE_CONFIG['num_lines'],
                            line_spacing=LINE_CONFIG['line_spacing'])
        print(f'centerline CSV 로드: {csv_path}')
    else:
        wp = load_waypoints(map_path=LINE_CONFIG['map_path'],
                            num_lines=LINE_CONFIG['num_lines'],
                            line_spacing=LINE_CONFIG['line_spacing'])
        print('맵 이미지에서 centerline 추출')
    print(f'라인 {len(wp["lines"])}개 로드 완료')
    return wp['lines']


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'모델 파일 없음: {path}')
    checkpoint = torch.load(path, map_location='cpu')
<<<<<<< Updated upstream

    # 저장된 config로 모델 재구성
=======
>>>>>>> Stashed changes
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        MODEL_CONFIG.update(checkpoint['model_config'])
    num_lines = MODEL_CONFIG.get('num_lines', LINE_CONFIG['num_lines'])
    obs_dim = get_obs_dim(OBS_CONFIG['lidar_size'], num_lines)
    model = SAC(obs_dim, MODEL_CONFIG['action_dim'],
                MODEL_CONFIG['hidden_dims'], num_lines=num_lines)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f'모델 로드 완료: {path}')
    return model


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print(f'\n평가 시작 | 모델: {MODEL_SAVE_PATH} | 에피소드: {EVAL_EPISODES}\n')

<<<<<<< Updated upstream
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

=======
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)
    model = load_model(MODEL_SAVE_PATH)
    waypoints_lines = load_racing_lines()
    num_lines = MODEL_CONFIG.get('num_lines', LINE_CONFIG['num_lines'])
    controller = PurePursuitController(max_speed=SPEED_MAX, min_speed=SPEED_MIN)
    init_poses = np.array([[0.0, 0.0, np.pi / 2]])
    all_results = []

    for episode in range(EVAL_EPISODES):
        metrics = EvalMetrics()
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        for _ in range(EVAL_MAX_STEPS):
            action = model.select_action(obs, training=False)
            line_idx = model.action_to_line_index(action)
            waypoints = waypoints_lines[line_idx]
            x, y = float(obs_raw['poses_x'][0]), float(obs_raw['poses_y'][0])
            heading = float(obs_raw['poses_theta'][0])
            speed = float(obs_raw['linear_vels_x'][0])

            steering, pp_speed = controller.compute(x, y, heading, speed, waypoints)
            final_speed = min(model.action_to_speed(action, SPEED_MIN, SPEED_MAX), pp_speed)

            next_obs_raw, _, done, _ = env.step(np.array([[steering, final_speed]]))
            metrics.update(next_obs_raw, steering, final_speed, line_idx, waypoints_lines)
            obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            obs_raw = next_obs_raw
>>>>>>> Stashed changes
            if done:
                break

        all_results.append(metrics.summary(episode))

<<<<<<< Updated upstream
    # 전체 에피소드 평균
    print(f'\n{"═"*40}')
    print(f'전체 {N_EPISODES} 에피소드 평균')
    print(f'{"═"*40}')
    print(f'평균 충돌 횟수 : {np.mean([r["collisions"]   for r in all_results]):.1f} 회')
    print(f'평균 속도      : {np.mean([r["avg_speed"]     for r in all_results]):.3f} m/s')
    print(f'평균 바퀴 떨림 : {np.mean([r["wheel_tremor"]  for r in all_results]):.4f} rad/step')
    print(f'{"═"*40}\n')

=======
    print(f'\n{"═"*45}')
    print(f'전체 {EVAL_EPISODES} 에피소드 평균')
    print(f'{"═"*45}')
    for key in ['collisions', 'avg_speed', 'wheel_tremor', 'avg_deviation', 'line_switches']:
        val = np.mean([r[key] for r in all_results])
        print(f'  {key:<16}: {val:.4f}')
    print(f'{"═"*45}\n')
>>>>>>> Stashed changes
    env.close()

if __name__ == '__main__':
    main()