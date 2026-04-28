"""
eval_node.py

학습된 SAC 모델 평가 코드.
SAC는 주행 라인과 목표 속도를 선택하고,
Pure Pursuit는 선택된 라인을 따라가기 위한 조향각을 계산한다.
"""

import os
import sys

import gym
import f110_gym
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import (
    ENV_CONFIG,
    OBS_CONFIG,
    LINE_CONFIG,
    MODEL_CONFIG,
    SPEED_MIN,
    SPEED_MAX,
    MODEL_SAVE_PATH,
    EVAL_EPISODES,
    EVAL_MAX_STEPS,
)

from sac_model import SAC, build_observation, get_obs_dim
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController


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


# ── 평가 지표 ─────────────────────────────────────────────────────────────────
class EvalMetrics:
    def __init__(self):
        self.collisions = 0
        self.speeds = []
        self.line_deviations = []
        self.line_switches = 0
        self.total_steps = 0
        self.laps_completed = 0
        self.prev_line_idx = None

    def update(
        self,
        obs_raw: dict,
        speed: float,
        line_idx: int,
        waypoints_lines: list,
    ):
        self.speeds.append(abs(float(speed)))

        if bool(obs_raw['collisions'][0]):
            self.collisions += 1

        self.laps_completed = int(obs_raw['lap_counts'][0])

        x = float(obs_raw['poses_x'][0])
        y = float(obs_raw['poses_y'][0])
        position = np.array([x, y], dtype=np.float32)

        waypoints = waypoints_lines[line_idx]
        nearest_idx = get_nearest_waypoint_idx(position, waypoints)
        deviation = float(np.linalg.norm(waypoints[nearest_idx] - position))
        self.line_deviations.append(deviation)

        if self.prev_line_idx is not None and line_idx != self.prev_line_idx:
            self.line_switches += 1

        self.prev_line_idx = line_idx
        self.total_steps += 1

    def summary(self, episode: int) -> dict:
        avg_speed = float(np.mean(self.speeds)) if self.speeds else 0.0
        avg_deviation = (
            float(np.mean(self.line_deviations))
            if self.line_deviations
            else 0.0
        )

        print(f'\n{"─" * 45}')
        print(f'에피소드 {episode + 1} 결과')
        print(f'{"─" * 45}')
        print(f'충돌 횟수     : {self.collisions}')
        print(f'완주 lap      : {self.laps_completed}')
        print(f'평균 속도     : {avg_speed:.3f} m/s')
        print(f'평균 라인 오차: {avg_deviation:.4f} m')
        print(f'라인 전환 횟수: {self.line_switches}')
        print(f'총 스텝       : {self.total_steps}')
        print(f'{"─" * 45}\n')

        return {
            'collisions': self.collisions,
            'laps_completed': self.laps_completed,
            'avg_speed': avg_speed,
            'avg_deviation': avg_deviation,
            'line_switches': self.line_switches,
            'total_steps': self.total_steps,
        }


# ── 레이싱 라인 로드 ──────────────────────────────────────────────────────────
def load_racing_lines() -> list:
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

    print(f'라인 {len(wp["lines"])}개 로드 완료')
    return wp['lines']


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(path: str) -> SAC:
    if not os.path.exists(path):
        raise FileNotFoundError(f'모델 파일 없음: {path}')

    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        MODEL_CONFIG.update(checkpoint['model_config'])

    num_lines = MODEL_CONFIG.get('num_lines', LINE_CONFIG['num_lines'])
    obs_dim = get_obs_dim(OBS_CONFIG['lidar_size'], num_lines)

    model = SAC(
        obs_dim,
        MODEL_CONFIG['action_dim'],
        MODEL_CONFIG['hidden_dims'],
        num_lines=num_lines,
    )

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f'모델 로드 완료: {path}')
    print(f'obs_dim: {obs_dim} | num_lines: {num_lines}')

    return model


# ── action → 환경 제어 ────────────────────────────────────────────────────────
def action_to_env(
    action: np.ndarray,
    obs_raw: dict,
    model: SAC,
    waypoints_lines: list,
    controller: PurePursuitController,
):
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
    target_speed = min(sac_speed, pp_speed)

    return steering, target_speed, line_idx


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print(f'\n평가 시작 | 모델: {MODEL_SAVE_PATH} | 에피소드: {EVAL_EPISODES}\n')

    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

    waypoints_lines = load_racing_lines()
    num_lines = MODEL_CONFIG.get('num_lines', LINE_CONFIG['num_lines'])

    model = load_model(MODEL_SAVE_PATH)

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=SPEED_MIN,
    )

    init_poses = np.array([[0.0, 0.0, np.pi / 2]])

    all_results = []

    for episode in range(EVAL_EPISODES):
        metrics = EvalMetrics()

        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        for _ in range(EVAL_MAX_STEPS):
            action = model.select_action(obs, training=False)

            steering, target_speed, line_idx = action_to_env(
                action,
                obs_raw,
                model,
                waypoints_lines,
                controller,
            )

            env_action = np.array([[steering, target_speed]], dtype=np.float32)
            next_obs_raw, _, done, _ = env.step(env_action)

            metrics.update(
                next_obs_raw,
                target_speed,
                line_idx,
                waypoints_lines,
            )

            obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            obs_raw = next_obs_raw

            if done or int(next_obs_raw['lap_counts'][0]) >= 2:
                break

        all_results.append(metrics.summary(episode))

    print(f'\n{"═" * 45}')
    print(f'전체 {EVAL_EPISODES} 에피소드 평균')
    print(f'{"═" * 45}')
    print(
        f'평균 충돌 횟수 : '
        f'{np.mean([r["collisions"] for r in all_results]):.2f}'
    )
    print(
        f'평균 완주 lap  : '
        f'{np.mean([r["laps_completed"] for r in all_results]):.2f}'
    )
    print(
        f'평균 속도      : '
        f'{np.mean([r["avg_speed"] for r in all_results]):.3f} m/s'
    )
    print(
        f'평균 라인 오차 : '
        f'{np.mean([r["avg_deviation"] for r in all_results]):.4f} m'
    )
    print(
        f'평균 라인 전환 : '
        f'{np.mean([r["line_switches"] for r in all_results]):.2f}'
    )
    print(
        f'평균 스텝 수   : '
        f'{np.mean([r["total_steps"] for r in all_results]):.2f}'
    )
    print(f'{"═" * 45}\n')

    env.close()


if __name__ == '__main__':
    main()
