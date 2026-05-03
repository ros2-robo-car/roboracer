"""
quantize_model.py
──────────────────────────────────
FP32 SAC actor와 INT8 dynamic quantized actor 비교 평가 스크립트.

목적:
- 학습된 FP32 SAC 모델을 불러온다.
- actor만 INT8 dynamic quantization을 적용한다.
- 같은 env/action_to_env 로직으로 FP32 vs INT8 주행 성능을 비교한다.
- CPU 기준 actor 추론 시간, 모델 크기, 주행 지표, action 차이를 비교한다.

사용 예:
    python train/quantize_model.py

필요 config 값:
    MODEL_SAVE_PATH

config.py 설정:
    QUANTIZED_PATH
    QUANTIZE_CONFIG = {
        'episodes',
        'max_steps',
        'compare_samples',
        'torch_threads',
    }
"""

import copy
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import gym
import f110_gym  # noqa: F401  # gym registry 등록용
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config as cfg

from sac_model import SAC, get_obs_dim
from waypoint_loader import get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController
from train_node import (
    TARGET_SPEED_MIN,
    MAX_LAPS,
    MAX_FORWARD_WP_JUMP,
    ForwardProgressTracker,
    action_to_env,
    format_lap_times,
    load_racing_lines,
    make_init_pose,
    preprocess_obs,
)


# ── 설정값 ───────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = cfg.MULTIMAP_FINAL_PATH

QUANTIZED_PATH = getattr(
    cfg,
    'QUANTIZED_PATH',
    os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'sac_model_actor_int8_dynamic.pth'),
)

QUANTIZE_CONFIG = cfg.QUANTIZE_CONFIG

N_QUANTIZE_EPISODES = int(QUANTIZE_CONFIG.get('episodes', 5))

QUANTIZE_MAX_STEPS = int(QUANTIZE_CONFIG.get('max_steps', 20000))

ACTION_COMPARE_SAMPLES = int(QUANTIZE_CONFIG.get('compare_samples', 3000))

TORCH_THREADS = int(QUANTIZE_CONFIG.get('torch_threads', 1))


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def safe_mean(values: List[float], default: float = 0.0) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(default)
    return float(np.mean(arr))


def safe_percentile(values: List[float], q: float, default: float = 0.0) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(default)
    return float(np.percentile(arr, q))


def get_module_size_mb(module: torch.nn.Module) -> float:
    """
    module.state_dict()를 임시 저장해 크기를 측정한다.
    actor 크기 비교에 사용한다.
    """
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(module.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024.0 * 1024.0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return float(size_mb)


def get_file_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024.0 * 1024.0)


# ── 모델 로드 / 양자화 ────────────────────────────────────────────────────────
def build_model(model_config: dict = None) -> SAC:
    """
    config.py와 checkpoint의 model_config를 기준으로 SAC 모델을 생성한다.

    checkpoint에 obs_dim/use_line_curvature가 저장되어 있으면 그 값을 우선 사용한다.
    이렇게 해야 곡률 feature를 켠 모델과 끈 모델을 모두 안전하게 불러올 수 있다.
    """
    if model_config is None:
        model_config = cfg.MODEL_CONFIG

    num_lines = int(model_config.get('num_lines', cfg.MODEL_CONFIG['num_lines']))

    saved_obs_dim = model_config.get('obs_dim', None)
    if saved_obs_dim is not None:
        obs_dim = int(saved_obs_dim)
    else:
        use_line_curvature = bool(
            model_config.get(
                'use_line_curvature',
                cfg.OBS_CONFIG.get('use_line_curvature', False),
            )
        )
        obs_dim = get_obs_dim(
            cfg.OBS_CONFIG['lidar_size'],
            num_lines,
            use_line_curvature=use_line_curvature,
        )

    model = SAC(
        obs_dim,
        model_config.get('action_dim', cfg.MODEL_CONFIG['action_dim']),
        model_config.get('hidden_dims', cfg.MODEL_CONFIG['hidden_dims']),
        num_lines,
    )
    return model


def load_fp32_model(path: str) -> SAC:
    if not os.path.exists(path):
        raise FileNotFoundError(f'FP32 model checkpoint not found: {path}')

    checkpoint = torch.load(path, map_location='cpu')

    model_config = dict(cfg.MODEL_CONFIG)
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_config.update(checkpoint['model_config'])

    # checkpoint에 곡률 feature 사용 여부가 저장되어 있으면
    # train_node.preprocess_obs()도 같은 observation 구성을 쓰도록 맞춘다.
    if 'use_line_curvature' in model_config:
        cfg.OBS_CONFIG['use_line_curvature'] = bool(model_config['use_line_curvature'])

    model = build_model(model_config)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            '지원하지 않는 checkpoint 형식입니다. '
            'dict 또는 {"model_state": state_dict} 형식이어야 합니다.'
        )

    model.load_state_dict(state_dict)
    model.cpu()
    model.eval()
    return model


def quantize_actor_dynamic(fp32_model: SAC) -> SAC:
    """
    SAC 전체가 아니라 actor만 INT8 dynamic quantization 적용.

    평가/배포 시 action을 만드는 것은 actor이므로 critic은 양자화하지 않는다.
    dynamic quantization은 CPU 추론에서 효과가 크므로 CPU에서 비교한다.
    """
    int8_model = copy.deepcopy(fp32_model)
    int8_model.cpu()
    int8_model.eval()

    int8_model.actor = torch.quantization.quantize_dynamic(
        int8_model.actor,
        {nn.Linear},
        dtype=torch.qint8,
    )

    int8_model.eval()
    return int8_model


def save_quantized_model(int8_model: SAC, path: str):
    """
    dynamic quantized actor가 포함된 모델 전체를 저장한다.

    주의:
    - 이 파일은 일반 FP32 SAC state_dict처럼 바로 load_state_dict 하기 위한 용도라기보다,
      실험 결과 보존 및 크기 확인용에 가깝다.
    - 재사용 시에는 동일한 클래스 정의가 필요하다.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(int8_model, path)


# ── 평가 지표 ────────────────────────────────────────────────────────────────
def compute_deviation(
    obs_raw: dict,
    waypoints_lines: list,
    line_idx: int,
) -> float:
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    position = np.array([x, y], dtype=np.float32)

    line_idx = int(np.clip(line_idx, 0, len(waypoints_lines) - 1))
    waypoints = waypoints_lines[line_idx]
    nearest_idx = get_nearest_waypoint_idx(position, waypoints)

    return float(np.linalg.norm(waypoints[nearest_idx] - position))


def evaluate_model(
    model: SAC,
    label: str,
    env,
    waypoints_lines: list,
    controller: PurePursuitController,
    init_poses: np.ndarray,
    num_lines: int,
    collect_obs_samples: bool = False,
) -> Tuple[Dict[str, float], List[np.ndarray], List[float]]:
    """
    같은 action_to_env 로직으로 모델을 평가한다.

    Returns:
        summary: 주행 성능 요약
        obs_samples: action similarity 비교용 observation 샘플
        inference_times_ms: actor select_action 시간 리스트(ms)
    """
    reference_line = waypoints_lines[num_lines // 2]
    n_waypoints = len(reference_line)
    timestep = float(cfg.ENV_CONFIG.get('timestep', 0.01))

    inference_times_ms: List[float] = []
    obs_samples: List[np.ndarray] = []

    episode_rewards = []
    progress_pcts = []
    avg_speeds = []
    first_lap_times = []
    full_lap_times = []
    crash_flags = []
    timeout_flags = []
    finish_flags = []
    line_switch_counts = []
    deviations = []
    tremors = []
    step_counts = []

    # 첫 obs 기준으로 timing warmup. PyTorch 초기 오버헤드 제거용.
    obs_raw, _, _, _ = env.reset(poses=init_poses)
    obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)
    with torch.no_grad():
        for _ in range(50):
            _ = model.select_action(obs, training=False)

    for episode in range(N_QUANTIZE_EPISODES):
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        progress_tracker = ForwardProgressTracker(
            reference_line,
            max_laps=MAX_LAPS,
            max_forward_jump=MAX_FORWARD_WP_JUMP,
        )
        progress_tracker.reset_from_obs(obs_raw)

        speeds = []
        lap_times = []
        last_lap_step = 0
        next_lap_progress = n_waypoints
        progress_score = 1
        progress_pct = progress_score / (n_waypoints * MAX_LAPS) * 100.0

        episode_reward_proxy = 0.0
        collision = False
        timeout = False
        forward_done = False

        last_line_idx = None
        line_switches = 0
        prev_steering = None
        episode_tremors = []
        episode_deviations = []

        for step_in_ep in range(QUANTIZE_MAX_STEPS):
            if collect_obs_samples and len(obs_samples) < ACTION_COMPARE_SAMPLES:
                obs_samples.append(obs.copy())

            with torch.no_grad():
                start_ns = time.perf_counter_ns()
                action = model.select_action(obs, training=False)
                end_ns = time.perf_counter_ns()

            inference_times_ms.append((end_ns - start_ns) / 1_000_000.0)

            line_idx = int(model.action_to_line_index(action))

            if last_line_idx is not None and line_idx != last_line_idx:
                line_switches += 1
            last_line_idx = line_idx

            env_action = action_to_env(
                action,
                obs_raw,
                model,
                waypoints_lines,
                controller,
            )

            steering = float(env_action[0])
            if prev_steering is not None:
                episode_tremors.append(abs(steering - prev_steering))
            prev_steering = steering

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))

            collision = bool(next_obs_raw['collisions'][0])
            speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))
            if np.isfinite(speed_value):
                speeds.append(speed_value)

            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            # reward proxy: 실제 학습 reward 대신 공통 비교용 진행률 기반 점수.
            # 모델 간 상대 비교용으로만 사용한다.
            if not collision:
                episode_reward_proxy += float(progress_delta)
            else:
                episode_reward_proxy -= 100.0

            episode_deviations.append(
                compute_deviation(next_obs_raw, waypoints_lines, line_idx)
            )

            while (
                progress_score >= next_lap_progress
                and len(lap_times) < MAX_LAPS
            ):
                lap_steps = step_in_ep + 1 - last_lap_step
                lap_time = lap_steps * timestep
                lap_times.append(lap_time)

                last_lap_step = step_in_ep + 1
                next_lap_progress += n_waypoints

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            obs = next_obs
            obs_raw = next_obs_raw

            timeout = step_in_ep == QUANTIZE_MAX_STEPS - 1

            if done or forward_done or collision or timeout:
                step_counts.append(step_in_ep + 1)
                break

        episode_rewards.append(episode_reward_proxy)
        progress_pcts.append(progress_pct)
        avg_speeds.append(safe_mean(speeds))
        crash_flags.append(1.0 if collision else 0.0)
        timeout_flags.append(1.0 if timeout and not forward_done and not collision else 0.0)
        finish_flags.append(1.0 if forward_done else 0.0)
        line_switch_counts.append(float(line_switches))
        deviations.append(safe_mean(episode_deviations))
        tremors.append(safe_mean(episode_tremors))

        if len(lap_times) >= 1:
            first_lap_times.append(lap_times[0])
        else:
            first_lap_times.append(np.nan)

        if forward_done and len(lap_times) >= MAX_LAPS:
            full_lap_times.append(float(sum(lap_times)))
        else:
            full_lap_times.append(np.nan)

        print(
            f'[{label}] ep {episode:3d} | '
            f'progress={progress_pct:5.1f}% | '
            f'finish={int(forward_done)} | '
            f'crash={int(collision)} | '
            f'timeout={int(timeout and not forward_done and not collision)} | '
            f'speed={safe_mean(speeds):.2f} | '
            f'lap_time={format_lap_times(lap_times)} | '
            f'line_switch={line_switches}'
        )

    summary = {
        'reward_proxy': safe_mean(episode_rewards),
        'finish_rate': safe_mean(finish_flags),
        'crash_rate': safe_mean(crash_flags),
        'timeout_rate': safe_mean(timeout_flags),
        'progress_pct': safe_mean(progress_pcts),
        'avg_speed': safe_mean(avg_speeds),
        'first_lap_time': safe_mean(first_lap_times, default=np.nan),
        'full_lap_time': safe_mean(full_lap_times, default=np.nan),
        'line_switches': safe_mean(line_switch_counts),
        'avg_deviation': safe_mean(deviations),
        'wheel_tremor': safe_mean(tremors),
        'avg_steps': safe_mean(step_counts),
        'infer_mean_ms': safe_mean(inference_times_ms),
        'infer_p95_ms': safe_percentile(inference_times_ms, 95),
        'infer_p99_ms': safe_percentile(inference_times_ms, 99),
    }

    return summary, obs_samples, inference_times_ms


# ── action 차이 비교 ─────────────────────────────────────────────────────────
def compare_actions(
    fp32_model: SAC,
    int8_model: SAC,
    obs_samples: List[np.ndarray],
) -> Dict[str, float]:
    if not obs_samples:
        return {
            'line_match_rate': 0.0,
            'speed_action_mae': 0.0,
            'speed_action_rmse': 0.0,
            'num_samples': 0,
        }

    line_matches = []
    speed_errors = []

    with torch.no_grad():
        for obs in obs_samples:
            fp32_action = fp32_model.select_action(obs, training=False)
            int8_action = int8_model.select_action(obs, training=False)

            fp32_line = int(fp32_model.action_to_line_index(fp32_action))
            int8_line = int(int8_model.action_to_line_index(int8_action))

            line_matches.append(1.0 if fp32_line == int8_line else 0.0)
            speed_errors.append(float(fp32_action[1] - int8_action[1]))

    speed_errors_arr = np.array(speed_errors, dtype=np.float64)

    return {
        'line_match_rate': float(np.mean(line_matches)),
        'speed_action_mae': float(np.mean(np.abs(speed_errors_arr))),
        'speed_action_rmse': float(np.sqrt(np.mean(speed_errors_arr ** 2))),
        'num_samples': len(obs_samples),
    }


# ── 출력 ─────────────────────────────────────────────────────────────────────
def print_comparison_table(
    fp32_summary: Dict[str, float],
    int8_summary: Dict[str, float],
    fp32_actor_size: float,
    int8_actor_size: float,
    fp32_checkpoint_size: float,
    int8_saved_size: float,
    action_diff: Dict[str, float],
):
    print(f'\n{"═" * 78}')
    print('최종 비교: FP32 actor vs INT8 dynamic quantized actor')
    print(f'{"═" * 78}')
    print(f'{"항목":<24} {"FP32":>14} {"INT8":>14} {"변화(INT8-FP32)":>20}')
    print(f'{"─" * 78}')

    rows = [
        ('평균 reward proxy', 'reward_proxy', '.2f'),
        ('완주율', 'finish_rate', '.3f'),
        ('충돌률', 'crash_rate', '.3f'),
        ('timeout률', 'timeout_rate', '.3f'),
        ('평균 진행률(%)', 'progress_pct', '.2f'),
        ('평균 속도', 'avg_speed', '.3f'),
        ('첫 lap time(s)', 'first_lap_time', '.3f'),
        ('전체 lap time(s)', 'full_lap_time', '.3f'),
        ('라인 전환 횟수', 'line_switches', '.2f'),
        ('라인 이탈(m)', 'avg_deviation', '.4f'),
        ('바퀴 떨림', 'wheel_tremor', '.4f'),
        ('평균 step', 'avg_steps', '.1f'),
        ('추론 mean(ms)', 'infer_mean_ms', '.4f'),
        ('추론 p95(ms)', 'infer_p95_ms', '.4f'),
        ('추론 p99(ms)', 'infer_p99_ms', '.4f'),
    ]

    for label, key, fmt in rows:
        f = fp32_summary[key]
        i = int8_summary[key]
        diff = i - f
        print(f'{label:<24} {f:>14{fmt}} {i:>14{fmt}} {diff:>+20{fmt}}')

    actor_reduction = (1.0 - int8_actor_size / max(fp32_actor_size, 1e-12)) * 100.0
    saved_reduction = (1.0 - int8_saved_size / max(fp32_checkpoint_size, 1e-12)) * 100.0

    print(f'{"─" * 78}')
    print(
        f'{"actor 크기(MB)":<24} '
        f'{fp32_actor_size:>14.4f} '
        f'{int8_actor_size:>14.4f} '
        f'{(int8_actor_size - fp32_actor_size):>+20.4f}'
    )
    print(
        f'{"actor 압축률(%)":<24} '
        f'{0.0:>14.2f} '
        f'{actor_reduction:>14.2f} '
        f'{actor_reduction:>+20.2f}'
    )
    print(
        f'{"저장 파일 크기(MB)":<24} '
        f'{fp32_checkpoint_size:>14.4f} '
        f'{int8_saved_size:>14.4f} '
        f'{(int8_saved_size - fp32_checkpoint_size):>+20.4f}'
    )
    print(
        f'{"저장 파일 감소율(%)":<24} '
        f'{0.0:>14.2f} '
        f'{saved_reduction:>14.2f} '
        f'{saved_reduction:>+20.2f}'
    )

    print(f'{"─" * 78}')
    print(f'Action 비교 샘플 수: {action_diff["num_samples"]}')
    print(f'line 선택 일치율: {action_diff["line_match_rate"] * 100.0:.2f}%')
    print(f'speed action MAE: {action_diff["speed_action_mae"]:.6f}')
    print(f'speed action RMSE: {action_diff["speed_action_rmse"]:.6f}')
    print(f'{"═" * 78}\n')


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    torch.set_num_threads(max(TORCH_THREADS, 1))

    print('\n양자화 비교 평가 시작')
    print(f'  device: CPU')
    print(f'  torch threads: {torch.get_num_threads()}')
    print(f'  FP32 model: {MODEL_SAVE_PATH}')
    print(f'  INT8 save path: {QUANTIZED_PATH}')
    print(f'  episodes: {N_QUANTIZE_EPISODES}')
    print(f'  max steps: {QUANTIZE_MAX_STEPS}')
    print(f'  use_line_curvature: {cfg.OBS_CONFIG.get("use_line_curvature", False)}')
    print(f'  curvature_use_pp_window: {cfg.OBS_CONFIG.get("curvature_use_pp_window", True)}')

    fp32_model = load_fp32_model(MODEL_SAVE_PATH)
    num_lines = int(getattr(fp32_model, 'num_lines', cfg.MODEL_CONFIG.get('num_lines', cfg.LINE_CONFIG['num_lines'])))

    waypoints_lines = load_racing_lines()
    init_poses = make_init_pose(waypoints_lines)

    controller = PurePursuitController(
        max_speed=cfg.SPEED_MAX,
        min_speed=TARGET_SPEED_MIN,
    )

    int8_model = quantize_actor_dynamic(fp32_model)

    save_quantized_model(int8_model, QUANTIZED_PATH)

    fp32_actor_size = get_module_size_mb(fp32_model.actor)
    int8_actor_size = get_module_size_mb(int8_model.actor)
    fp32_checkpoint_size = get_file_size_mb(MODEL_SAVE_PATH)
    int8_saved_size = get_file_size_mb(QUANTIZED_PATH)

    print('\n모델 크기')
    print(f'  FP32 actor state_dict: {fp32_actor_size:.4f} MB')
    print(f'  INT8 actor state_dict: {int8_actor_size:.4f} MB')
    print(
        f'  actor 압축률: '
        f'{(1.0 - int8_actor_size / max(fp32_actor_size, 1e-12)) * 100.0:.2f}%'
    )
    print(f'  FP32 checkpoint file: {fp32_checkpoint_size:.4f} MB')
    print(f'  INT8 saved file: {int8_saved_size:.4f} MB')

    env = gym.make('f110_gym:f110-v0', **cfg.ENV_CONFIG)

    try:
        print('\n[1/2] FP32 평가')
        fp32_summary, obs_samples, _ = evaluate_model(
            fp32_model,
            label='FP32',
            env=env,
            waypoints_lines=waypoints_lines,
            controller=controller,
            init_poses=init_poses,
            num_lines=num_lines,
            collect_obs_samples=True,
        )

        print('\n[2/2] INT8 평가')
        int8_summary, _, _ = evaluate_model(
            int8_model,
            label='INT8',
            env=env,
            waypoints_lines=waypoints_lines,
            controller=controller,
            init_poses=init_poses,
            num_lines=num_lines,
            collect_obs_samples=False,
        )

    finally:
        env.close()

    action_diff = compare_actions(fp32_model, int8_model, obs_samples)

    print_comparison_table(
        fp32_summary=fp32_summary,
        int8_summary=int8_summary,
        fp32_actor_size=fp32_actor_size,
        int8_actor_size=int8_actor_size,
        fp32_checkpoint_size=fp32_checkpoint_size,
        int8_saved_size=int8_saved_size,
        action_diff=action_diff,
    )

    print('해석 기준')
    print('  - 완주율/충돌률/진행률이 유지되면 주행 성능 유지')
    print('  - 추론 mean, p95, p99가 감소하면 실시간 추론 효율 개선')
    print('  - actor 크기가 감소하면 임베디드 배포 부담 감소')
    print('  - line 일치율이 높고 speed MAE가 작으면 양자화 후 정책 변화가 작음')
    print('\n완료')


if __name__ == '__main__':
    main()
