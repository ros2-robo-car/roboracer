"""
quantize_model_multimap.py
──────────────────────────────────
23개 등 여러 F1TENTH 맵에서 FP32 SAC actor와 INT8 dynamic quantized actor를 비교한다.

비교 방식:
- 학습된 FP32 SAC 모델을 불러온다.
- actor만 INT8 dynamic quantization을 적용한다.
- 각 맵마다 FP32와 INT8을 같은 시작 위치/같은 횟수로 평가한다.
- 각 맵 종료 시 FP32 vs INT8 비교 지표를 출력한다.
- 전체 맵 평가 종료 후 최종 평균 지표를 출력한다.

권장 실행:
    cd /home/user/ros2_ws/src/roboracer
    python train/quantize_model_multimap.py

config.py에서 사용할 수 있는 선택 설정:
    QUANTIZED_PATH
    QUANTIZE_EPISODES_PER_MAP      # 맵마다 각 모델을 몇 번 평가할지
    QUANTIZE_EPISODES              # EPISODES_PER_MAP 없을 때 fallback
    QUANTIZE_START_RATIOS          # 예: [0.0, 0.33, 0.66]
    EVAL_MAX_STEPS
    QUANTIZE_COMPARE_SAMPLES
    QUANTIZE_TORCH_THREADS
    QUANTIZE_MAP_LIST              # 없으면 MAP_LIST 사용
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
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController
from train_node import (
    TARGET_SPEED_MIN,
    MAX_LAPS,
    MAX_FORWARD_WP_JUMP,
    ForwardProgressTracker,
    action_to_env,
    format_lap_times,
    preprocess_obs,
)


# ── 설정값 ───────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = cfg.MULTIMAP_PATH

QUANTIZED_PATH = getattr(
    cfg,
    'QUANTIZED_PATH',
    os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'sac_model_actor_int8_dynamic.pth'),
)

QUANTIZE_CONFIG = getattr(cfg, 'QUANTIZE_CONFIG', {})

EPISODES_PER_MAP = int(
    QUANTIZE_CONFIG.get(
        'episodes_per_map',
        QUANTIZE_CONFIG.get('episodes', 3),
    )
)

EVAL_MAX_STEPS = int(
    QUANTIZE_CONFIG.get(
        'max_steps',
        cfg.TRAIN_CONFIG.get('max_steps', 15000),
    )
)

ACTION_COMPARE_SAMPLES = int(
    QUANTIZE_CONFIG.get('compare_samples', 3000)
)

TORCH_THREADS = int(
    QUANTIZE_CONFIG.get('torch_threads', 1)
)

START_RATIOS = list(
    QUANTIZE_CONFIG.get('start_ratios', [0.0, 0.33, 0.66])
)

if not START_RATIOS:
    START_RATIOS = [0.0]

MAP_NAMES = list(
    QUANTIZE_CONFIG.get(
        'map_list',
        getattr(cfg, 'QUANTIZE_MAP_LIST', getattr(cfg, 'MAP_LIST', [])),
    )
)


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


def format_float(value: float, fmt: str) -> str:
    if value is None or not np.isfinite(value):
        return 'nan'
    return format(value, fmt)


def get_module_size_mb(module: torch.nn.Module) -> float:
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


# ── 맵 로드 ───────────────────────────────────────────────────────────────────
def get_map_paths(map_name: str) -> dict:
    map_dir = os.path.join(cfg.RACETRACKS_DIR, map_name)
    map_path = os.path.join(map_dir, f'{map_name}_map')
    centerline_csv = os.path.join(map_dir, f'{map_name}_centerline.csv')

    return {
        'map_name': map_name,
        'map_dir': map_dir,
        'map_path': map_path,
        'map_ext': '.png',
        'centerline_csv': centerline_csv,
    }


def validate_maps(map_names: List[str]) -> List[str]:
    valid = []

    for name in map_names:
        paths = get_map_paths(name)
        has_csv = os.path.exists(paths['centerline_csv'])
        has_img = os.path.exists(paths['map_path'] + '.png')
        has_yaml = os.path.exists(paths['map_path'] + '.yaml')

        if has_csv or (has_img and has_yaml):
            valid.append(name)
        else:
            print(f'  [스킵] {name}: 맵 파일 없음')

    return valid


def make_env_config(map_paths: dict) -> dict:
    env_cfg = dict(cfg.ENV_CONFIG)
    env_cfg['map'] = map_paths['map_path']
    env_cfg['map_ext'] = map_paths['map_ext']
    return env_cfg


def load_map_waypoints(map_paths: dict) -> dict:
    csv_path = map_paths['centerline_csv']

    if os.path.exists(csv_path):
        return load_waypoints(
            centerline_path=csv_path,
            num_lines=cfg.LINE_CONFIG['num_lines'],
            line_spacing=cfg.LINE_CONFIG['line_spacing'],
            width_fraction=cfg.LINE_CONFIG.get('line_width_fraction', 0.60),
        )

    return load_waypoints(
        map_path=map_paths['map_path'],
        map_ext=map_paths['map_ext'],
        num_lines=cfg.LINE_CONFIG['num_lines'],
        line_spacing=cfg.LINE_CONFIG['line_spacing'],
        width_fraction=cfg.LINE_CONFIG.get('line_width_fraction', 0.60),
    )


def make_init_pose_from_waypoint(
    waypoints_lines: list,
    start_ratio: float,
    start_line_idx: int = None,
    lookahead_idx: int = None,
) -> np.ndarray:
    num_lines = len(waypoints_lines)

    if start_line_idx is None:
        start_line_idx = cfg.LINE_CONFIG.get('start_line_idx', None)
        if start_line_idx is None:
            start_line_idx = num_lines // 2

    if lookahead_idx is None:
        lookahead_idx = cfg.LINE_CONFIG.get('start_lookahead_idx', 5)

    reference_line = waypoints_lines[start_line_idx]
    n = len(reference_line)

    ratio = float(start_ratio) % 1.0
    start_wp_idx = int(round(ratio * n)) % n

    p0 = reference_line[start_wp_idx]
    p1 = reference_line[(start_wp_idx + lookahead_idx) % n]

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    heading = float(np.arctan2(dy, dx))

    return np.array([[float(p0[0]), float(p0[1]), heading]], dtype=np.float32)


def load_map_data(map_name: str) -> dict:
    paths = get_map_paths(map_name)
    wp = load_map_waypoints(paths)

    waypoints_lines = wp['lines']
    num_lines = len(waypoints_lines)
    reference_line = waypoints_lines[num_lines // 2]
    n_waypoints = len(reference_line)

    return {
        'map_name': map_name,
        'paths': paths,
        'waypoints_lines': waypoints_lines,
        'reference_line': reference_line,
        'n_waypoints': n_waypoints,
        'num_lines': num_lines,
    }


# ── 모델 로드 / 양자화 ────────────────────────────────────────────────────────
def build_model(model_config: dict = None) -> SAC:
    """
    config.py와 checkpoint의 model_config를 기준으로 SAC 모델을 생성한다.

    use_line_curvature=True이면 observation 차원은
    기존 obs_dim + num_lines가 된다.
    """
    if model_config is None:
        model_config = cfg.MODEL_CONFIG

    num_lines = int(model_config.get('num_lines', cfg.MODEL_CONFIG['num_lines']))
    obs_dim = get_obs_dim(
        cfg.OBS_CONFIG['lidar_size'],
        num_lines,
        use_line_curvature=cfg.OBS_CONFIG.get('use_line_curvature', False),
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(int8_model, path)


# ── 평가 지표 ────────────────────────────────────────────────────────────────
def compute_deviation(obs_raw: dict, waypoints_lines: list, line_idx: int) -> float:
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    position = np.array([x, y], dtype=np.float32)

    line_idx = int(np.clip(line_idx, 0, len(waypoints_lines) - 1))
    waypoints = waypoints_lines[line_idx]
    nearest_idx = get_nearest_waypoint_idx(position, waypoints)

    return float(np.linalg.norm(waypoints[nearest_idx] - position))


def evaluate_one_episode(
    model: SAC,
    label: str,
    map_data: dict,
    controller: PurePursuitController,
    start_ratio: float,
    collect_obs_samples: bool = False,
    obs_samples: List[np.ndarray] = None,
) -> Tuple[dict, List[float]]:
    if obs_samples is None:
        obs_samples = []

    map_name = map_data['map_name']
    env_cfg = make_env_config(map_data['paths'])
    env = gym.make('f110_gym:f110-v0', **env_cfg)

    waypoints_lines = map_data['waypoints_lines']
    reference_line = map_data['reference_line']
    n_waypoints = map_data['n_waypoints']
    num_lines = cfg.MODEL_CONFIG['num_lines']
    timestep = float(env_cfg.get('timestep', cfg.ENV_CONFIG.get('timestep', 0.01)))

    init_poses = make_init_pose_from_waypoint(waypoints_lines, start_ratio)

    inference_times_ms = []

    try:
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        # PyTorch 초기 오버헤드 제거용. 각 episode마다 아주 짧게 warmup.
        with torch.no_grad():
            for _ in range(10):
                _ = model.select_action(obs, training=False)

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

        reward_proxy = 0.0
        collision = False
        timeout = False
        forward_done = False
        done = False

        last_line_idx = None
        line_switches = 0
        prev_steering = None
        tremors = []
        deviations = []
        step_count = 0

        for step_in_ep in range(EVAL_MAX_STEPS):
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
                tremors.append(abs(steering - prev_steering))
            prev_steering = steering

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))
            step_count = step_in_ep + 1

            collision = bool(next_obs_raw['collisions'][0])
            speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))
            if np.isfinite(speed_value):
                speeds.append(speed_value)

            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            if not collision:
                reward_proxy += float(progress_delta)
            else:
                reward_proxy -= 100.0

            deviations.append(compute_deviation(next_obs_raw, waypoints_lines, line_idx))

            while progress_score >= next_lap_progress and len(lap_times) < MAX_LAPS:
                lap_steps = step_in_ep + 1 - last_lap_step
                lap_times.append(lap_steps * timestep)
                last_lap_step = step_in_ep + 1
                next_lap_progress += n_waypoints

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            obs = next_obs
            obs_raw = next_obs_raw

            timeout = step_in_ep == EVAL_MAX_STEPS - 1
            if done or forward_done or collision or timeout:
                break

    finally:
        env.close()

    result = {
        'model': label,
        'map': map_name,
        'start_ratio': float(start_ratio),
        'reward_proxy': float(reward_proxy),
        'finish': 1.0 if forward_done else 0.0,
        'crash': 1.0 if collision else 0.0,
        'timeout': 1.0 if timeout and not forward_done and not collision else 0.0,
        'progress_pct': float(progress_pct),
        'avg_speed': safe_mean(speeds),
        'first_lap_time': float(lap_times[0]) if len(lap_times) >= 1 else np.nan,
        'full_lap_time': float(sum(lap_times)) if forward_done and len(lap_times) >= MAX_LAPS else np.nan,
        'line_switches': float(line_switches),
        'avg_deviation': safe_mean(deviations),
        'wheel_tremor': safe_mean(tremors),
        'steps': float(step_count),
        'lap_time_str': format_lap_times(lap_times),
    }

    print(
        f'[{label:4s}] {map_name:16s} | '
        f'start={start_ratio:4.2f} | '
        f'progress={progress_pct:5.1f}% | '
        f'finish={int(forward_done)} | '
        f'crash={int(collision)} | '
        f'timeout={int(timeout and not forward_done and not collision)} | '
        f'speed={safe_mean(speeds):.2f} | '
        f'lap_time={format_lap_times(lap_times)} | '
        f'line_switch={line_switches}'
    )

    return result, inference_times_ms


def summarize_episode_rows(rows: List[dict], inference_times_ms: List[float]) -> Dict[str, float]:
    return {
        'reward_proxy': safe_mean([r['reward_proxy'] for r in rows]),
        'finish_rate': safe_mean([r['finish'] for r in rows]),
        'crash_rate': safe_mean([r['crash'] for r in rows]),
        'timeout_rate': safe_mean([r['timeout'] for r in rows]),
        'progress_pct': safe_mean([r['progress_pct'] for r in rows]),
        'avg_speed': safe_mean([r['avg_speed'] for r in rows]),
        'first_lap_time': safe_mean([r['first_lap_time'] for r in rows], default=np.nan),
        'full_lap_time': safe_mean([r['full_lap_time'] for r in rows], default=np.nan),
        'line_switches': safe_mean([r['line_switches'] for r in rows]),
        'avg_deviation': safe_mean([r['avg_deviation'] for r in rows]),
        'wheel_tremor': safe_mean([r['wheel_tremor'] for r in rows]),
        'avg_steps': safe_mean([r['steps'] for r in rows]),
        'infer_mean_ms': safe_mean(inference_times_ms),
        'infer_p95_ms': safe_percentile(inference_times_ms, 95),
        'infer_p99_ms': safe_percentile(inference_times_ms, 99),
        'episodes': float(len(rows)),
    }


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
def print_model_comparison(
    title: str,
    fp32_summary: Dict[str, float],
    int8_summary: Dict[str, float],
):
    print(f'\n{"═" * 90}')
    print(title)
    print(f'{"═" * 90}')
    print(f'{"항목":<24} {"FP32":>14} {"INT8":>14} {"변화(INT8-FP32)":>20}')
    print(f'{"─" * 90}')

    rows = [
        ('평가 episode 수', 'episodes', '.0f'),
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
        f = fp32_summary.get(key, np.nan)
        i = int8_summary.get(key, np.nan)
        diff = i - f if np.isfinite(f) and np.isfinite(i) else np.nan
        print(
            f'{label:<24} '
            f'{format_float(f, fmt):>14} '
            f'{format_float(i, fmt):>14} '
            f'{format_float(diff, "+" + fmt):>20}'
        )

    print(f'{"═" * 90}\n')


def print_size_and_action_summary(
    fp32_actor_size: float,
    int8_actor_size: float,
    fp32_checkpoint_size: float,
    int8_saved_size: float,
    action_diff: Dict[str, float],
):
    actor_reduction = (1.0 - int8_actor_size / max(fp32_actor_size, 1e-12)) * 100.0
    saved_reduction = (1.0 - int8_saved_size / max(fp32_checkpoint_size, 1e-12)) * 100.0

    print(f'\n{"═" * 90}')
    print('모델 크기 및 action 일치도')
    print(f'{"═" * 90}')
    print(f'FP32 actor state_dict : {fp32_actor_size:.4f} MB')
    print(f'INT8 actor state_dict : {int8_actor_size:.4f} MB')
    print(f'actor 압축률          : {actor_reduction:.2f}%')
    print(f'FP32 checkpoint file  : {fp32_checkpoint_size:.4f} MB')
    print(f'INT8 saved file       : {int8_saved_size:.4f} MB')
    print(f'저장 파일 감소율      : {saved_reduction:.2f}%')
    print(f'{"─" * 90}')
    print(f'Action 비교 샘플 수   : {action_diff["num_samples"]}')
    print(f'line 선택 일치율      : {action_diff["line_match_rate"] * 100.0:.2f}%')
    print(f'speed action MAE      : {action_diff["speed_action_mae"]:.6f}')
    print(f'speed action RMSE     : {action_diff["speed_action_rmse"]:.6f}')
    print(f'{"═" * 90}\n')


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    torch.set_num_threads(max(TORCH_THREADS, 1))

    if not MAP_NAMES:
        raise RuntimeError('MAP_LIST 또는 QUANTIZE_MAP_LIST가 비어 있습니다.')

    valid_maps = validate_maps(MAP_NAMES)
    if not valid_maps:
        raise RuntimeError('평가 가능한 맵이 없습니다.')

    print('\n멀티맵 양자화 비교 평가 시작')
    print(f'  device: CPU')
    print(f'  torch threads: {torch.get_num_threads()}')
    print(f'  FP32 model: {MODEL_SAVE_PATH}')
    print(f'  INT8 save path: {QUANTIZED_PATH}')
    print(f'  maps: {len(valid_maps)}개')
    print(f'  episodes per map per model: {EPISODES_PER_MAP}')
    print(f'  start ratios: {START_RATIOS}')
    print(f'  max steps: {EVAL_MAX_STEPS}')
    print(f'  action compare samples max: {ACTION_COMPARE_SAMPLES}')
    print(f'  use_line_curvature: {cfg.OBS_CONFIG.get("use_line_curvature", False)}')
    print(f'  curvature_use_pp_window: {cfg.OBS_CONFIG.get("curvature_use_pp_window", True)}')

    controller = PurePursuitController(
        max_speed=cfg.SPEED_MAX,
        min_speed=TARGET_SPEED_MIN,
    )

    fp32_model = load_fp32_model(MODEL_SAVE_PATH)
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

    all_fp32_rows: List[dict] = []
    all_int8_rows: List[dict] = []
    all_fp32_times: List[float] = []
    all_int8_times: List[float] = []
    obs_samples: List[np.ndarray] = []

    for map_idx, map_name in enumerate(valid_maps, start=1):
        print(f'\n{"#" * 90}')
        print(f'맵 {map_idx}/{len(valid_maps)}: {map_name}')
        print(f'{"#" * 90}')

        map_data = load_map_data(map_name)

        map_fp32_rows: List[dict] = []
        map_int8_rows: List[dict] = []
        map_fp32_times: List[float] = []
        map_int8_times: List[float] = []

        # FP32 먼저 평가
        print(f'\n[FP32 평가] {map_name}')
        for ep in range(EPISODES_PER_MAP):
            start_ratio = START_RATIOS[ep % len(START_RATIOS)]
            row, times = evaluate_one_episode(
                fp32_model,
                label='FP32',
                map_data=map_data,
                controller=controller,
                start_ratio=start_ratio,
                collect_obs_samples=True,
                obs_samples=obs_samples,
            )
            map_fp32_rows.append(row)
            map_fp32_times.extend(times)

        # INT8 평가
        print(f'\n[INT8 평가] {map_name}')
        for ep in range(EPISODES_PER_MAP):
            start_ratio = START_RATIOS[ep % len(START_RATIOS)]
            row, times = evaluate_one_episode(
                int8_model,
                label='INT8',
                map_data=map_data,
                controller=controller,
                start_ratio=start_ratio,
                collect_obs_samples=False,
                obs_samples=None,
            )
            map_int8_rows.append(row)
            map_int8_times.extend(times)

        map_fp32_summary = summarize_episode_rows(map_fp32_rows, map_fp32_times)
        map_int8_summary = summarize_episode_rows(map_int8_rows, map_int8_times)

        print_model_comparison(
            title=f'맵별 비교 결과: {map_name}',
            fp32_summary=map_fp32_summary,
            int8_summary=map_int8_summary,
        )

        all_fp32_rows.extend(map_fp32_rows)
        all_int8_rows.extend(map_int8_rows)
        all_fp32_times.extend(map_fp32_times)
        all_int8_times.extend(map_int8_times)

    final_fp32_summary = summarize_episode_rows(all_fp32_rows, all_fp32_times)
    final_int8_summary = summarize_episode_rows(all_int8_rows, all_int8_times)

    action_diff = compare_actions(fp32_model, int8_model, obs_samples)

    print_model_comparison(
        title='최종 전체 맵 평균 비교: FP32 actor vs INT8 dynamic quantized actor',
        fp32_summary=final_fp32_summary,
        int8_summary=final_int8_summary,
    )

    print_size_and_action_summary(
        fp32_actor_size=fp32_actor_size,
        int8_actor_size=int8_actor_size,
        fp32_checkpoint_size=fp32_checkpoint_size,
        int8_saved_size=int8_saved_size,
        action_diff=action_diff,
    )

    print('해석 기준')
    print('  - 완주율/충돌률/진행률이 유지되면 주행 성능 유지')
    print('  - 추론 mean, p95, p99가 감소하면 CPU 실시간 추론 효율 개선')
    print('  - actor 크기가 감소하면 임베디드 배포 부담 감소')
    print('  - line 일치율이 높고 speed MAE가 작으면 양자화 후 정책 변화가 작음')
    print('\n완료')


if __name__ == '__main__':
    main()
