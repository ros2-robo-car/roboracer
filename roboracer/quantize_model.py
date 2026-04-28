"""
quantize_model.py (라인 선택 버전)
──────────────────────────────────
FP32 → INT8 동적 양자화 및 비교 평가
설정은 config.py에서 관리
"""

import os
import sys
import time
import numpy as np
import torch
import gym
import f110_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    ENV_CONFIG, OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH, QUANTIZED_PATH,
    QUANTIZE_EPISODES, EVAL_MAX_STEPS,
)
from sac_model import SAC, get_obs_dim
from pure_pursuit import PurePursuitController
from eval_node import (
    EvalMetrics, preprocess_obs, load_racing_lines, load_model,
)


def quantize_dynamic(model):
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)


def evaluate_model(model, env, waypoints_lines, controller, num_lines, label):
    init_poses = np.array([[0.0, 0.0, np.pi / 2]])
    all_results, inference_times = [], []

    for episode in range(QUANTIZE_EPISODES):
        metrics = EvalMetrics()
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        for _ in range(EVAL_MAX_STEPS):
            start = time.perf_counter()
            action = model.select_action(obs, training=False)
            inference_times.append(time.perf_counter() - start)

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
            if done:
                break

        all_results.append(metrics.summary(episode))

    avg_time = np.mean(inference_times) * 1000
    print(f'\n[{label}] 평균 추론 시간: {avg_time:.3f} ms')
    return all_results, avg_time


def get_model_size(model, path=None):
    if path and os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    tmp = '/tmp/tmp_model.pth'
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return size


def main():
    print('\n양자화 비교 평가 시작\n')

    waypoints_lines = load_racing_lines()
    num_lines = MODEL_CONFIG.get('num_lines', LINE_CONFIG['num_lines'])
    controller = PurePursuitController(max_speed=SPEED_MAX, min_speed=SPEED_MIN)

    fp32_model = load_model(MODEL_SAVE_PATH)
    fp32_size = get_model_size(fp32_model, MODEL_SAVE_PATH)

    int8_model = quantize_dynamic(fp32_model)
    torch.save(int8_model.state_dict(), QUANTIZED_PATH)
    int8_size = get_model_size(int8_model, QUANTIZED_PATH)

    print(f'모델 크기: FP32={fp32_size:.2f}MB | INT8={int8_size:.2f}MB | '
          f'압축률={( 1 - int8_size / fp32_size) * 100:.1f}%\n')

    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

    fp32_results, fp32_time = evaluate_model(fp32_model, env, waypoints_lines, controller, num_lines, "FP32")
    int8_results, int8_time = evaluate_model(int8_model, env, waypoints_lines, controller, num_lines, "INT8")

    print(f'\n{"═"*60}')
    print(f'최종 비교 (FP32 vs INT8)')
    print(f'{"═"*60}')
    print(f'{"항목":<18} {"FP32":>10} {"INT8":>10} {"변화":>10}')
    print(f'{"─"*60}')

    for label, key, fmt in [
        ('충돌 횟수', 'collisions', '.1f'),
        ('평균 속도', 'avg_speed', '.3f'),
        ('바퀴 떨림', 'wheel_tremor', '.4f'),
        ('라인 이탈(m)', 'avg_deviation', '.4f'),
        ('라인 전환', 'line_switches', '.1f'),
    ]:
        f = np.mean([r[key] for r in fp32_results])
        i = np.mean([r[key] for r in int8_results])
        print(f'{label:<18} {f:>10{fmt}} {i:>10{fmt}} {i-f:>+10{fmt}}')

    print(f'{"추론 시간(ms)":<18} {fp32_time:>10.3f} {int8_time:>10.3f} {int8_time-fp32_time:>+10.3f}')
    print(f'{"모델 크기(MB)":<18} {fp32_size:>10.2f} {int8_size:>10.2f} {int8_size-fp32_size:>+10.2f}')
    print(f'{"═"*60}\n')

    env.close()

if __name__ == '__main__':
    main()