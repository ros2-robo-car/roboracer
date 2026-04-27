import os
import sys
import time
import numpy as np
import torch
import torch.quantization as quant
import gym
import f110_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sac_model import SAC
from train_node import (
    ENV_CONFIG, OBS_CONFIG, MODEL_CONFIG,
    preprocess_obs, get_obs_dim, build_model
)
from eval_node import EvalMetrics

MODEL_PATH     = os.path.join(os.path.dirname(__file__), '../models/sac_model.pth')
QUANTIZED_PATH = os.path.join(os.path.dirname(__file__), '../models/sac_model_quantized.pth')
N_EPISODES     = 5
MAX_STEPS      = 1000


def load_fp32_model(path):
    """FP32 원본 모델 로드"""
    checkpoint = torch.load(path, map_location='cpu')
    obs_dim = get_obs_dim()
    model = build_model(obs_dim)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def quantize_dynamic(model):
    """동적 양자화 (Linear 레이어 대상, FP32 → INT8)"""
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Linear 레이어만 양자화
        dtype=torch.qint8
    )
    return quantized


def evaluate_model(model, env, label="모델"):
    """모델 평가 + 추론 시간 측정"""
    init_poses = np.array([[0.0, 0.0, np.pi / 2]])
    all_results = []
    inference_times = []

    for episode in range(N_EPISODES):
        metrics = EvalMetrics()
        obs, _, _, _ = env.reset(poses=init_poses)
        processed_obs = preprocess_obs(obs)

        for _ in range(MAX_STEPS):
            # 추론 시간 측정
            start = time.perf_counter()
            action = model.select_action(processed_obs, training=False)
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

            next_obs, reward, done, _ = env.step(np.array([action]))
            metrics.update(next_obs, action, done)
            processed_obs = preprocess_obs(next_obs)

            if done:
                break

        result = metrics.summary(episode)
        all_results.append(result)

    avg_time = np.mean(inference_times) * 1000  # ms 변환

    print(f'\n{"═"*50}')
    print(f'[{label}] 전체 {N_EPISODES} 에피소드 평균')
    print(f'{"═"*50}')
    print(f'평균 충돌 횟수 : {np.mean([r["collisions"] for r in all_results]):.1f} 회')
    print(f'평균 속도      : {np.mean([r["avg_speed"] for r in all_results]):.3f} m/s')
    print(f'평균 바퀴 떨림 : {np.mean([r["wheel_tremor"] for r in all_results]):.4f} rad/step')
    print(f'평균 추론 시간 : {avg_time:.3f} ms')
    print(f'{"═"*50}\n')

    return all_results, avg_time


def get_model_size(model, path=None):
    """모델 파일 크기 (MB)"""
    if path and os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    # 임시 저장해서 크기 측정
    tmp = '/tmp/tmp_model.pth'
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return size


def main():
    print('\n양자화 비교 평가 시작')
    print(f'원본 모델: {MODEL_PATH}\n')

    # 1) 원본 FP32 모델 로드
    fp32_model = load_fp32_model(MODEL_PATH)
    fp32_size = get_model_size(fp32_model, MODEL_PATH)

    # 2) 동적 양자화 적용 (FP32 → INT8)
    int8_model = quantize_dynamic(fp32_model)
    torch.save(int8_model.state_dict(), QUANTIZED_PATH)
    int8_size = get_model_size(int8_model, QUANTIZED_PATH)

    print(f'모델 크기 비교:')
    print(f'  FP32 : {fp32_size:.2f} MB')
    print(f'  INT8 : {int8_size:.2f} MB')
    print(f'  압축률: {(1 - int8_size/fp32_size)*100:.1f}%\n')

    # 3) 환경 생성 및 비교 평가
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)

    print('─' * 50)
    print('FP32 원본 모델 평가')
    print('─' * 50)
    fp32_results, fp32_time = evaluate_model(fp32_model, env, "FP32")

    print('─' * 50)
    print('INT8 양자화 모델 평가')
    print('─' * 50)
    int8_results, int8_time = evaluate_model(int8_model, env, "INT8")

    # 4) 최종 비교 요약
    print(f'\n{"═"*50}')
    print(f'최종 비교 요약 (FP32 vs INT8)')
    print(f'{"═"*50}')
    print(f'{"항목":<15} {"FP32":>10} {"INT8":>10} {"변화":>10}')
    print(f'{"─"*50}')

    fp32_col = np.mean([r["collisions"] for r in fp32_results])
    int8_col = np.mean([r["collisions"] for r in int8_results])
    print(f'{"충돌 횟수":<15} {fp32_col:>10.1f} {int8_col:>10.1f} {int8_col-fp32_col:>+10.1f}')

    fp32_spd = np.mean([r["avg_speed"] for r in fp32_results])
    int8_spd = np.mean([r["avg_speed"] for r in int8_results])
    print(f'{"평균 속도":<15} {fp32_spd:>10.3f} {int8_spd:>10.3f} {int8_spd-fp32_spd:>+10.3f}')

    fp32_trm = np.mean([r["wheel_tremor"] for r in fp32_results])
    int8_trm = np.mean([r["wheel_tremor"] for r in int8_results])
    print(f'{"바퀴 떨림":<15} {fp32_trm:>10.4f} {int8_trm:>10.4f} {int8_trm-fp32_trm:>+10.4f}')

    print(f'{"추론 시간(ms)":<15} {fp32_time:>10.3f} {int8_time:>10.3f} {int8_time-fp32_time:>+10.3f}')
    print(f'{"모델 크기(MB)":<15} {fp32_size:>10.2f} {int8_size:>10.2f} {int8_size-fp32_size:>+10.2f}')
    print(f'{"═"*50}\n')

    env.close()


if __name__ == '__main__':
    main()