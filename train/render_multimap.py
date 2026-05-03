"""
render_multimap.py

multimap_train.py로 학습한 SAC 모델을 여러 맵에서 렌더링하는 스크립트.

사용 예시:
    python train/render_multimap.py
    python train/render_multimap.py --model final
    python train/render_multimap.py --maps Austin Spielberg
    python train/render_multimap.py --sleep 0.01 --max-steps 5000
"""

import os
import sys
import time
import argparse

import gym
import f110_gym  # noqa: F401  # gym registry 등록용
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import (
    RACETRACKS_DIR,
    ENV_CONFIG,
    OBS_CONFIG,
    LINE_CONFIG,
    MODEL_CONFIG,
    MAP_LIST,
    SPEED_MIN,
    SPEED_MAX,
    MULTIMAP_PATH,
    MULTIMAP_FINAL_PATH,
)

from sac_model import get_obs_dim
from waypoint_loader import load_waypoints
from pure_pursuit import PurePursuitController

from train_node import (
    Trainer,
    ForwardProgressTracker,
    preprocess_obs,
    action_to_env,
    make_init_pose,
    is_valid_obs,
    TARGET_SPEED_MIN,
    MAX_LAPS,
    MAX_FORWARD_WP_JUMP,
)


def get_map_paths(map_name: str) -> dict:
    map_dir = os.path.join(RACETRACKS_DIR, map_name)
    map_path = os.path.join(map_dir, f'{map_name}_map')
    centerline_csv = os.path.join(map_dir, f'{map_name}_centerline.csv')

    return {
        'map_name': map_name,
        'map_dir': map_dir,
        'map_path': map_path,
        'map_ext': '.png',
        'centerline_csv': centerline_csv,
    }


def make_env_config(map_paths: dict) -> dict:
    cfg = dict(ENV_CONFIG)
    cfg['map'] = map_paths['map_path']
    cfg['map_ext'] = map_paths['map_ext']
    return cfg


def load_map_waypoints(map_paths: dict) -> dict:
    csv_path = map_paths['centerline_csv']

    common_args = {
        'num_lines': LINE_CONFIG['num_lines'],
        'line_spacing': LINE_CONFIG['line_spacing'],
        'width_fraction': LINE_CONFIG.get('line_width_fraction', 0.60),
    }

    if os.path.exists(csv_path):
        print(f'centerline CSV 로드: {csv_path}')
        return load_waypoints(
            centerline_path=csv_path,
            **common_args,
        )

    print(f'CSV 없음 → 맵 이미지에서 centerline 추출: {map_paths["map_name"]}')
    return load_waypoints(
        map_path=map_paths['map_path'],
        map_ext=map_paths['map_ext'],
        **common_args,
    )


def validate_map(map_name: str) -> bool:
    paths = get_map_paths(map_name)

    has_csv = os.path.exists(paths['centerline_csv'])
    has_img = os.path.exists(paths['map_path'] + '.png')
    has_yaml = os.path.exists(paths['map_path'] + '.yaml')

    return has_csv or (has_img and has_yaml)


def load_trainer_checkpoint(trainer: Trainer, model_path: str):
    """
    checkpoint를 직접 읽어서 trainer.model에 적용한다.
    Trainer는 생성 시점의 MODEL_CONFIG와 obs_dim으로 model을 만든다.
    따라서 config.py의 hidden_dims, num_lines, use_line_curvature 설정은
    학습 당시 checkpoint와 동일해야 한다.
    """
    checkpoint = torch.load(
        model_path,
        map_location=trainer.device,
    )

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        trainer.model.load_state_dict(checkpoint['model_state'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        trainer.model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        trainer.model.load_state_dict(checkpoint)
    else:
        raise RuntimeError(
            f'지원하지 않는 모델 파일 형식입니다: {type(checkpoint)}'
        )

    trainer.model.to(trainer.device)
    trainer.model.eval()


def render_one_map(
    map_name: str,
    trainer: Trainer,
    controller: PurePursuitController,
    max_steps: int,
    sleep_sec: float,
    render_every: int,
    print_every: int,
):
    print(f'\n{"═" * 80}')
    print(f'[RENDER] map: {map_name}')
    print(f'{"═" * 80}')

    if not validate_map(map_name):
        print(f'[SKIP] {map_name}: 맵 파일 없음')
        return

    paths = get_map_paths(map_name)
    wp = load_map_waypoints(paths)

    waypoints_lines = wp['lines']
    num_lines = MODEL_CONFIG['num_lines']

    if len(waypoints_lines) != num_lines:
        print(
            f'[WARN] waypoint lines={len(waypoints_lines)}, '
            f'model num_lines={num_lines}. 설정 확인 필요.'
        )

    reference_line = waypoints_lines[num_lines // 2]
    n_waypoints = len(reference_line)
    init_poses = make_init_pose(waypoints_lines)

    env_cfg = make_env_config(paths)
    env = gym.make('f110_gym:f110-v0', **env_cfg)

    try:
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        progress_tracker = ForwardProgressTracker(
            reference_line,
            max_laps=MAX_LAPS,
            max_forward_jump=MAX_FORWARD_WP_JUMP,
        )
        progress_tracker.reset_from_obs(obs_raw)

        progress_score = 1
        progress_pct = (
            progress_score / (n_waypoints * MAX_LAPS) * 100.0
            if n_waypoints > 0
            else 0.0
        )

        for step in range(max_steps):
            if not is_valid_obs(obs):
                print('[WARN] invalid obs before select_action. terminate render.')
                break

            action = trainer.model.select_action(obs, training=False)

            env_action = action_to_env(
                action,
                obs_raw,
                trainer.model,
                waypoints_lines,
                controller,
            )

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))

            current_collision = bool(next_obs_raw['collisions'][0])

            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            speed = abs(float(next_obs_raw['linear_vels_x'][0]))
            line_idx = trainer.model.action_to_line_index(action)

            if step % render_every == 0:
                env.render()
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

            if step % print_every == 0:
                print(
                    f'step {step:5d} | '
                    f'wp: {progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%) | '
                    f'speed: {speed:.2f} | '
                    f'line: {line_idx} | '
                    f'action: {np.round(action, 3)} | '
                    f'env_action: {np.round(env_action, 3)}'
                )

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)

            if not is_valid_obs(next_obs):
                print('[WARN] invalid next_obs after env.step. terminate render.')
                break

            obs = next_obs
            obs_raw = next_obs_raw

            if current_collision:
                print(
                    f'[CRASH] step={step} | '
                    f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%) | '
                    f'speed={speed:.2f} | '
                    f'line={line_idx} | '
                    f'action={np.round(action, 3)} | '
                    f'env_action={np.round(env_action, 3)}'
                )
                break

            if done:
                print(f'[DONE] env done at step={step}')
                break

            if forward_done:
                print(
                    f'[FINISH] reached max progress | '
                    f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%)'
                )
                break

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default='best',
        choices=['best', 'final'],
        help='best=MULTIMAP_PATH, final=MULTIMAP_FINAL_PATH',
    )
    parser.add_argument(
        '--maps',
        nargs='*',
        default=None,
        help='렌더링할 맵 이름 목록. 생략하면 MAP_LIST 전체 사용.',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=8000,
        help='맵당 최대 step 수',
    )
    parser.add_argument(
        '--sleep',
        type=float,
        default=0.01,
        help='render 후 sleep 시간. 너무 느리면 0 또는 0.001로 조정.',
    )
    parser.add_argument(
        '--render-every',
        type=int,
        default=1,
        help='몇 step마다 env.render() 호출할지',
    )
    parser.add_argument(
        '--print-every',
        type=int,
        default=200,
        help='몇 step마다 상태 로그를 출력할지',
    )

    args = parser.parse_args()

    model_path = MULTIMAP_PATH if args.model == 'best' else MULTIMAP_FINAL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'모델 파일 없음: {model_path}')

    maps = args.maps if args.maps else MAP_LIST

    obs_dim = get_obs_dim(
        OBS_CONFIG['lidar_size'],
        MODEL_CONFIG['num_lines'],
        use_line_curvature=OBS_CONFIG.get('use_line_curvature', False),
    )

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=TARGET_SPEED_MIN,
    )

    trainer = Trainer(obs_dim)
    load_trainer_checkpoint(trainer, model_path)

    print(f'\n모델 로드 완료: {model_path}')
    print(f'렌더링 맵: {maps}')
    print(f'SPEED_MIN={SPEED_MIN}, TARGET_SPEED_MIN={TARGET_SPEED_MIN}, SPEED_MAX={SPEED_MAX}')
    print(f'num_lines={MODEL_CONFIG["num_lines"]}')
    print(f'obs_dim={obs_dim}')
    print(f'use_line_curvature={OBS_CONFIG.get("use_line_curvature", False)}')
    print(f'curvature_use_pp_window={OBS_CONFIG.get("curvature_use_pp_window", True)}')
    print(f'line_width_fraction={LINE_CONFIG.get("line_width_fraction", None)}')

    for map_name in maps:
        render_one_map(
            map_name=map_name,
            trainer=trainer,
            controller=controller,
            max_steps=args.max_steps,
            sleep_sec=args.sleep,
            render_every=max(1, args.render_every),
            print_every=max(1, args.print_every),
        )

    print('\n렌더링 종료')


if __name__ == '__main__':
    main()
