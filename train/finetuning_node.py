"""
finetune_line_stable.py

기존 학습된 SAC 모델을 로드한 뒤,
직진 구간에서의 불필요한 라인 전환에 패널티를 주어 fine-tuning한다.

핵심 아이디어:
  - 현재 waypoint 부근의 곡률이 낮으면(직진) 라인 전환 시 패널티 부여
  - 곡률이 높으면(커브) 라인 전환은 허용 (패널티 없음 또는 감소)
  - 기존 checkpoint reward, speed reward, steer penalty 등은 그대로 유지

사용법:
    python train/finetune_line_stable.py
    python train/finetune_line_stable.py --model models/sac_model.pth
    python train/finetune_line_stable.py --episodes 500 --lr 1e-5
"""

import os
import sys
import argparse

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
    TRAIN_CONFIG,
    REWARD_CONFIG,
    PURE_PURSUIT_CONFIG,
    SPEED_MIN,
    SPEED_MAX,
    MODEL_SAVE_PATH,
)

from sac_model import SAC, get_obs_dim, encode_action
from waypoint_loader import load_waypoints, get_nearest_waypoint_idx
from pure_pursuit import PurePursuitController

from train_node import (
    CheckpointTracker,
    WarmupCheckpointBaseline,
    Trainer,
    ForwardProgressTracker,
    ReplayBuffer,
    preprocess_obs,
    action_to_env,
    compute_reward,
    make_init_pose,
    is_valid_obs,
    is_valid_transition,
    get_collision_penalty,
    compute_steer_change_penalty,
    compute_three_point_curvature,
    build_checkpoint_indices,
    apply_brake,
    format_lap_times,
    NUM_CHECKPOINTS,
    BASELINE_STEPS,
    WARMUP_BASELINE_MIN_SAMPLES,
    TARGET_SPEED_MIN,
    MAX_LAPS,
    MAX_FORWARD_WP_JUMP,
    WAYPOINT_PROGRESS_REWARD,
    COLLISION_CURRICULUM_EPISODES,
    INVALID_OBS_PENALTY_SCALE,
    SPEED_ACTION_NOISE_STD,
    NO_PROGRESS_CHECK_INTERVAL,
    NO_PROGRESS_MIN_DELTA,
    NO_PROGRESS_PENALTY,
    NO_PROGRESS_PATIENCE,
    NO_PROGRESS_TERMINAL_PENALTY,
    TIMEOUT_FIXED_PENALTY,
    TIMEOUT_PENALTY_SCALE,
    SAC_SPEED_SCALE_RANGE,
    STEER_SPEED_THRESHOLD,
    STEER_DEADZONE,
    STEER_PENALTY,
    REWARD_CLAMP_MIN,
    REWARD_CLAMP_MAX,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fine-tuning 전용 설정
# ══════════════════════════════════════════════════════════════════════════════
FINETUNE_CONFIG = {
    # 학습 설정 (기존보다 보수적)
    'max_episodes': 1000,
    'max_steps': 10000,
    'lr_actor': 3e-5,       # 기존 3e-4의 1/10
    'lr_critic': 3e-5,
    'lr_alpha': 1e-4,       # 기존 1e-3의 1/10
    'eval_interval': 5,
    'warmup_steps': 0,      # 이미 학습된 모델이므로 warmup 불필요
    'buffer_size': 200000,  # 기존보다 작게 (fine-tuning이니까)
    'batch_size': 256,

    # ── 라인 entropy 제어 ──
    'line_entropy_scale': 0.0,

    # ── 직진 라인 전환 패널티 ──
    # 곡률이 이 값 이하면 "직진"으로 판정
    'straight_curvature_threshold': 0.3,

    # 직진에서 라인 전환 시 기본 패널티
    'line_switch_penalty': 20.0,

    # 곡률에 따른 패널티 감소 (곡률이 threshold에 가까울수록 패널티가 줄어듦)
    # 실제 패널티 = line_switch_penalty * (1 - curvature / threshold)
    'use_graduated_penalty': True,

    # 곡률 계산에 사용할 전방 waypoint 수
    'curvature_lookahead': 3,
    'curvature_sample_step': 1,

    # 저장 경로
    'save_path': os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'models', 'sac_model_finetuned.pth',
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 직진 구간 곡률 계산
# ══════════════════════════════════════════════════════════════════════════════
def get_local_curvature(
    obs_raw: dict,
    waypoints: np.ndarray,
    lookahead: int = 10,
    sample_step: int = 2,
) -> float:
    """
    현재 위치 부근 waypoint의 전방 최대 곡률을 계산한다.

    Returns:
        float: 곡률 값 (0에 가까우면 직진, 클수록 커브)
    """
    x = float(obs_raw['poses_x'][0])
    y = float(obs_raw['poses_y'][0])
    position = np.array([x, y], dtype=np.float32)

    n = len(waypoints)
    nearest_idx = get_nearest_waypoint_idx(position, waypoints)

    curvatures = []
    for offset in range(0, lookahead, sample_step):
        i0 = (nearest_idx + offset) % n
        i1 = (nearest_idx + offset + sample_step) % n
        i2 = (nearest_idx + offset + 2 * sample_step) % n

        curv = compute_three_point_curvature(
            waypoints[i0],
            waypoints[i1],
            waypoints[i2],
        )
        curvatures.append(curv)

    return float(np.max(curvatures)) if curvatures else 0.0


def compute_line_switch_penalty(
    curvature: float,
    line_changed: bool,
    config: dict = FINETUNE_CONFIG,
) -> float:
    """
    직진 구간에서 라인을 변경하면 패널티를 부여한다.

    - 라인이 변경되지 않았으면 패널티 0
    - 곡률이 threshold 이상이면 (커브) 패널티 0
    - 곡률이 threshold 미만이면 (직진) 패널티 부여
      - use_graduated_penalty=True면 곡률에 비례해서 패널티 감소
    """
    if not line_changed:
        return 0.0

    threshold = config['straight_curvature_threshold']
    base_penalty = config['line_switch_penalty']

    # 커브 구간이면 패널티 없음
    if curvature >= threshold:
        return 0.0

    # 직진 구간: 패널티 부여
    if config['use_graduated_penalty']:
        # 곡률이 0에 가까울수록 패널티 최대, threshold에 가까울수록 패널티 감소
        ratio = 1.0 - (curvature / threshold)
        return base_penalty * ratio
    else:
        return base_penalty


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_pretrained_model(path: str) -> dict:
    """
    기존 checkpoint를 로드하고 model_config를 반환한다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'모델 파일 없음: {path}')

    checkpoint = torch.load(path, map_location='cpu')

    model_config = dict(MODEL_CONFIG)
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_config.update(checkpoint['model_config'])

    if 'use_line_curvature' in model_config:
        OBS_CONFIG['use_line_curvature'] = bool(model_config['use_line_curvature'])

    return checkpoint, model_config


def create_finetuning_trainer(checkpoint, model_config: dict, config: dict) -> Trainer:
    """
    기존 checkpoint를 로드한 Trainer를 만들되,
    learning rate를 fine-tuning용으로 낮춘다.
    """
    num_lines = int(model_config.get('num_lines', MODEL_CONFIG['num_lines']))

    saved_obs_dim = model_config.get('obs_dim', None)
    if saved_obs_dim is not None:
        obs_dim = int(saved_obs_dim)
    else:
        obs_dim = get_obs_dim(
            OBS_CONFIG['lidar_size'],
            num_lines,
            use_line_curvature=OBS_CONFIG.get('use_line_curvature', False),
        )

    # 기존 TRAIN_CONFIG를 fine-tuning용으로 임시 교체
    original_lr_actor = TRAIN_CONFIG['lr_actor']
    original_lr_critic = TRAIN_CONFIG['lr_critic']
    original_lr_alpha = TRAIN_CONFIG['lr_alpha']
    original_buffer_size = TRAIN_CONFIG['buffer_size']

    TRAIN_CONFIG['lr_actor'] = config['lr_actor']
    TRAIN_CONFIG['lr_critic'] = config['lr_critic']
    TRAIN_CONFIG['lr_alpha'] = config['lr_alpha']
    TRAIN_CONFIG['buffer_size'] = config['buffer_size']

    trainer = Trainer(obs_dim)

    # 모델 가중치 로드
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        trainer.model.load_state_dict(checkpoint['model_state'])

        if 'log_alpha_line' in checkpoint:
            trainer.log_alpha_line.data.copy_(
                checkpoint['log_alpha_line'].to(trainer.device)
            )
            trainer.alpha_line = trainer.log_alpha_line.exp()

        if 'log_alpha_speed' in checkpoint:
            trainer.log_alpha_speed.data.copy_(
                checkpoint['log_alpha_speed'].to(trainer.device)
            )
            trainer.alpha_speed = trainer.log_alpha_speed.exp()
    elif isinstance(checkpoint, dict):
        trainer.model.load_state_dict(checkpoint)

    trainer.model.to(trainer.device)

    # ── line entropy target 조정 ──
    line_entropy_scale = config.get('line_entropy_scale', 0.1)
    original_target = trainer.target_entropy_line
    trainer.target_entropy_line = float(
        -np.log(1.0 / num_lines) * line_entropy_scale
    )
    print(
        f'line entropy target: {original_target:.4f} → '
        f'{trainer.target_entropy_line:.4f} '
        f'(scale={line_entropy_scale})'
    )

    # TRAIN_CONFIG 복원
    TRAIN_CONFIG['lr_actor'] = original_lr_actor
    TRAIN_CONFIG['lr_critic'] = original_lr_critic
    TRAIN_CONFIG['lr_alpha'] = original_lr_alpha
    TRAIN_CONFIG['buffer_size'] = original_buffer_size

    print(f'모델 로드 완료 | obs_dim: {obs_dim} | num_lines: {num_lines}')
    print(
        f'fine-tuning lr: '
        f'actor={config["lr_actor"]}, '
        f'critic={config["lr_critic"]}, '
        f'alpha={config["lr_alpha"]}'
    )

    return trainer


# ══════════════════════════════════════════════════════════════════════════════
# 웨이포인트 / 환경 설정
# ══════════════════════════════════════════════════════════════════════════════
def load_racing_lines() -> dict:
    csv_path = LINE_CONFIG['centerline_csv']

    if os.path.exists(csv_path):
        print(f'centerline CSV 로드: {csv_path}')
        wp = load_waypoints(
            centerline_path=csv_path,
            num_lines=LINE_CONFIG['num_lines'],
            line_spacing=LINE_CONFIG['line_spacing'],
            width_fraction=LINE_CONFIG.get('line_width_fraction', 0.60),
        )
    else:
        print('CSV 없음 → 맵 이미지에서 centerline 추출')
        wp = load_waypoints(
            map_path=LINE_CONFIG['map_path'],
            map_ext=LINE_CONFIG['map_ext'],
            num_lines=LINE_CONFIG['num_lines'],
            line_spacing=LINE_CONFIG['line_spacing'],
            width_fraction=LINE_CONFIG.get('line_width_fraction', 0.60),
        )

    print(f'라인 {len(wp["lines"])}개 생성 완료 (점 수: {len(wp["lines"][0])})')
    return wp


# ══════════════════════════════════════════════════════════════════════════════
# 메인 학습 루프
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='SAC 라인 안정화 Fine-tuning')
    parser.add_argument(
        '--model', type=str, default=MODEL_SAVE_PATH,
        help='기존 학습된 모델 경로',
    )
    parser.add_argument(
        '--episodes', type=int, default=FINETUNE_CONFIG['max_episodes'],
        help='fine-tuning 에피소드 수',
    )
    parser.add_argument(
        '--lr', type=float, default=FINETUNE_CONFIG['lr_actor'],
        help='fine-tuning learning rate (actor/critic 공통)',
    )
    parser.add_argument(
        '--penalty', type=float, default=FINETUNE_CONFIG['line_switch_penalty'],
        help='직진 라인 전환 패널티 크기',
    )
    parser.add_argument(
        '--threshold', type=float,
        default=FINETUNE_CONFIG['straight_curvature_threshold'],
        help='직진 판정 곡률 임계값',
    )
    parser.add_argument(
        '--entropy-scale', type=float,
        default=FINETUNE_CONFIG['line_entropy_scale'],
        help='line entropy target 배율 (기존 0.5 → 이 값으로, 낮을수록 라인 고정)',
    )
    parser.add_argument(
        '--save', type=str, default=FINETUNE_CONFIG['save_path'],
        help='fine-tuned 모델 저장 경로',
    )
    args = parser.parse_args()

    # CLI 인자로 설정 덮어쓰기
    config = dict(FINETUNE_CONFIG)
    config['max_episodes'] = args.episodes
    config['lr_actor'] = args.lr
    config['lr_critic'] = args.lr
    config['line_switch_penalty'] = args.penalty
    config['straight_curvature_threshold'] = args.threshold
    config['line_entropy_scale'] = args.entropy_scale
    config['save_path'] = args.save

    print(f'\n{"═" * 60}')
    print('SAC 라인 안정화 Fine-tuning')
    print(f'{"═" * 60}')
    print(f'기존 모델       : {args.model}')
    print(f'에피소드 수     : {config["max_episodes"]}')
    print(f'learning rate   : {config["lr_actor"]}')
    print(f'직진 곡률 임계값: {config["straight_curvature_threshold"]}')
    print(f'라인전환 패널티  : {config["line_switch_penalty"]}')
    print(f'graduated 패널티: {config["use_graduated_penalty"]}')
    print(f'line entropy scale: {config["line_entropy_scale"]}')
    print(f'저장 경로       : {config["save_path"]}')
    print(f'{"═" * 60}\n')

    # ── 모델 로드 ──
    checkpoint, model_config = load_pretrained_model(args.model)
    trainer = create_finetuning_trainer(checkpoint, model_config, config)

    num_lines = MODEL_CONFIG['num_lines']

    # ── 환경 / waypoint 준비 ──
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)
    wp = load_racing_lines()
    waypoints_lines = wp['lines']
    progress_reference_line = waypoints_lines[num_lines // 2]
    n_waypoints = len(progress_reference_line)

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=TARGET_SPEED_MIN,
    )
    init_poses = make_init_pose(waypoints_lines)

    cp_indices = build_checkpoint_indices(n_waypoints)
    print(f'체크포인트 {NUM_CHECKPOINTS}개: {cp_indices}')
    print(f'총 waypoint 수: {n_waypoints}\n')

    warmup_baseline = WarmupCheckpointBaseline(
        num_checkpoints=NUM_CHECKPOINTS,
        fallback_steps=BASELINE_STEPS,
        min_samples=WARMUP_BASELINE_MIN_SAMPLES,
        multiplier=REWARD_CONFIG.get('warmup_baseline_multiplier', 1.0),
    )
    trainer.checkpoint_baselines = warmup_baseline

    best_reward = -float('inf')
    total_steps = 0
    total_line_switches = 0
    total_straight_switches = 0

    for episode in range(config['max_episodes']):
        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        if not is_valid_obs(obs):
            print(f'[WARN] ep {episode}: invalid initial obs. skip.')
            continue

        progress_tracker = ForwardProgressTracker(
            progress_reference_line,
            max_laps=MAX_LAPS,
            max_forward_jump=MAX_FORWARD_WP_JUMP,
        )
        progress_tracker.reset_from_obs(obs_raw)

        checkpoint_tracker = CheckpointTracker(n_waypoints)

        episode_reward = 0.0
        prev_steering = 0.0
        prev_line_idx = None
        progress_score = 1
        progress_pct = 0.0
        collisions = 0
        speeds = []
        lap_times = []
        last_lap_step = 0
        next_lap_progress = n_waypoints
        last_line_idx = -1

        ep_line_switches = 0
        ep_straight_switches = 0

        progress_window_sum = 0.0
        progress_window_steps = 0
        no_progress_bad_count = 0

        for step_in_ep in range(config['max_steps']):
            action = trainer.model.select_action(obs, training=True)

            # speed action에 약간의 noise 추가
            action[1] += float(np.random.normal(0, SPEED_ACTION_NOISE_STD))
            action[1] = float(np.clip(action[1], -1.0, 1.0))

            env_action = action_to_env(
                action, obs_raw, trainer.model,
                waypoints_lines, controller,
            )

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))
            current_collision = bool(next_obs_raw['collisions'][0])
            if current_collision:
                collisions += 1

            next_obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)

            if not is_valid_obs(next_obs):
                invalid_penalty = (
                    get_collision_penalty(COLLISION_CURRICULUM_EPISODES)
                    * INVALID_OBS_PENALTY_SCALE
                )
                episode_reward += float(invalid_penalty)
                print('[WARN] invalid next_obs. terminate.')
                break

            # ── progress 업데이트 ──
            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            while (
                progress_score >= next_lap_progress
                and len(lap_times) < MAX_LAPS
            ):
                lap_steps = step_in_ep - last_lap_step
                lap_time = lap_steps * ENV_CONFIG['timestep']
                lap_times.append(lap_time)
                last_lap_step = step_in_ep
                next_lap_progress += n_waypoints

            speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))
            if np.isfinite(speed_value):
                speeds.append(speed_value)

            # ── 기존 reward 계산 ──
            reward, line_idx, _, checkpoint_passed, segment_steps, checkpoint_idx = (
                compute_reward(
                    next_obs_raw,
                    action,
                    trainer.model,
                    waypoints_lines,
                    checkpoint_tracker,
                    episode=COLLISION_CURRICULUM_EPISODES,  # 이미 학습된 모델이므로 최대 패널티 사용
                    baseline_provider=warmup_baseline,
                    use_speed_reward=True,
                )
            )
            last_line_idx = line_idx

            # ── 기존 steer penalty ──
            current_steering = float(env_action[0])
            reward -= compute_steer_change_penalty(
                speed_value, current_steering, prev_steering,
            )
            prev_steering = current_steering

            # ═══════════════════════════════════════════════════════════════
            # ★ 핵심: 직진 구간 라인 전환 패널티 ★
            # ═══════════════════════════════════════════════════════════════
            line_changed = (
                prev_line_idx is not None
                and line_idx != prev_line_idx
            )

            if line_changed:
                ep_line_switches += 1
                total_line_switches += 1

                # 현재 따라가는 라인의 곡률 계산
                current_waypoints = waypoints_lines[line_idx]
                curvature = get_local_curvature(
                    next_obs_raw,
                    current_waypoints,
                    lookahead=config['curvature_lookahead'],
                    sample_step=config['curvature_sample_step'],
                )

                switch_penalty = compute_line_switch_penalty(
                    curvature, line_changed, config,
                )

                if switch_penalty > 0:
                    reward -= switch_penalty
                    ep_straight_switches += 1
                    total_straight_switches += 1

            prev_line_idx = line_idx

            # ── no-progress / waypoint reward ──
            no_progress_done = False
            if not current_collision:
                if progress_delta > 0.0:
                    reward += WAYPOINT_PROGRESS_REWARD * progress_delta

                progress_window_sum += progress_delta
                progress_window_steps += 1

                if progress_window_steps >= NO_PROGRESS_CHECK_INTERVAL:
                    if progress_window_sum < NO_PROGRESS_MIN_DELTA:
                        reward += NO_PROGRESS_PENALTY
                        no_progress_bad_count += 1
                    else:
                        no_progress_bad_count = 0

                    progress_window_sum = 0.0
                    progress_window_steps = 0

                if no_progress_bad_count >= NO_PROGRESS_PATIENCE:
                    reward += NO_PROGRESS_TERMINAL_PENALTY
                    no_progress_done = True

            # ── timeout ──
            timeout_done = step_in_ep == config['max_steps'] - 1
            if (
                timeout_done
                and not forward_done
                and not current_collision
                and not no_progress_done
            ):
                timeout_penalty = (
                    TIMEOUT_FIXED_PENALTY
                    + TIMEOUT_PENALTY_SCALE * (1.0 - progress_pct / 100.0)
                )
                reward += timeout_penalty

            # ── reward clamp ──
            reward = float(np.clip(reward, REWARD_CLAMP_MIN, REWARD_CLAMP_MAX))

            terminal = bool(
                done or forward_done or current_collision
                or no_progress_done or timeout_done
            )

            if is_valid_transition(obs, action, reward, next_obs):
                trainer.buffer.push(obs, action, reward, next_obs, float(terminal))

            episode_reward += float(reward)
            total_steps += 1

            # 매 스텝 업데이트 (fine-tuning은 warmup 없음)
            trainer.update()

            obs = next_obs
            obs_raw = next_obs_raw

            if terminal:
                break

        # ── 에피소드 로그 ──
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        lap_time_str = format_lap_times(lap_times)

        print(
            f'[finetune] '
            f'ep {episode:4d} | '
            f'reward: {episode_reward:8.1f} | '
            f'wp: {progress_score}/{n_waypoints * MAX_LAPS} '
            f'({progress_pct:.1f}%) | '
            f'speed: {avg_speed:.2f} | '
            f'line: {last_line_idx} | '
            f'switches: {ep_line_switches} '
            f'(straight: {ep_straight_switches}) | '
            f'crash: {collisions} | '
            f'lap: {lap_time_str} | '
            f'steps: {step_in_ep + 1}'
        )

        # ── eval & save ──
        if (
            episode > 0
            and episode % config.get('eval_interval', 5) == 0
        ):
            eval_reward = trainer.evaluate(
                env, waypoints_lines, controller, init_poses,
            )
            print(
                f'  [EVAL] reward: {eval_reward:.1f} '
                f'(best: {best_reward:.1f}) | '
                f'누적 라인전환: {total_line_switches} '
                f'(직진: {total_straight_switches})'
            )

            if eval_reward > best_reward:
                best_reward = eval_reward
                save_path = config['save_path']
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                trainer.save(save_path)
                print(f'  모델 저장: {save_path}')

    env.close()

    print(f'\n{"═" * 60}')
    print('Fine-tuning 완료')
    print(f'{"═" * 60}')
    print(f'총 에피소드     : {config["max_episodes"]}')
    print(f'총 라인 전환    : {total_line_switches}')
    print(f'직진 라인 전환  : {total_straight_switches}')
    print(f'best eval reward: {best_reward:.1f}')
    print(f'저장 경로       : {config["save_path"]}')
    print(f'{"═" * 60}\n')


if __name__ == '__main__':
    main()