"""
multimap_train.py
─────────────────
여러 맵을 순서대로 돌며 학습하는 멀티맵 학습 스크립트.

현재 train_node.py 구조에 맞춘 버전:
- ForwardProgressTracker 사용
- WarmupBaselineCollector 사용하지 않음
- baseline_segment_steps 사용하지 않음
- compute_reward()는 기존 train_node.py 시그니처 유지
"""

import os
import sys

import gym
import f110_gym
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import (
    RACETRACKS_DIR,
    ENV_CONFIG,
    OBS_CONFIG,
    LINE_CONFIG,
    MODEL_CONFIG,
    REWARD_CONFIG,
    MULTIMAP_CONFIG,
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
    CheckpointTracker,
    Trainer,
    ForwardProgressTracker,
    preprocess_obs,
    action_to_env,
    compute_reward,
    make_warmup_action,
    make_init_pose,
    is_valid_obs,
    NUM_CHECKPOINTS,
    MAX_LAPS,
    MAX_FORWARD_WP_JUMP,
    WAYPOINT_PROGRESS_REWARD,
)


STEERING_CHANGE_PENALTY = REWARD_CONFIG.get('steering_change_penalty', 0.0)
STEERING_CHANGE_THRESHOLD = REWARD_CONFIG.get('steering_change_threshold', 0.1)


DEFAULT_FIXED_EVAL_MAPS = [
    'Spielberg',
    'Sepang',
    'Monza',
    'Austin',
    'Nuerburgring',
]


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

    if os.path.exists(csv_path):
        return load_waypoints(
            centerline_path=csv_path,
            num_lines=LINE_CONFIG['num_lines'],
            line_spacing=LINE_CONFIG['line_spacing'],
        )

    return load_waypoints(
        map_path=map_paths['map_path'],
        map_ext=map_paths['map_ext'],
        num_lines=LINE_CONFIG['num_lines'],
        line_spacing=LINE_CONFIG['line_spacing'],
    )


def validate_maps() -> list:
    valid = []

    for name in MAP_LIST:
        paths = get_map_paths(name)

        has_csv = os.path.exists(paths['centerline_csv'])
        has_img = os.path.exists(paths['map_path'] + '.png')
        has_yaml = os.path.exists(paths['map_path'] + '.yaml')

        if has_csv or (has_img and has_yaml):
            valid.append(name)
        else:
            print(f'  [스킵] {name}: 맵 파일 없음')

    return valid


def build_fixed_eval_maps(valid_maps: list) -> list:
    configured = MULTIMAP_CONFIG.get('eval_maps', DEFAULT_FIXED_EVAL_MAPS)

    eval_maps = [
        name for name in configured
        if name in valid_maps
    ]

    if not eval_maps:
        eval_maps = valid_maps[:min(5, len(valid_maps))]

    return eval_maps


class MapCache:
    def __init__(self, map_names: list):
        self.map_names = map_names
        self.cache = {}

    def load(self, map_name: str) -> dict:
        if map_name in self.cache:
            return self.cache[map_name]

        paths = get_map_paths(map_name)
        wp = load_map_waypoints(paths)

        waypoints_lines = wp['lines']
        num_lines = len(waypoints_lines)
        reference_line = waypoints_lines[num_lines // 2]
        n_waypoints = len(reference_line)
        init_poses = make_init_pose(waypoints_lines)

        data = {
            'map_name': map_name,
            'paths': paths,
            'waypoints_lines': waypoints_lines,
            'reference_line': reference_line,
            'n_waypoints': n_waypoints,
            'init_poses': init_poses,
        }

        self.cache[map_name] = data
        return data


def run_episode(
    map_data: dict,
    trainer: Trainer,
    controller: PurePursuitController,
    is_warmup: bool,
    max_steps_ep: int,
) -> dict:
    map_name = map_data['map_name']
    waypoints_lines = map_data['waypoints_lines']
    reference_line = map_data['reference_line']
    n_waypoints = map_data['n_waypoints']
    init_poses = map_data['init_poses']
    num_lines = MODEL_CONFIG['num_lines']

    env_cfg = make_env_config(map_data['paths'])
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

        tracker = CheckpointTracker(n_waypoints)

        episode_reward = 0.0
        speeds = []

        prev_steering = 0.0
        ep_steps = 0
        line_idx = 0

        progress_score = 1
        progress_pct = (
            progress_score / (n_waypoints * MAX_LAPS) * 100.0
            if n_waypoints > 0
            else 0.0
        )

        for _ in range(max_steps_ep):
            if is_warmup:
                action, env_action = make_warmup_action(
                    obs_raw,
                    waypoints_lines,
                    num_lines,
                    controller,
                )
            else:
                if not is_valid_obs(obs):
                    print('[WARN] invalid obs before select_action. terminate episode.')
                    break

                action = trainer.model.select_action(obs, training=True)
                action = action + np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, -1.0, 1.0)

                env_action = action_to_env(
                    action,
                    obs_raw,
                    trainer.model,
                    waypoints_lines,
                    controller,
                )

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))

            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            speeds.append(abs(float(next_obs_raw['linear_vels_x'][0])))

            reward, line_idx, _ = compute_reward(
                next_obs_raw,
                action,
                trainer.model,
                waypoints_lines,
                tracker,
            )

            reward += WAYPOINT_PROGRESS_REWARD * progress_delta

            cur_steer = float(env_action[0])
            steer_change = abs(cur_steer - prev_steering)

            if steer_change > STEERING_CHANGE_THRESHOLD:
                reward -= STEERING_CHANGE_PENALTY * (
                    steer_change - STEERING_CHANGE_THRESHOLD
                )

            prev_steering = cur_steer

            next_obs = preprocess_obs(
                next_obs_raw,
                waypoints_lines,
                num_lines,
            )

            terminal = bool(done or forward_done)

            if is_valid_obs(obs) and is_valid_obs(next_obs):
                trainer.buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    float(terminal),
                )

            obs = next_obs
            obs_raw = next_obs_raw
            episode_reward += float(reward)
            ep_steps += 1

            if not is_warmup:
                trainer.update()

            if terminal:
                break

        lap_count = int(obs_raw['lap_counts'][0])
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

    finally:
        env.close()

    return {
        'map_name': map_name,
        'reward': episode_reward,
        'lap_count': lap_count,
        'cp_passed': tracker.checkpoints_passed,
        'cp_total': NUM_CHECKPOINTS,
        'avg_speed': avg_speed,
        'line_idx': line_idx,
        'ep_steps': ep_steps,
        'progress_score': progress_score,
        'progress_pct': progress_pct,
        'n_waypoints': n_waypoints,
        'ignored_jump': progress_tracker.ignored_jump_count,
    }


def print_episode(tag: str, ep: int, r: dict, total_steps: int):
    n_wp_total = r['n_waypoints'] * MAX_LAPS

    print(
        f'[{tag:12s}] ep {ep:4d} | '
        f'{r["map_name"]:16s} | '
        f'reward: {r["reward"]:8.2f} | '
        f'lap: {r["lap_count"]} | '
        f'cp: {r["cp_passed"]:2d}/{r["cp_total"]} | '
        f'speed: {r["avg_speed"]:.2f} | '
        f'line: {r["line_idx"]} | '
        f'steps: {total_steps} | '
        f'wp: {r["progress_score"]}/{n_wp_total} '
        f'({r["progress_pct"]:.1f}%) | '
        f'ignored_jump: {r["ignored_jump"]}'
    )


def run_evaluation(
    eval_maps: list,
    map_cache: MapCache,
    trainer: Trainer,
    controller: PurePursuitController,
    max_steps_ep: int,
    n_episodes_per_map: int,
) -> float:
    total_reward = 0.0
    total_episodes = 0
    num_lines = MODEL_CONFIG['num_lines']

    for map_name in eval_maps:
        map_data = map_cache.load(map_name)

        for _ in range(n_episodes_per_map):
            env_cfg = make_env_config(map_data['paths'])
            env = gym.make('f110_gym:f110-v0', **env_cfg)

            try:
                obs_raw, _, _, _ = env.reset(poses=map_data['init_poses'])
                obs = preprocess_obs(
                    obs_raw,
                    map_data['waypoints_lines'],
                    num_lines,
                )

                progress_tracker = ForwardProgressTracker(
                    map_data['reference_line'],
                    max_laps=MAX_LAPS,
                    max_forward_jump=MAX_FORWARD_WP_JUMP,
                )
                progress_tracker.reset_from_obs(obs_raw)

                eval_tracker = CheckpointTracker(map_data['n_waypoints'])

                episode_reward = 0.0
                prev_steering = 0.0

                for _ in range(max_steps_ep):
                    if not is_valid_obs(obs):
                        break

                    action = trainer.model.select_action(
                        obs,
                        training=False,
                    )

                    env_action = action_to_env(
                        action,
                        obs_raw,
                        trainer.model,
                        map_data['waypoints_lines'],
                        controller,
                    )

                    next_obs_raw, _, done, _ = env.step(
                        np.array([env_action])
                    )

                    progress_score, _, forward_done, progress_delta = (
                        progress_tracker.update(next_obs_raw)
                    )

                    reward, _, _ = compute_reward(
                        next_obs_raw,
                        action,
                        trainer.model,
                        map_data['waypoints_lines'],
                        eval_tracker,
                    )

                    reward += WAYPOINT_PROGRESS_REWARD * progress_delta

                    cur_steer = float(env_action[0])
                    steer_change = abs(cur_steer - prev_steering)

                    if steer_change > STEERING_CHANGE_THRESHOLD:
                        reward -= STEERING_CHANGE_PENALTY * (
                            steer_change - STEERING_CHANGE_THRESHOLD
                        )

                    prev_steering = cur_steer

                    obs = preprocess_obs(
                        next_obs_raw,
                        map_data['waypoints_lines'],
                        num_lines,
                    )

                    obs_raw = next_obs_raw
                    episode_reward += float(reward)

                    if done or forward_done:
                        break

                total_reward += episode_reward
                total_episodes += 1

            finally:
                env.close()

    if total_episodes == 0:
        return -float('inf')

    return total_reward / total_episodes


def main():
    num_cycles = MULTIMAP_CONFIG['num_cycles']
    warmup_steps = MULTIMAP_CONFIG['warmup_steps']
    train_steps = MULTIMAP_CONFIG['train_steps']
    max_steps_ep = MULTIMAP_CONFIG['max_steps_per_ep']
    eval_episodes = MULTIMAP_CONFIG['eval_episodes']

    total_target = (warmup_steps + train_steps) * num_cycles

    print(f'\n멀티맵 순차 학습')
    print(f'  사이클: {num_cycles}회')
    print(
        f'  사이클당: warmup {warmup_steps:,} + '
        f'학습 {train_steps:,} = {warmup_steps + train_steps:,} 스텝'
    )
    print(f'  총 목표: 약 {total_target:,} 스텝')
    print(f'  max_laps: {MAX_LAPS}')
    print(f'  max_forward_wp_jump: {MAX_FORWARD_WP_JUMP}')
    print(f'  waypoint_progress_reward: {WAYPOINT_PROGRESS_REWARD}')

    print(f'\n맵 검증 중... (racetracks: {RACETRACKS_DIR})')

    valid_maps = validate_maps()

    print(f'사용 가능한 맵: {len(valid_maps)}개')
    print(f'학습 맵 순서: {valid_maps}\n')

    if not valid_maps:
        print('사용 가능한 맵이 없습니다.')
        return

    fixed_eval_maps = build_fixed_eval_maps(valid_maps)

    print(f'고정 평가 맵: {fixed_eval_maps}\n')

    map_cache = MapCache(valid_maps)

    obs_dim = get_obs_dim(
        OBS_CONFIG['lidar_size'],
        MODEL_CONFIG['num_lines'],
    )

    controller = PurePursuitController(
        max_speed=SPEED_MAX,
        min_speed=SPEED_MIN,
    )

    trainer = Trainer(obs_dim)

    best_reward = -float('inf')
    total_steps = 0
    total_episodes = 0

    for cycle in range(num_cycles):
        cycle_start = total_steps

        cycle_map_name = valid_maps[cycle % len(valid_maps)]
        cycle_map_data = map_cache.load(cycle_map_name)

        print(f'\n{"═" * 70}')
        print(
            f'사이클 {cycle + 1}/{num_cycles} | '
            f'맵: {cycle_map_name} | '
            f'총 스텝: {total_steps:,} | '
            f'총 에피소드: {total_episodes}'
        )
        print(f'{"═" * 70}')

        phase_start = total_steps
        phase_target = phase_start + warmup_steps
        phase_ep = 0

        print(
            f'\n[Warmup] 목표: ~{warmup_steps:,} 스텝 | '
            f'맵: {cycle_map_name}'
        )

        while total_steps < phase_target:
            result = run_episode(
                map_data=cycle_map_data,
                trainer=trainer,
                controller=controller,
                is_warmup=True,
                max_steps_ep=max_steps_ep,
            )

            total_steps += result['ep_steps']
            total_episodes += 1
            phase_ep += 1

            print_episode(
                'warmup',
                total_episodes,
                result,
                total_steps,
            )

        print(
            f'Warmup 완료 | '
            f'맵: {cycle_map_name} | '
            f'실제: {total_steps - phase_start:,} 스텝, '
            f'{phase_ep} 에피소드'
        )

        phase_start = total_steps
        phase_target = phase_start + train_steps
        phase_ep = 0

        print(
            f'\n[학습] 목표: ~{train_steps:,} 스텝 | '
            f'맵: {cycle_map_name}'
        )

        while total_steps < phase_target:
            result = run_episode(
                map_data=cycle_map_data,
                trainer=trainer,
                controller=controller,
                is_warmup=False,
                max_steps_ep=max_steps_ep,
            )

            total_steps += result['ep_steps']
            total_episodes += 1
            phase_ep += 1

            print_episode(
                f'C{cycle + 1} train',
                total_episodes,
                result,
                total_steps,
            )

        print(
            f'학습 완료 | '
            f'맵: {cycle_map_name} | '
            f'실제: {total_steps - phase_start:,} 스텝, '
            f'{phase_ep} 에피소드'
        )

        eval_reward_current = run_evaluation(
            eval_maps=[cycle_map_name],
            map_cache=map_cache,
            trainer=trainer,
            controller=controller,
            max_steps_ep=max_steps_ep,
            n_episodes_per_map=eval_episodes,
        )

        print(
            f'  [현재맵 평가] 사이클 {cycle + 1} | '
            f'{cycle_map_name} | '
            f'eval reward: {eval_reward_current:.2f}'
        )

        eval_reward_global = run_evaluation(
            eval_maps=fixed_eval_maps,
            map_cache=map_cache,
            trainer=trainer,
            controller=controller,
            max_steps_ep=max_steps_ep,
            n_episodes_per_map=eval_episodes,
        )

        print(
            f'  [고정맵 평가] 사이클 {cycle + 1} | '
            f'eval reward: {eval_reward_global:.2f} | '
            f'best: {best_reward:.2f}'
        )

        if eval_reward_global >= best_reward:
            best_reward = eval_reward_global
            trainer.save(MULTIMAP_PATH)

        cycle_steps = total_steps - cycle_start

        print(
            f'\n사이클 {cycle + 1} 요약 | '
            f'맵: {cycle_map_name} | '
            f'스텝: {cycle_steps:,} | '
            f'누적: {total_steps:,}'
        )

    trainer.save(MULTIMAP_FINAL_PATH)

    print(f'\n{"═" * 70}')
    print('전체 학습 완료')
    print(f'  총 스텝: {total_steps:,}')
    print(f'  총 에피소드: {total_episodes}')
    print(f'  best eval reward: {best_reward:.2f}')
    print(f'{"═" * 70}\n')


if __name__ == '__main__':
    main()