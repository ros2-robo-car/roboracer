"""
multimap_train.py

여러 맵을 순서대로 돌며 학습하는 멀티맵 학습 스크립트

"""

import os
import sys

import gym
import f110_gym
import numpy as np
import torch

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
    WarmupCheckpointBaseline,
    Trainer,
    ForwardProgressTracker,
    preprocess_obs,
    action_to_env,
    compute_reward,
    make_warmup_action,
    make_init_pose,
    is_valid_obs,
    get_collision_penalty,
    compute_steer_change_penalty,
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
    format_lap_times,
)


DEFAULT_FIXED_EVAL_MAPS = [
    'Spielberg',
    'Sepang',
    'Monza',
    'Austin',
    'Nuerburgring',
]


def is_valid_transition(obs, action, reward, next_obs) -> bool:
    return (
        is_valid_obs(obs)
        and is_valid_obs(next_obs)
        and np.isfinite(action).all()
        and np.isfinite(reward)
    )


def save_trainer(trainer: Trainer, path: str):
    """
    train_node.Trainer에 save()가 없는 경우를 대비한 저장 함수.
    """
    if hasattr(trainer, 'save') and callable(getattr(trainer, 'save')):
        trainer.save(path)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            'model_state': trainer.model.state_dict(),
            'model_config': {
                'action_dim': MODEL_CONFIG['action_dim'],
                'hidden_dims': MODEL_CONFIG['hidden_dims'],
                'num_lines': MODEL_CONFIG['num_lines'],
                'use_line_curvature': OBS_CONFIG.get('use_line_curvature', False),
                'obs_dim': get_obs_dim(
                    OBS_CONFIG['lidar_size'],
                    MODEL_CONFIG['num_lines'],
                    use_line_curvature=OBS_CONFIG.get('use_line_curvature', False),
                ),
            },
        },
        path,
    )

    print(f'  모델 저장: {path}')


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
            width_fraction=LINE_CONFIG.get('line_width_fraction', 0.60),
        )

    return load_waypoints(
        map_path=map_paths['map_path'],
        map_ext=map_paths['map_ext'],
        num_lines=LINE_CONFIG['num_lines'],
        line_spacing=LINE_CONFIG['line_spacing'],
        width_fraction=LINE_CONFIG.get('line_width_fraction', 0.60),
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


def build_map_baselines(valid_maps: list) -> dict:
    baselines = {}
    multiplier = REWARD_CONFIG.get('warmup_baseline_multiplier', 1.5)
    for map_name in valid_maps:
        baselines[map_name] = WarmupCheckpointBaseline(
            num_checkpoints=NUM_CHECKPOINTS,
            fallback_steps=BASELINE_STEPS,
            min_samples=WARMUP_BASELINE_MIN_SAMPLES,
            multiplier=multiplier,
        )
    return baselines


def run_episode(
    map_data: dict,
    trainer: Trainer,
    controller: PurePursuitController,
    baseline_provider: WarmupCheckpointBaseline,
    is_warmup: bool,
    max_steps_ep: int,
    episode_for_penalty: int,
) -> dict:
    map_name = map_data['map_name']
    waypoints_lines = map_data['waypoints_lines']
    reference_line = map_data['reference_line']
    n_waypoints = map_data['n_waypoints']
    init_poses = map_data['init_poses']
    num_lines = MODEL_CONFIG['num_lines']

    env_cfg = make_env_config(map_data['paths'])
    env = gym.make('f110_gym:f110-v0', **env_cfg)

    episode_reward = 0.0
    speeds = []
    ep_steps = 0
    lap_count = 0

    progress_score = 1
    progress_pct = (
        progress_score / (n_waypoints * MAX_LAPS) * 100.0
        if n_waypoints > 0
        else 0.0
    )

    line_idx = -1
    collisions = 0

    progress_tracker = None
    tracker = None
    prev_steering = None
    no_progress_bad_count = 0
    progress_window_sum = 0.0
    progress_window_steps = 0
    timeout_done = False
    no_progress_done = False
    lap_times = []
    last_lap_step = 0
    next_lap_progress = n_waypoints

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

        for step_in_ep in range(max_steps_ep):
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
                if SPEED_ACTION_NOISE_STD > 0.0:
                    speed_noise = np.random.normal(0.0, SPEED_ACTION_NOISE_STD)
                    action[1] = float(np.clip(action[1] + speed_noise, -1.0, 1.0))

                env_action = action_to_env(
                    action,
                    obs_raw,
                    trainer.model,
                    waypoints_lines,
                    controller,
                )

            next_obs_raw, _, done, _ = env.step(np.array([env_action]))

            current_collision = bool(next_obs_raw['collisions'][0])
            if current_collision:
                collisions += 1

            next_obs = preprocess_obs(
                next_obs_raw,
                waypoints_lines,
                num_lines,
            )

            if not is_valid_obs(next_obs):
                invalid_penalty = (
                    get_collision_penalty(episode_for_penalty)
                    * INVALID_OBS_PENALTY_SCALE
                )
                reward = invalid_penalty

                if (
                    is_valid_obs(obs)
                    and np.isfinite(action).all()
                    and np.isfinite(reward)
                ):
                    trainer.buffer.push(
                        obs,
                        action,
                        reward,
                        obs,
                        1.0,
                    )

                episode_reward += float(reward)
                ep_steps += 1

                print('[WARN] invalid next_obs after env.step. apply penalty and terminate episode.')
                break

            progress_score, progress_pct, forward_done, progress_delta = (
                progress_tracker.update(next_obs_raw)
            )

            while progress_score >= next_lap_progress and len(lap_times) < MAX_LAPS:
                lap_steps = ep_steps + 1 - last_lap_step
                lap_time = lap_steps * env_cfg.get('timestep', ENV_CONFIG.get('timestep', 0.01))
                lap_times.append(float(lap_time))
                last_lap_step = ep_steps + 1
                next_lap_progress += n_waypoints

            speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))
            if np.isfinite(speed_value):
                speeds.append(speed_value)

            (
                reward,
                line_idx,
                _,
                checkpoint_passed,
                segment_steps,
                checkpoint_idx,
            ) = compute_reward(
                next_obs_raw,
                action,
                trainer.model,
                waypoints_lines,
                tracker,
                episode=episode_for_penalty,
                baseline_provider=baseline_provider,
                use_speed_reward=not is_warmup,
            )

            # warmup 중에는 해당 맵의 checkpoint별 baseline sample을 저장
            if is_warmup and checkpoint_passed:
                baseline_provider.add(checkpoint_idx, segment_steps)

            current_steering = float(env_action[0])
            steer_change_penalty = compute_steer_change_penalty(
                speed_value,
                current_steering,
                prev_steering,
            )
            reward -= steer_change_penalty
            prev_steering = current_steering

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
                    print(
                        f'[NO_PROGRESS][{map_name}] terminate | '
                        f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                        f'({progress_pct:.1f}%) | '
                        f'bad_count={no_progress_bad_count} | '
                        f'penalty={NO_PROGRESS_TERMINAL_PENALTY}'
                    )

            timeout_done = step_in_ep == max_steps_ep - 1
            if timeout_done and not forward_done and not current_collision and not no_progress_done:
                timeout_penalty = (
                    TIMEOUT_FIXED_PENALTY
                    + TIMEOUT_PENALTY_SCALE * (1.0 - progress_pct / 100.0)
                )
                reward += timeout_penalty
                print(
                    f'[TIMEOUT][{map_name}] terminate | '
                    f'wp={progress_score}/{n_waypoints * MAX_LAPS} '
                    f'({progress_pct:.1f}%) | '
                    f'penalty={timeout_penalty:.1f}'
                )

            terminal = bool(
                done or forward_done or current_collision or no_progress_done or timeout_done
            )

            if is_valid_transition(obs, action, reward, next_obs):
                trainer.buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    float(terminal),
                )
            else:
                print('[WARN] invalid transition skipped.')

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

    ignored_jump = (
        progress_tracker.ignored_jump_count
        if progress_tracker is not None
        else 0
    )

    cp_passed = (
        tracker.checkpoints_passed
        if tracker is not None
        else 0
    )

    baseline_summary = (
        baseline_provider.compact_summary()
        if baseline_provider is not None
        else '-'
    )

    return {
        'map_name': map_name,
        'reward': episode_reward,
        'lap_count': lap_count,
        'cp_passed': cp_passed,
        'cp_total': NUM_CHECKPOINTS,
        'avg_speed': avg_speed,
        'line_idx': line_idx,
        'ep_steps': ep_steps,
        'progress_score': progress_score,
        'progress_pct': progress_pct,
        'n_waypoints': n_waypoints,
        'ignored_jump': ignored_jump,
        'crash': collisions,
        'timeout': int(timeout_done and not forward_done and collisions == 0 and not no_progress_done),
        'no_prog_bad': no_progress_bad_count,
        'lap_time': format_lap_times(lap_times),
        'baseline': baseline_summary,
    }


def print_episode(tag: str, ep: int, r: dict, total_steps: int):
    n_wp_total = r['n_waypoints'] * MAX_LAPS

    print(
        f'[{tag:12s}] ep {ep:4d} | '
        f'{r["map_name"]:16s} | '
        f'reward: {r["reward"]:8.2f} | '
        f'lap: {r["lap_count"]} | '
        f'crash: {r["crash"]} | '
        f'speed: {r["avg_speed"]:.2f} | '
        f'line: {r["line_idx"]} | '
        f'no_prog_bad: {r.get("no_prog_bad", 0)} | '
        f'timeout: {r.get("timeout", 0)} | '
        f'lap_time: {r.get("lap_time", "-")} | '
        f'steps: {total_steps} | '
        f'wp: {r["progress_score"]}/{n_wp_total} '
        f'({r["progress_pct"]:.1f}%) | '
    )


def run_evaluation(
    eval_maps: list,
    map_cache: MapCache,
    map_baselines: dict,
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
        baseline_provider = map_baselines.get(map_name, None)

        for _ in range(n_episodes_per_map):
            env_cfg = make_env_config(map_data['paths'])
            env = gym.make('f110_gym:f110-v0', **env_cfg)

            episode_reward = 0.0
            prev_steering = None
            progress_window_sum = 0.0
            progress_window_steps = 0
            no_progress_bad_count = 0

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

                for step_in_ep in range(max_steps_ep):
                    if not is_valid_obs(obs):
                        print('[WARN][eval] invalid obs before select_action. terminate episode.')
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

                    current_collision = bool(next_obs_raw['collisions'][0])
                    speed_value = abs(float(next_obs_raw['linear_vels_x'][0]))

                    next_obs = preprocess_obs(
                        next_obs_raw,
                        map_data['waypoints_lines'],
                        num_lines,
                    )

                    if not is_valid_obs(next_obs):
                        invalid_penalty = (
                            get_collision_penalty(COLLISION_CURRICULUM_EPISODES)
                            * INVALID_OBS_PENALTY_SCALE
                        )
                        episode_reward += float(invalid_penalty)

                        print('[WARN][eval] invalid next_obs. apply penalty and terminate episode.')
                        break

                    progress_score, progress_pct, forward_done, progress_delta = (
                        progress_tracker.update(next_obs_raw)
                    )

                    reward, line_idx, _, _, _, _ = compute_reward(
                        next_obs_raw,
                        action,
                        trainer.model,
                        map_data['waypoints_lines'],
                        eval_tracker,
                        episode=COLLISION_CURRICULUM_EPISODES,
                        baseline_provider=baseline_provider,
                        use_speed_reward=True,
                    )

                    current_steering = float(env_action[0])
                    steer_change_penalty = compute_steer_change_penalty(
                        speed_value,
                        current_steering,
                        prev_steering,
                    )
                    reward -= steer_change_penalty
                    prev_steering = current_steering

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

                    timeout_done = step_in_ep == max_steps_ep - 1
                    if timeout_done and not forward_done and not current_collision and not no_progress_done:
                        timeout_penalty = (
                            TIMEOUT_FIXED_PENALTY
                            + TIMEOUT_PENALTY_SCALE * (1.0 - progress_pct / 100.0)
                        )
                        reward += timeout_penalty

                    obs = next_obs
                    obs_raw = next_obs_raw
                    episode_reward += float(reward)

                    if done or forward_done or current_collision or no_progress_done or timeout_done:
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
    print(f"  line curvature feature: {OBS_CONFIG.get('use_line_curvature', False)}")
    print(f'  speed_action_noise_std: {SPEED_ACTION_NOISE_STD}')
    print(
        f'  no_progress: interval={NO_PROGRESS_CHECK_INTERVAL}, '
        f'min_delta={NO_PROGRESS_MIN_DELTA}, '
        f'patience={NO_PROGRESS_PATIENCE}, '
        f'terminal_penalty={NO_PROGRESS_TERMINAL_PENALTY}'
    )
    print(
        f'  timeout_penalty: fixed={TIMEOUT_FIXED_PENALTY}, '
        f'scale={TIMEOUT_PENALTY_SCALE}'
    )

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
    map_baselines = build_map_baselines(valid_maps)

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

    best_reward = -float('inf')
    total_steps = 0
    total_episodes = 0

    for cycle in range(num_cycles):
        cycle_start = total_steps

        cycle_map_name = valid_maps[cycle % len(valid_maps)]
        cycle_map_data = map_cache.load(cycle_map_name)
        cycle_baseline = map_baselines[cycle_map_name]

        # train_node.Trainer.evaluate()를 쓸 경우를 대비해 현재 맵 baseline을 연결
        trainer.checkpoint_baselines = cycle_baseline

        print(f'\n{"═" * 70}')
        print(
            f'사이클 {cycle + 1}/{num_cycles} | '
            f'맵: {cycle_map_name} | '
            f'총 스텝: {total_steps:,} | '
            f'총 에피소드: {total_episodes}'
        )
        print(f'baseline: {cycle_baseline.compact_summary()}')
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
                baseline_provider=cycle_baseline,
                is_warmup=True,
                max_steps_ep=max_steps_ep,
                episode_for_penalty=total_episodes,
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

            if result['ep_steps'] <= 0:
                print('[WARN] episode step is 0. break warmup loop to avoid infinite loop.')
                break

        print(
            f'Warmup 완료 | '
            f'맵: {cycle_map_name} | '
            f'실제: {total_steps - phase_start:,} 스텝, '
            f'{phase_ep} 에피소드'
        )
        print(f'Warmup baseline detail | {cycle_baseline.detail_summary()}')

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
                baseline_provider=cycle_baseline,
                is_warmup=False,
                max_steps_ep=max_steps_ep,
                episode_for_penalty=total_episodes,
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

            if result['ep_steps'] <= 0:
                print('[WARN] episode step is 0. break train loop to avoid infinite loop.')
                break

        print(
            f'학습 완료 | '
            f'맵: {cycle_map_name} | '
            f'실제: {total_steps - phase_start:,} 스텝, '
            f'{phase_ep} 에피소드'
        )

        eval_reward_current = run_evaluation(
            eval_maps=[cycle_map_name],
            map_cache=map_cache,
            map_baselines=map_baselines,
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
            map_baselines=map_baselines,
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
            save_trainer(trainer, MULTIMAP_PATH)

        cycle_steps = total_steps - cycle_start

        print(
            f'\n사이클 {cycle + 1} 요약 | '
            f'맵: {cycle_map_name} | '
            f'스텝: {cycle_steps:,} | '
            f'누적: {total_steps:,}'
        )

    save_trainer(trainer, MULTIMAP_FINAL_PATH)

    print(f'\n{"═" * 70}')
    print('전체 학습 완료')
    print(f'  총 스텝: {total_steps:,}')
    print(f'  총 에피소드: {total_episodes}')
    print(f'  best eval reward: {best_reward:.2f}')
    print(f'{"═" * 70}\n')


if __name__ == '__main__':
    main()