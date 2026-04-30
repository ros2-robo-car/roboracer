"""
config.py
─────────
프로젝트 전체 설정 파일

맵을 바꾸고 싶으면 MAP_NAME만 변경하면 됩니다.
"""

import os

# ══════════════════════════════════════════════════════════════════════════════
# 맵 설정 (여기만 바꾸면 전체 적용)
# ══════════════════════════════════════════════════════════════════════════════
MAP_NAME = 'Spielberg'    # 사용할 맵 이름

RACETRACKS_DIR = os.path.expanduser('~/f1tenth_racetracks')
MAP_DIR        = os.path.join(RACETRACKS_DIR, MAP_NAME)
MAP_PATH       = os.path.join(MAP_DIR, f'{MAP_NAME}_map')
MAP_EXT        = '.png'
CENTERLINE_CSV = os.path.join(MAP_DIR, f'{MAP_NAME}_centerline.csv')


# ══════════════════════════════════════════════════════════════════════════════
# 환경 설정
# ══════════════════════════════════════════════════════════════════════════════
ENV_CONFIG = {
    'map'       : MAP_PATH,
    'map_ext'   : MAP_EXT,
    'num_agents': 1,
    'timestep'  : 0.01,
    'ego_idx'   : 0,
    'params'    : {
        'mu'      : 1.0489,
        'C_Sf'    : 4.718,
        'C_Sr'    : 5.4562,
        'lf'      : 0.15875,
        'lr'      : 0.17145,
        'h'       : 0.074,
        'm'       : 3.74,
        'I'       : 0.04712,
        's_min'   : -0.4189,
        's_max'   : 0.4189,
        'sv_min'  : -3.2,
        'sv_max'  : 3.2,
        'v_switch': 7.319,
        'a_max'   : 9.51,
        'v_min'   : -5.0,
        'v_max'   : 20.0,
        'width'   : 0.31,
        'length'  : 0.58,
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# 관측 설정
# ══════════════════════════════════════════════════════════════════════════════
OBS_CONFIG = {
    'lidar_size'     : 108,
    'lidar_range_min': 0.1,
    'lidar_range_max': 10.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 라인 설정
# ══════════════════════════════════════════════════════════════════════════════
LINE_CONFIG = {
    'num_lines'     : 5,
    'line_spacing'  : 0.15,
    'centerline_csv': CENTERLINE_CSV,
    'map_path'      : MAP_PATH,
    'map_ext'       : MAP_EXT,
    # 시작 위치 설정
    'start_line_idx'     : None,   # None이면 가운데 line 사용
    'start_wp_idx'       : 0,      # 몇 번째 waypoint에서 시작할지
    'start_lookahead_idx': 5,
}


# ══════════════════════════════════════════════════════════════════════════════
# 모델 설정
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIG = {
    'type'       : 'SAC',
    'hidden_dims': [1024, 512, 1024, 1024, 512, 256],
    'action_dim' : 2,       # [라인 선택, 목표 속도]
    'num_lines'  : LINE_CONFIG['num_lines'],
}


# ══════════════════════════════════════════════════════════════════════════════
# 학습 설정
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_CONFIG = {
    'buffer_size'  : 100000,
    'batch_size'   : 256,
    'gamma'        : 0.99,
    'tau'          : 0.005,
    'lr_actor'     : 3e-4,
    'lr_critic'    : 3e-4,
    'lr_alpha'     : 1e-3,
    'max_episodes' : 5000,
    'max_steps'    : 15000,
    'eval_interval': 5,
    'warmup_steps' : 20000,
}


# ══════════════════════════════════════════════════════════════════════════════
# 속도 설정
# ══════════════════════════════════════════════════════════════════════════════
SPEED_MIN = -5.0
SPEED_MAX = 13.0

# ══════════════════════════════════════════════════════════════════════════════
# Pure Pursuit 설정
# ══════════════════════════════════════════════════════════════════════════════
PURE_PURSUIT_CONFIG = {
    'wheelbase'                 : ENV_CONFIG['params']['lf'] + ENV_CONFIG['params']['lr'],

    # 조향용 lookahead: 짧게 봐야 코너에서 조향각이 충분히 나옴
    'min_lookahead'             : 0.3,
    'max_lookahead'             : 1.0,
    'lookahead_gain'            : 0.15,

    # 속도 감속용: 멀리 있는 커브를 미리 보고 감속
    'curvature_gain'            : 3.0,
    'lookahead_window_base'     : 5,
    'lookahead_window_speed_scale': 2,

    # 곡률 샘플링 간격: window가 커져도 촘촘하게 곡률 계산
    'curvature_sample_step'     : 2,

    # 조향 중일 때 속도 회복을 막기 위한 감속 비율
    'steering_speed_gain'       : 1.0,

    'max_steering'              : ENV_CONFIG['params']['s_max'],
    'smooth_alpha'              : 0.8,
}


# ══════════════════════════════════════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'sac_model.pth')
MULTIMAP_PATH  = os.path.join(PROJECT_ROOT, 'models', 'sac_model_multimap.pth')
MULTIMAP_FINAL_PATH  = os.path.join(PROJECT_ROOT, 'models', 'sac_model_multimap_final.pth')
QUANTIZED_PATH  = os.path.join(PROJECT_ROOT, 'models', 'sac_model_quantized.pth')


# ══════════════════════════════════════════════════════════════════════════════
# 평가 설정
# ══════════════════════════════════════════════════════════════════════════════
EVAL_EPISODES     = 5
EVAL_MAX_STEPS    = 15000
QUANTIZE_EPISODES = 5


# ══════════════════════════════════════════════════════════════════════════════
# reward 설정
# ══════════════════════════════════════════════════════════════════════════════
REWARD_CONFIG = {
    # 체크포인트 설정
    'num_checkpoints'     : 10,
    'checkpoint_arrival'  : 10.0,
    'speed_reward_scale'  : 5.0,
    'baseline_steps'      : 500,

    # 충돌 페널티 (curriculum)
    'collision_penalty_start': -5.0,
    'collision_penalty_end'  : -100.0,
    'collision_curriculum_episodes': 100,

    # 정지 페널티
    'stall_speed_threshold': 0.3,
    'stall_penalty'        : -0.5,

    # waypoint 진행 보상 (dense)
    'waypoint_progress_reward': 1.0,

    # forward progress 설정
    'max_laps'            : 2,
    'max_forward_wp_jump' : 30,
}

# ══════════════════════════════════════════════════════════════════════════════
# 멀티맵 학습 설정
# ══════════════════════════════════════════════════════════════════════════════
MULTIMAP_CONFIG = {
    'num_cycles'        : 25,       # 총 사이클 수
    'warmup_steps'      : 30000,    # 사이클당 warmup 스텝
    'train_steps'       : 75000,    # 사이클당 학습 스텝
    'max_steps_per_ep'  : 15000,     # 에피소드당 최대 스텝
    'eval_episodes'     : 1,        # 평가 시 에피소드 수
}
 
# f1tenth_racetracks 내 사용할 맵 목록
MAP_LIST = [
    'Austin',
    'BrandsHatch',
    'Budapest',
    'Catalunya',
    'Hockenheim',
    'IMS',
    'Melbourne',
    'MexicoCity',
    'Montreal',
    'Monza',
    'MoscowRaceway',
    'Nuerburgring',
    'Oschersleben',
    'Sakhir',
    'SaoPaulo',
    'Sepang',
    'Shanghai',
    'Silverstone',
    'Sochi',
    'Spa',
    'Spielberg',
    'YasMarina',
    'Zandvoort',
]
