"""
waypoint_loader.py
──────────────────
트랙 맵에서 centerline을 로드하고
여러 후보 레이싱 라인(웨이포인트 배열)을 생성하는 모듈

지원하는 입력:
  1) centerline CSV 파일 (f1tenth_racetracks 형식)
     컬럼: [x_m, y_m, w_tr_right_m, w_tr_left_m]
  2) 맵 이미지 + yaml (centerline 자동 추출)
"""

import os
import csv
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── 설정 ─────────────────────────────────────────────────────────────────────
DEFAULT_NUM_LINES   = 5       # 생성할 후보 라인 수
DEFAULT_LINE_SPACE  = 0.15    # 라인 간 간격 (m)
DEFAULT_POINT_SPACE = 0.1     # 웨이포인트 간 최소 간격 (m)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Centerline 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_centerline_csv(csv_path: str) -> dict:
    """
    f1tenth_racetracks 형식 centerline CSV 로드

    반환:
        {
            'centerline': np.ndarray (N, 2),   # (x, y)
            'w_right':    np.ndarray (N,),      # 오른쪽 트랙 폭
            'w_left':     np.ndarray (N,),      # 왼쪽 트랙 폭
        }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'centerline 파일 없음: {csv_path}')

    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # 주석이나 헤더 건너뜀
            if not row or row[0].startswith('#'):
                continue
            try:
                vals = [float(v) for v in row]
                data.append(vals)
            except ValueError:
                continue

    data = np.array(data)

    if data.shape[1] >= 4:
        # [x_m, y_m, w_tr_right_m, w_tr_left_m]
        return {
            'centerline': data[:, :2],
            'w_right':    data[:, 2],
            'w_left':     data[:, 3],
        }
    elif data.shape[1] >= 2:
        # 최소 (x, y)만 있는 경우
        return {
            'centerline': data[:, :2],
            'w_right':    np.ones(len(data)) * 0.5,
            'w_left':     np.ones(len(data)) * 0.5,
        }
    else:
        raise ValueError(f'CSV 형식 오류: 최소 2개 컬럼 필요, {data.shape[1]}개 발견')


def load_map_yaml(yaml_path: str) -> dict:
    """맵 yaml 파일에서 resolution, origin 읽기"""
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    return {
        'resolution': float(meta['resolution']),
        'origin_x':   float(meta['origin'][0]),
        'origin_y':   float(meta['origin'][1]),
        'image':       meta.get('image', ''),
    }


def extract_centerline_from_map(map_path: str, map_ext: str = '.png',
                                 point_spacing: float = DEFAULT_POINT_SPACE) -> dict:
    """
    맵 이미지에서 centerline 자동 추출
    (EDT + Skeletonize 방식)

    Args:
        map_path: yaml 파일 경로 (확장자 제외)
        map_ext:  이미지 확장자
        point_spacing: 웨이포인트 간격 (m)

    반환: load_centerline_csv 와 동일한 형식
    """
    if not HAS_CV2:
        raise ImportError('OpenCV 필요: pip install opencv-python')

    # yaml 로드
    yaml_path = map_path + '.yaml' if not map_path.endswith('.yaml') else map_path
    map_meta = load_map_yaml(yaml_path)
    resolution = map_meta['resolution']
    origin_x   = map_meta['origin_x']
    origin_y   = map_meta['origin_y']

    # 이미지 로드
    img_path = os.path.splitext(yaml_path)[0] + map_ext
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'맵 이미지 없음: {img_path}')

    # 이진화: 흰색(주행 가능) = 1, 검은색(벽) = 0
    _, binary = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # 상하 반전 (ROS 맵 컨벤션: 원점이 좌하단)
    binary = np.flipud(binary)

    # EDT (Euclidean Distance Transform)
    edt = distance_transform_edt(binary)

    # EDT 임계값 적용 후 스켈레톤화
    threshold = np.max(edt) * 0.3
    edt_thresh = (edt > threshold).astype(np.uint8)
    skeleton = skeletonize(edt_thresh).astype(np.uint8)

    # 스켈레톤 위의 점 추출
    points_px = np.argwhere(skeleton > 0)  # (row, col) = (y, x)

    if len(points_px) == 0:
        raise RuntimeError('centerline 추출 실패: 스켈레톤 점 없음')

    # 점 정렬 (가장 가까운 이웃 순서로 연결)
    ordered = _order_points(points_px)

    # 픽셀 → 월드 좌표 변환
    world_points = np.zeros((len(ordered), 2))
    for i, (row, col) in enumerate(ordered):
        world_points[i, 0] = col * resolution + origin_x   # x
        world_points[i, 1] = row * resolution + origin_y   # y

    # 간격 기반 다운샘플링
    world_points = _resample_by_distance(world_points, point_spacing)

    # EDT 값으로 트랙 폭 추정
    widths = []
    for row, col in ordered:
        if 0 <= row < edt.shape[0] and 0 <= col < edt.shape[1]:
            widths.append(edt[row, col] * resolution)
        else:
            widths.append(0.5)

    widths = np.array(widths)
    # 다운샘플링에 맞게 폭도 보간
    if len(widths) != len(world_points):
        widths = np.interp(
            np.linspace(0, 1, len(world_points)),
            np.linspace(0, 1, len(widths)),
            widths
        )

    return {
        'centerline': world_points,
        'w_right':    widths,
        'w_left':     widths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. 후보 라인 생성
# ══════════════════════════════════════════════════════════════════════════════

def generate_racing_lines(centerline_data: dict,
                          num_lines: int = DEFAULT_NUM_LINES,
                          line_spacing: float = DEFAULT_LINE_SPACE) -> dict:
    """
    centerline 기반으로 여러 후보 라인 생성

    Args:
        centerline_data: load_centerline_csv 반환값
        num_lines:       생성할 라인 수 (홀수 권장, 중앙 = centerline)
        line_spacing:    라인 간 간격 (m)

    반환:
        {
            'lines':      list of np.ndarray (N, 2),  # 각 라인의 웨이포인트
            'centerline': np.ndarray (N, 2),
            'normals':    np.ndarray (N, 2),           # 각 점의 법선 벡터
            'offsets':    list of float,               # 각 라인의 오프셋 (m)
        }
    """
    center = centerline_data['centerline']
    w_right = centerline_data['w_right']
    w_left  = centerline_data['w_left']
    N = len(center)

    # 각 점에서의 법선 벡터 계산
    normals = _compute_normals(center)

    # 오프셋 목록 생성 (음수 = 안쪽/오른쪽, 양수 = 바깥쪽/왼쪽)
    half = num_lines // 2
    offsets = [i * line_spacing for i in range(-half, half + 1)]

    # 라인이 num_lines보다 많으면 자르기
    offsets = offsets[:num_lines]

    lines = []
    valid_offsets = []

    for offset in offsets:
        line = np.zeros_like(center)
        valid = True

        for i in range(N):
            # 오프셋 적용 (법선 방향으로 이동)
            shifted = center[i] + offset * normals[i]

            # 트랙 범위 체크: 벽을 넘지 않도록 클리핑
            max_right = w_right[i] * 0.85   # 벽에서 15% 여유
            max_left  = w_left[i]  * 0.85

            if offset < 0 and abs(offset) > max_right:
                # 오른쪽 벽 초과 → 최대한 안쪽으로
                shifted = center[i] + (-max_right) * normals[i]
            elif offset > 0 and offset > max_left:
                # 왼쪽 벽 초과 → 최대한 바깥쪽으로
                shifted = center[i] + max_left * normals[i]

            line[i] = shifted

        lines.append(line)
        valid_offsets.append(offset)

    return {
        'lines':      lines,
        'centerline': center,
        'normals':    normals,
        'offsets':    valid_offsets,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 편의 함수
# ══════════════════════════════════════════════════════════════════════════════

def load_waypoints(centerline_path: str = None,
                   map_path: str = None,
                   map_ext: str = '.png',
                   num_lines: int = DEFAULT_NUM_LINES,
                   line_spacing: float = DEFAULT_LINE_SPACE) -> dict:
    """
    편의 함수: centerline 로드 + 후보 라인 생성을 한 번에

    사용법:
        # CSV가 있는 경우
        wp = load_waypoints(centerline_path='vegas_centerline.csv')

        # 맵 이미지에서 추출하는 경우
        wp = load_waypoints(map_path='~/f1tenth_gym/.../vegas')
    """
    if centerline_path is not None:
        cl_data = load_centerline_csv(centerline_path)
    elif map_path is not None:
        cl_data = extract_centerline_from_map(map_path, map_ext)
    else:
        raise ValueError('centerline_path 또는 map_path 중 하나를 지정해야 합니다')

    return generate_racing_lines(cl_data, num_lines, line_spacing)


def get_nearest_waypoint_idx(position: np.ndarray,
                              waypoints: np.ndarray) -> int:
    """현재 위치에서 가장 가까운 웨이포인트 인덱스 반환"""
    dists = np.linalg.norm(waypoints - position[:2], axis=1)
    return int(np.argmin(dists))


def get_lookahead_point(position: np.ndarray,
                         waypoints: np.ndarray,
                         lookahead_dist: float,
                         nearest_idx: int = None) -> tuple:
    """
    현재 위치에서 lookahead 거리에 해당하는 웨이포인트 반환

    Args:
        position:       현재 위치 (x, y)
        waypoints:      웨이포인트 배열 (N, 2)
        lookahead_dist: 탐색 거리 (m)
        nearest_idx:    가장 가까운 점 인덱스 (None이면 자동 계산)

    반환:
        (target_point, target_idx)
    """
    if nearest_idx is None:
        nearest_idx = get_nearest_waypoint_idx(position, waypoints)

    N = len(waypoints)

    # 가장 가까운 점부터 앞쪽으로 탐색
    for i in range(1, N):
        idx = (nearest_idx + i) % N
        dist = np.linalg.norm(waypoints[idx] - position[:2])
        if dist >= lookahead_dist:
            return waypoints[idx], idx

    # 찾지 못하면 마지막으로 시도한 점 반환
    last_idx = (nearest_idx + N - 1) % N
    return waypoints[last_idx], last_idx


# ══════════════════════════════════════════════════════════════════════════════
# 내부 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def _compute_normals(points: np.ndarray) -> np.ndarray:
    """
    경로의 각 점에서 법선 벡터 계산
    (접선 벡터를 90도 회전)
    """
    N = len(points)
    normals = np.zeros_like(points)

    for i in range(N):
        # 전후 점으로 접선 벡터 계산 (순환 경로)
        prev_idx = (i - 1) % N
        next_idx = (i + 1) % N
        tangent = points[next_idx] - points[prev_idx]

        # 정규화
        length = np.linalg.norm(tangent)
        if length > 1e-6:
            tangent = tangent / length

        # 90도 회전 (왼쪽 법선)
        normals[i] = np.array([-tangent[1], tangent[0]])

    return normals


def _order_points(points: np.ndarray) -> np.ndarray:
    """
    점들을 가장 가까운 이웃 순서로 정렬 (그리디 TSP)
    points: (N, 2) 형태 (row, col)
    """
    N = len(points)
    if N == 0:
        return points

    visited = np.zeros(N, dtype=bool)
    order = [0]
    visited[0] = True

    for _ in range(N - 1):
        current = points[order[-1]]
        dists = np.linalg.norm(points - current, axis=1)
        dists[visited] = np.inf
        nearest = np.argmin(dists)
        order.append(nearest)
        visited[nearest] = True

    return points[order]


def _resample_by_distance(points: np.ndarray,
                           spacing: float) -> np.ndarray:
    """일정 간격으로 경로를 리샘플링"""
    if len(points) < 2:
        return points

    # 누적 거리 계산
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_dist = cum_dist[-1]

    if total_dist < spacing:
        return points

    # 일정 간격으로 보간
    n_points = int(total_dist / spacing)
    target_dists = np.linspace(0, total_dist, n_points, endpoint=False)

    resampled = np.zeros((n_points, 2))
    for i, d in enumerate(target_dists):
        idx = np.searchsorted(cum_dist, d, side='right') - 1
        idx = min(idx, len(points) - 2)

        seg_start = cum_dist[idx]
        seg_len   = seg_lengths[idx]
        if seg_len < 1e-9:
            resampled[i] = points[idx]
        else:
            t = (d - seg_start) / seg_len
            resampled[i] = points[idx] + t * (points[idx + 1] - points[idx])

    return resampled


# ══════════════════════════════════════════════════════════════════════════════
# 메인 (테스트용)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Waypoint Loader 테스트')
    parser.add_argument('--csv', type=str, default=None,
                        help='centerline CSV 파일 경로')
    parser.add_argument('--map', type=str, default=None,
                        help='맵 경로 (확장자 제외)')
    parser.add_argument('--map-ext', type=str, default='.png',
                        help='맵 이미지 확장자')
    parser.add_argument('--num-lines', type=int, default=DEFAULT_NUM_LINES,
                        help='생성할 후보 라인 수')
    parser.add_argument('--spacing', type=float, default=DEFAULT_LINE_SPACE,
                        help='라인 간 간격 (m)')
    args = parser.parse_args()

    wp = load_waypoints(
        centerline_path=args.csv,
        map_path=args.map,
        map_ext=args.map_ext,
        num_lines=args.num_lines,
        line_spacing=args.spacing,
    )

    print(f'\n{"═"*50}')
    print(f'Waypoint Loader 결과')
    print(f'{"═"*50}')
    print(f'centerline 점 수: {len(wp["centerline"])}')
    print(f'생성된 라인 수  : {len(wp["lines"])}')
    for i, (line, offset) in enumerate(zip(wp['lines'], wp['offsets'])):
        label = '중앙' if abs(offset) < 1e-6 else \
                f'안쪽 {abs(offset):.2f}m' if offset < 0 else \
                f'바깥 {offset:.2f}m'
        print(f'  라인 {i}: {label} ({len(line)} 점)')
    print(f'{"═"*50}\n')