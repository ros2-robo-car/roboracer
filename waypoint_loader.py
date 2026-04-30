"""
waypoint_loader.py

트랙 맵에서 centerline을 로드하고
여러 후보 레이싱 라인(웨이포인트 배열)을 생성하는 모듈.

지원하는 입력:
  1) centerline CSV 파일 (f1tenth_racetracks 형식)
     컬럼: [x_m, y_m, w_tr_right_m, w_tr_left_m]
  2) 맵 이미지 + yaml (centerline 자동 추출)

라인 생성 방식:
  - 기존 고정 line_spacing 방식 대신,
    각 waypoint의 w_right / w_left를 이용해 트랙 폭 비율 기반 라인 생성.
  - 예: num_lines=5, width_fraction=0.6
      line 0: 오른쪽 폭의 60%
      line 1: 오른쪽 폭의 30%
      line 2: centerline
      line 3: 왼쪽 폭의 30%
      line 4: 왼쪽 폭의 60%
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

from config import(
    LINE_CONFIG,
)
# ── 설정 ─────────────────────────────────────────────────────────────────────
DEFAULT_NUM_LINES = 5
DEFAULT_LINE_SPACE = 0.15       # 호환성 유지용. adaptive 방식에서는 직접 사용하지 않음.
DEFAULT_POINT_SPACE = 0.1
DEFAULT_WIDTH_FRACTION = 0.60   # 각 방향 트랙 폭의 몇 %까지 라인을 만들지


# ══════════════════════════════════════════════════════════════════════════════
# 1. Centerline 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_centerline_csv(csv_path: str) -> dict:
    """
    f1tenth_racetracks 형식 centerline CSV 로드.

    CSV 형식:
        [x_m, y_m, w_tr_right_m, w_tr_left_m]

    반환:
        {
            'centerline': np.ndarray (N, 2),
            'w_right':    np.ndarray (N,),
            'w_left':     np.ndarray (N,),
        }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'centerline 파일 없음: {csv_path}')

    data = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            if not row or row[0].startswith('#'):
                continue

            try:
                vals = [float(v) for v in row]
                data.append(vals)
            except ValueError:
                continue

    data = np.array(data, dtype=np.float32)

    if len(data) == 0:
        raise ValueError(f'CSV에 유효한 waypoint 데이터가 없습니다: {csv_path}')

    if data.shape[1] >= 4:
        return {
            'centerline': data[:, :2],
            'w_right': data[:, 2],
            'w_left': data[:, 3],
        }

    if data.shape[1] >= 2:
        # 폭 정보가 없는 경우 fallback
        return {
            'centerline': data[:, :2],
            'w_right': np.ones(len(data), dtype=np.float32) * 0.5,
            'w_left': np.ones(len(data), dtype=np.float32) * 0.5,
        }

    raise ValueError(f'CSV 형식 오류: 최소 2개 컬럼 필요, {data.shape[1]}개 발견')


def load_map_yaml(yaml_path: str) -> dict:
    """맵 yaml 파일에서 resolution, origin 읽기."""
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    return {
        'resolution': float(meta['resolution']),
        'origin_x': float(meta['origin'][0]),
        'origin_y': float(meta['origin'][1]),
        'image': meta.get('image', ''),
    }


def extract_centerline_from_map(
    map_path: str,
    map_ext: str = '.png',
    point_spacing: float = DEFAULT_POINT_SPACE,
) -> dict:
    """
    맵 이미지에서 centerline 자동 추출.
    EDT + skeletonize 방식.

    CSV가 없는 경우 fallback 용도.
    """
    if not HAS_CV2:
        raise ImportError('OpenCV 필요: pip install opencv-python')

    if not HAS_SKIMAGE:
        raise ImportError('scikit-image 필요: pip install scikit-image')

    yaml_path = map_path + '.yaml' if not map_path.endswith('.yaml') else map_path
    map_meta = load_map_yaml(yaml_path)

    resolution = map_meta['resolution']
    origin_x = map_meta['origin_x']
    origin_y = map_meta['origin_y']

    img_path = os.path.splitext(yaml_path)[0] + map_ext
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f'맵 이미지 없음: {img_path}')

    # 흰색 = 주행 가능 영역
    _, binary = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # ROS map convention에 맞게 상하 반전
    binary = np.flipud(binary)

    edt = distance_transform_edt(binary)

    threshold = np.max(edt) * 0.3
    edt_thresh = (edt > threshold).astype(np.uint8)

    skeleton = skeletonize(edt_thresh).astype(np.uint8)

    points_px = np.argwhere(skeleton > 0)

    if len(points_px) == 0:
        raise RuntimeError('centerline 추출 실패: 스켈레톤 점 없음')

    ordered = _order_points(points_px)

    world_points = np.zeros((len(ordered), 2), dtype=np.float32)

    for i, (row, col) in enumerate(ordered):
        world_points[i, 0] = col * resolution + origin_x
        world_points[i, 1] = row * resolution + origin_y

    world_points = _resample_by_distance(world_points, point_spacing)

    widths = []

    for row, col in ordered:
        if 0 <= row < edt.shape[0] and 0 <= col < edt.shape[1]:
            widths.append(edt[row, col] * resolution)
        else:
            widths.append(0.5)

    widths = np.array(widths, dtype=np.float32)

    if len(widths) != len(world_points):
        widths = np.interp(
            np.linspace(0, 1, len(world_points)),
            np.linspace(0, 1, len(widths)),
            widths,
        ).astype(np.float32)

    return {
        'centerline': world_points,
        'w_right': widths,
        'w_left': widths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. 후보 라인 생성
# ══════════════════════════════════════════════════════════════════════════════

def generate_racing_lines(
    centerline_data: dict,
    num_lines: int = DEFAULT_NUM_LINES,
    line_spacing: float = DEFAULT_LINE_SPACE,
    width_fraction: float = DEFAULT_WIDTH_FRACTION,
    smooth_width_window: int = 7,
) -> dict:
    center = centerline_data['centerline'].astype(np.float32)
    w_right = centerline_data['w_right'].astype(np.float32)
    w_left = centerline_data['w_left'].astype(np.float32)

    if len(center) < 3:
        raise ValueError('centerline 점이 너무 적습니다.')

    normals = _compute_normals(center)

    w_right = np.where(np.isfinite(w_right), w_right, 0.5)
    w_left = np.where(np.isfinite(w_left), w_left, 0.5)

    w_right = np.maximum(w_right, 0.05)
    w_left = np.maximum(w_left, 0.05)

    w_right = _smooth_circular(w_right, smooth_width_window)
    w_left = _smooth_circular(w_left, smooth_width_window)

    width_fraction = float(np.clip(width_fraction, 0.1, 0.9))

    # 차량 폭 고려
    vehicle_half_width = 0.31 / 2.0
    safety_margin = LINE_CONFIG.get('line_safety_margin', 0.10)
    min_usable_width = 0.05

    if num_lines <= 1:
        normalized_offsets = [0.0]
    else:
        normalized_offsets = np.linspace(1.0, -1.0, num_lines).tolist()

    lines = []
    valid_offsets = []

    for norm_offset in normalized_offsets:
        line = np.zeros_like(center)

        for i in range(len(center)):
            usable_right = max(
                float(w_right[i]) - vehicle_half_width - safety_margin,
                min_usable_width,
            )
            usable_left = max(
                float(w_left[i]) - vehicle_half_width - safety_margin,
                min_usable_width,
            )

            if norm_offset < 0:
                offset_m = norm_offset * width_fraction * usable_right
            elif norm_offset > 0:
                offset_m = norm_offset * width_fraction * usable_left
            else:
                offset_m = 0.0

            shifted = center[i] + offset_m * normals[i]
            line[i] = shifted

        lines.append(line)
        valid_offsets.append(norm_offset)

    return {
        'lines': lines,
        'centerline': center,
        'normals': normals,
        'offsets': valid_offsets,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 편의 함수
# ══════════════════════════════════════════════════════════════════════════════

def load_waypoints(
    centerline_path: str = None,
    map_path: str = None,
    map_ext: str = '.png',
    num_lines: int = DEFAULT_NUM_LINES,
    line_spacing: float = DEFAULT_LINE_SPACE,
    width_fraction: float = DEFAULT_WIDTH_FRACTION,
    smooth_width_window: int = 7,
) -> dict:
    """
    편의 함수: centerline 로드 + 후보 라인 생성을 한 번에 수행.

    CSV가 있으면 centerline CSV를 우선 사용하고,
    없으면 map image + yaml에서 centerline을 추출한다.
    """
    if centerline_path is not None:
        cl_data = load_centerline_csv(centerline_path)
    elif map_path is not None:
        cl_data = extract_centerline_from_map(map_path, map_ext)
    else:
        raise ValueError('centerline_path 또는 map_path 중 하나를 지정해야 합니다')

    return generate_racing_lines(
        centerline_data=cl_data,
        num_lines=num_lines,
        line_spacing=line_spacing,
        width_fraction=width_fraction,
        smooth_width_window=smooth_width_window,
    )


def get_nearest_waypoint_idx(
    position: np.ndarray,
    waypoints: np.ndarray,
) -> int:
    """현재 위치에서 가장 가까운 웨이포인트 인덱스 반환."""
    dists = np.linalg.norm(waypoints - position[:2], axis=1)
    return int(np.argmin(dists))


def get_lookahead_point(
    position: np.ndarray,
    waypoints: np.ndarray,
    lookahead_dist: float,
    nearest_idx: int = None,
) -> tuple:
    """
    현재 위치에서 lookahead 거리에 해당하는 웨이포인트 반환.

    Returns:
        (target_point, target_idx)
    """
    if nearest_idx is None:
        nearest_idx = get_nearest_waypoint_idx(position, waypoints)

    n = len(waypoints)

    for i in range(1, n):
        idx = (nearest_idx + i) % n
        dist = np.linalg.norm(waypoints[idx] - position[:2])

        if dist >= lookahead_dist:
            return waypoints[idx], idx

    last_idx = (nearest_idx + n - 1) % n
    return waypoints[last_idx], last_idx


# ══════════════════════════════════════════════════════════════════════════════
# 내부 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def _compute_normals(points: np.ndarray) -> np.ndarray:
    """
    경로의 각 점에서 법선 벡터 계산.
    접선 벡터를 90도 회전한 왼쪽 법선을 사용한다.
    """
    n = len(points)
    normals = np.zeros_like(points)

    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        tangent = points[next_idx] - points[prev_idx]

        length = np.linalg.norm(tangent)
        if length > 1e-6:
            tangent = tangent / length
        else:
            tangent = np.array([1.0, 0.0], dtype=np.float32)

        normals[i] = np.array([-tangent[1], tangent[0]], dtype=np.float32)

    return normals


def _smooth_circular(values: np.ndarray, window: int = 7) -> np.ndarray:
    """
    순환 경로용 moving average smoothing.
    트랙은 폐곡선이므로 앞뒤를 이어서 smoothing한다.
    """
    values = values.astype(np.float32)

    if window <= 1:
        return values

    if window % 2 == 0:
        window += 1

    if len(values) < window:
        return values

    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]])

    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(padded, kernel, mode='valid')

    return smoothed.astype(np.float32)


def _order_points(points: np.ndarray) -> np.ndarray:
    """
    점들을 가장 가까운 이웃 순서로 정렬.
    map image에서 centerline을 추출할 때 사용.
    """
    n = len(points)

    if n == 0:
        return points

    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = points[order[-1]]
        dists = np.linalg.norm(points - current, axis=1)
        dists[visited] = np.inf

        nearest = np.argmin(dists)
        order.append(nearest)
        visited[nearest] = True

    return points[order]


def _resample_by_distance(
    points: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """일정 간격으로 경로를 리샘플링."""
    if len(points) < 2:
        return points

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_dist = cum_dist[-1]

    if total_dist < spacing:
        return points

    n_points = int(total_dist / spacing)
    target_dists = np.linspace(0, total_dist, n_points, endpoint=False)

    resampled = np.zeros((n_points, 2), dtype=np.float32)

    for i, d in enumerate(target_dists):
        idx = np.searchsorted(cum_dist, d, side='right') - 1
        idx = min(idx, len(points) - 2)

        seg_start = cum_dist[idx]
        seg_len = seg_lengths[idx]

        if seg_len < 1e-9:
            resampled[i] = points[idx]
        else:
            t = (d - seg_start) / seg_len
            resampled[i] = points[idx] + t * (points[idx + 1] - points[idx])

    return resampled


# ══════════════════════════════════════════════════════════════════════════════
# 메인 테스트
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
                        help='기존 line_spacing 호환용')
    parser.add_argument('--width-fraction', type=float, default=DEFAULT_WIDTH_FRACTION,
                        help='트랙 반폭 중 후보 라인에 사용할 비율')
    args = parser.parse_args()

    wp = load_waypoints(
        centerline_path=args.csv,
        map_path=args.map,
        map_ext=args.map_ext,
        num_lines=args.num_lines,
        line_spacing=args.spacing,
        width_fraction=args.width_fraction,
    )

    print(f'\n{"═" * 50}')
    print('Waypoint Loader 결과')
    print(f'{"═" * 50}')
    print(f'centerline 점 수: {len(wp["centerline"])}')
    print(f'생성된 라인 수  : {len(wp["lines"])}')

    for i, (line, offset) in enumerate(zip(wp['lines'], wp['offsets'])):
        if abs(offset) < 1e-6:
            label = '중앙'
        elif offset < 0:
            label = f'오른쪽 폭의 {abs(offset) * args.width_fraction * 100:.1f}%'
        else:
            label = f'왼쪽 폭의 {offset * args.width_fraction * 100:.1f}%'

        print(f'  라인 {i}: {label} ({len(line)} 점)')

    print(f'{"═" * 50}\n')