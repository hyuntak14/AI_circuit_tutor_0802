import cv2
import numpy as np
import math
import os
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Using simple clustering instead.")


def find_additional_leads_from_lines(body_lines, selected_leads, centroid, hull, 
                                   min_distance_from_body=30, max_total_leads=2):
    """
    이미 선택된 리드가 부족할 때, Body Lines의 끝점에서 추가 리드 찾기
    """
    if len(selected_leads) >= max_total_leads:
        return selected_leads
    
    # 이미 선택된 리드의 위치들
    existing_positions = [lead['hole_pos'] for lead in selected_leads]
    
    # Body Lines의 먼 끝점들을 후보로 수집
    line_endpoints = []
    for line_info in body_lines:
        far_point = line_info['far_point']
        fx, fy = far_point
        
        # 이미 선택된 위치와 너무 가깝지 않은지 확인
        too_close_to_existing = False
        for ex, ey in existing_positions:
            if math.hypot(fx - ex, fy - ey) < 20:  # 20픽셀 이내면 중복
                too_close_to_existing = True
                break
        
        if too_close_to_existing:
            continue
        
        # LED 몸체와의 거리 확인
        if hull is not None:
            body_distance = abs(cv2.pointPolygonTest(hull, (float(fx), float(fy)), True))
            if body_distance < min_distance_from_body:
                continue
        
        # 중심에서의 거리
        distance_from_center = math.hypot(fx - centroid[0], fy - centroid[1])
        
        line_endpoints.append({
            'position': (fx, fy),
            'distance_from_center': distance_from_center,
            'line_info': line_info,
            'source': 'line_endpoint'  # 구멍이 아닌 선 끝점임을 표시
        })
    
    # 거리 순으로 정렬 (멀수록 우선)
    line_endpoints.sort(key=lambda x: x['distance_from_center'], reverse=True)
    
    # 필요한 만큼 추가
    additional_leads = []
    needed = max_total_leads - len(selected_leads)
    
    for i in range(min(needed, len(line_endpoints))):
        endpoint = line_endpoints[i]
        additional_leads.append({
            'hole_pos': endpoint['position'],  # 호환성을 위해 같은 키 사용
            'line_info': endpoint['line_info'],
            'distance_to_endpoint': 0,  # 끝점 자체이므로 0
            'score': endpoint['distance_from_center'] * 2,  # 점수 계산
            'source': 'line_endpoint'
        })
    
    # 대칭성 고려하여 최종 선택
    if len(selected_leads) == 1 and len(additional_leads) > 0:
        # 기존 1개와 가장 대칭적인 것 선택
        best_symmetric = find_best_symmetric_endpoint(selected_leads[0], additional_leads, centroid)
        if best_symmetric:
            return selected_leads + [best_symmetric]
    
    return selected_leads + additional_leads

def find_best_symmetric_endpoint(existing_lead, candidates, centroid):
    """기존 리드와 가장 대칭적인 끝점 찾기"""
    if not candidates or not centroid:
        return candidates[0] if candidates else None
    
    ex, ey = existing_lead['hole_pos']
    cx, cy = centroid
    
    # 기존 리드의 각도
    existing_angle = math.atan2(ey - cy, ex - cx)
    existing_dist = math.hypot(ex - cx, ey - cy)
    
    best_candidate = None
    best_score = float('-inf')
    
    for candidate in candidates:
        px, py = candidate['hole_pos']
        
        # 후보의 각도와 거리
        candidate_angle = math.atan2(py - cy, px - cx)
        candidate_dist = math.hypot(px - cx, py - cy)
        
        # 대칭성 점수 계산
        # 180도 차이에 가까울수록 높은 점수
        angle_diff = abs(abs(existing_angle - candidate_angle) - math.pi)
        angle_score = 1.0 / (1.0 + angle_diff * 10)
        
        # 거리 유사성
        dist_diff = abs(existing_dist - candidate_dist)
        dist_score = 1.0 / (1.0 + dist_diff * 0.1)
        
        total_score = angle_score * 100 + dist_score * 50 + candidate['score'] * 0.1
        
        if total_score > best_score:
            best_score = total_score
            best_candidate = candidate
    
    return best_candidate


# --- 간단한 DBSCAN 대체 함수 ---
def simple_dbscan(points, eps, min_samples):
    """sklearn이 없을 때 사용하는 간단한 클러스터링"""
    if len(points) == 0:
        return np.array([])
    
    points = np.array(points).reshape(-1, 1)
    labels = np.full(len(points), -1)
    cluster_id = 0
    
    for i, point in enumerate(points):
        if labels[i] != -1:
            continue
            
        neighbors = []
        for j, other_point in enumerate(points):
            if abs(point[0] - other_point[0]) <= eps:
                neighbors.append(j)
        
        if len(neighbors) >= min_samples:
            for neighbor_idx in neighbors:
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
            cluster_id += 1
    
    return type('', (), {'labels_': labels})()


def detect_capacitor_body(img, min_hull_area=350):
    """
    Capacitor body detection using Otsu thresholding and convex hull.

    Args:
        img (np.ndarray): 입력 BGR 이미지
        min_hull_area (int): convex hull 최소 면적 (픽셀)

    Returns:
        merged (np.ndarray or None): 통합 convex hull (Nx1x2 배열)
        centroid (tuple or None): hull의 중심좌표 (x, y)
        mask_clean (np.ndarray): hull 검출에 사용된 이진 마스크
    """
    # 1. Gray 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. Otsu 이진화
    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3. Invert: 어두운(검은색) 영역 검출
    mask_inv = cv2.bitwise_not(mask_otsu)
    # 4. Morphology: 노이즈 제거 및 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=2)
    # 5. Contours
    contours, _ = safe_find_contours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask_clean
    # 6. Select hulls above area threshold
    kept = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull) >= min_hull_area:
            kept.append(hull)
    if not kept:
        return None, None, mask_clean
    # 7. Merge hulls and compute final hull
    all_pts = np.vstack(kept)
    merged = cv2.convexHull(all_pts)
    # 8. Centroid 계산
    M = cv2.moments(merged)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
    else:
        centroid = None
    return merged, centroid, mask_clean



# --- 구멍 검출 관련 함수들 ---
def safe_find_contours(image, mode, method):
    """OpenCV 버전에 상관없이 안전하게 contours를 찾는 함수"""
    result = cv2.findContours(image, mode, method)
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result
    return contours, hierarchy

def remove_red_blue(img):
    """빨간색과 파란색을 제거하는 함수"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lr1 = np.array([0, 100, 100], dtype=np.uint8)
    ur1 = np.array([10, 255, 255], dtype=np.uint8)
    lr2 = np.array([160, 100, 100], dtype=np.uint8)
    ur2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lr1, ur1),
        cv2.inRange(hsv, lr2, ur2)
    )
    lb = np.array([100, 150, 50], dtype=np.uint8)
    ub = np.array([140, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lb, ub)
    mask_rb = cv2.bitwise_or(mask_red, mask_blue)
    clean = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_rb))
    return clean

def apply_clahe(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    """CLAHE 적용 함수"""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)

def detect_square_holes(binary_mask, min_area=50, max_area=2000, 
                       min_circularity=0.3, max_circularity=0.9,
                       min_square_ratio=0.6, max_square_ratio=1.4):
    """정사각형 형태의 구멍을 검출하는 함수"""
    contours, _ = safe_find_contours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_centers = []
    hole_info = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity or circularity > max_circularity:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect_ratio = w / h
        if aspect_ratio < min_square_ratio or aspect_ratio > max_square_ratio:
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
        if convexity < 0.7:
            continue
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            hole_centers.append((cx, cy))
            hole_info.append({
                'center': (cx, cy),
                'area': area,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'convexity': convexity,
                'bbox': (x, y, w, h),
                'contour': contour
            })
    
    return hole_centers, hole_info

# --- 보간 관련 함수들 ---
def remove_duplicate_points(points, threshold=10):
    """중복되거나 너무 가까운 점들 제거"""
    if not points:
        return points
    
    unique_points = []
    for point in points:
        is_duplicate = False
        for existing in unique_points:
            if np.hypot(point[0] - existing[0], point[1] - existing[1]) < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    
    return unique_points

def cluster_rows_columns(hole_centers, row_eps=15, col_eps=15, min_samples=2):
    """전체 좌표에서 행/열 기준점들을 찾고 격자 정보를 생성"""
    if not hole_centers:
        return [], [], {}, 0, 0
    
    points = np.array(hole_centers)
    
    # Y좌표 기준으로 행 클러스터링
    y_coords = points[:, 1].reshape(-1, 1)
    if SKLEARN_AVAILABLE:
        row_clustering = DBSCAN(eps=row_eps, min_samples=min_samples).fit(y_coords)
    else:
        row_clustering = simple_dbscan(y_coords, row_eps, min_samples)
    
    # X좌표 기준으로 열 클러스터링
    x_coords = points[:, 0].reshape(-1, 1)
    if SKLEARN_AVAILABLE:
        col_clustering = DBSCAN(eps=col_eps, min_samples=min_samples).fit(x_coords)
    else:
        col_clustering = simple_dbscan(x_coords, col_eps, min_samples)
    
    # 각 행의 기준 Y좌표 계산
    row_y_centers = []
    row_clusters = {}
    for i, label in enumerate(row_clustering.labels_):
        if label >= 0:
            if label not in row_clusters:
                row_clusters[label] = []
            row_clusters[label].append(points[i])
    
    for row_id, points_in_row in row_clusters.items():
        y_center = np.mean([p[1] for p in points_in_row])
        row_y_centers.append(y_center)
    row_y_centers = sorted(row_y_centers)
    
    # 각 열의 기준 X좌표 계산
    col_x_centers = []
    col_clusters = {}
    for i, label in enumerate(col_clustering.labels_):
        if label >= 0:
            if label not in col_clusters:
                col_clusters[label] = []
            col_clusters[label].append(points[i])
    
    for col_id, points_in_col in col_clusters.items():
        x_center = np.mean([p[0] for p in points_in_col])
        col_x_centers.append(x_center)
    col_x_centers = sorted(col_x_centers)
    
    # 현재 존재하는 구멍들의 격자 위치 맵핑
    grid_map = {}
    for cx, cy in hole_centers:
        closest_row_idx = min(range(len(row_y_centers)), 
                             key=lambda i: abs(row_y_centers[i] - cy))
        closest_col_idx = min(range(len(col_x_centers)), 
                             key=lambda i: abs(col_x_centers[i] - cx))
        
        if (abs(row_y_centers[closest_row_idx] - cy) <= row_eps and 
            abs(col_x_centers[closest_col_idx] - cx) <= col_eps):
            grid_map[(closest_row_idx, closest_col_idx)] = (cx, cy)
    
    return row_y_centers, col_x_centers, grid_map, len(row_y_centers), len(col_x_centers)

def grid_based_interpolation(row_y_centers, col_x_centers, grid_map, num_rows, num_cols):
    """격자 기반 보간: 모든 격자 교차점에서 누락된 구멍을 찾아 보간"""
    interpolated_points = []
    
    if not row_y_centers or not col_x_centers:
        return interpolated_points
    
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if (row_idx, col_idx) not in grid_map:
                x_coord = col_x_centers[col_idx]
                y_coord = row_y_centers[row_idx]
                interpolated_points.append((x_coord, y_coord))
    
    return interpolated_points

# --- 선 검출 관련 함수들 ---
def detect_lines_lsd(img_gray, min_length=10):
    """LSD를 사용하여 선을 검출하고 최소 길이 필터링"""
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(img_gray)
    if lines is None:
        return []
    
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            filtered_lines.append((int(x1), int(y1), int(x2), int(y2)))
    
    return filtered_lines

# --- LED 몸체 검출 함수 ---
def detect_led_body(img):
    """LED 몸체 검출 (빨간색, 노란색, 초록색 기준)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 빨간색 범위
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # 노란색 범위
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    
    # 초록색 범위
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    
    # 모든 색상 마스크 합치기
    body_mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(yellow_mask, green_mask))
    
    # 노이즈 제거 및 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 컨투어 찾기
    contours, _ = safe_find_contours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    # 가장 큰 컨투어 선택
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convex hull 계산
    hull = cv2.convexHull(largest_contour)
    
    # 중심점 계산
    M = cv2.moments(hull)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
    else:
        centroid = None
    
    return hull, centroid, body_mask

import numpy as np
import math
from collections import deque

def merge_lines(lines, angle_thresh_deg=10, dist_thresh=20):
    """
    비슷한 각도와 가까운 거리에 있는 선분들을 병합합니다.

    Args:
        lines (list): [(x1, y1, x2, y2), ...] 형태의 선분 리스트.
        angle_thresh_deg (float): 두 선분이 평행하다고 판단할 최대 각도 차이 (도).
        dist_thresh (float): 두 선분의 끝점이 가깝다고 판단할 최대 거리.

    Returns:
        list: 병합된 선분들의 리스트.
    """
    if not lines or len(lines) < 2:
        return lines

    num_lines = len(lines)
    
    # 각 선분의 속성(각도, 단위벡터) 미리 계산
    line_props = []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = math.atan2(y2 - y1, x2 - x1)
        vec = np.array([x2 - x1, y2 - y1])
        norm = np.linalg.norm(vec)
        unit_vec = vec / norm if norm > 0 else np.array([0, 0])
        line_props.append({'angle': angle, 'unit_vec': unit_vec, 'endpoints': [(x1, y1), (x2, y2)]})

    # 인접 리스트로 그래프 구성
    adj = [[] for _ in range(num_lines)]
    angle_thresh_rad = math.radians(angle_thresh_deg)

    for i in range(num_lines):
        for j in range(i + 1, num_lines):
            # 1. 각도 유사성 검사
            # 단위 벡터의 내적을 사용하여 방향에 무관한 평행성 검사
            dot_product = abs(np.dot(line_props[i]['unit_vec'], line_props[j]['unit_vec']))
            if dot_product < math.cos(angle_thresh_rad):
                continue

            # 2. 거리 근접성 검사
            p1, p2 = line_props[i]['endpoints']
            p3, p4 = line_props[j]['endpoints']
            
            min_dist = min(
                math.hypot(p1[0] - p3[0], p1[1] - p3[1]),
                math.hypot(p1[0] - p4[0], p1[1] - p4[1]),
                math.hypot(p2[0] - p3[0], p2[1] - p3[1]),
                math.hypot(p2[0] - p4[0], p2[1] - p4[1])
            )

            if min_dist < dist_thresh:
                adj[i].append(j)
                adj[j].append(i)

    # BFS/DFS를 사용하여 연결된 그룹(클러스터) 찾기
    visited = [False] * num_lines
    groups = []
    for i in range(num_lines):
        if not visited[i]:
            current_group = []
            q = deque([i])
            visited[i] = True
            while q:
                u = q.popleft()
                current_group.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            groups.append(current_group)

    # 각 그룹을 하나의 선분으로 병합
    merged_lines = []
    for group in groups:
        if len(group) == 1:
            merged_lines.append(lines[group[0]])
            continue

        # 그룹 내 모든 끝점 수집
        points = []
        for line_idx in group:
            points.extend(line_props[line_idx]['endpoints'])

        # 가장 멀리 떨어진 두 끝점 찾기
        max_dist_sq = -1
        best_pair = None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist_sq = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                    best_pair = (points[i], points[j])
        
        p_start, p_end = best_pair
        merged_lines.append((int(p_start[0]), int(p_start[1]), int(p_end[0]), int(p_end[1])))

    return merged_lines


# ========== 새로운 LED 리드 검출 함수들 ==========

def detect_lines_to_body(lines, centroid, hull, direction_threshold=0.6, 
                        body_proximity_threshold=30, min_line_length=20):
    """
    LED 몸체로 향하는 선들을 검출합니다.
    LED 몸체에서 멀리 있는 선들을 우선적으로 선택합니다.
    """
    if not lines or not centroid:
        return []
    
    body_lines = []
    cx, cy = centroid
    
    for line in lines:
        x1, y1, x2, y2 = line
        
        # 선의 길이 체크
        line_length = math.hypot(x2 - x1, y2 - y1)
        if line_length < min_line_length:
            continue
        
        # 각 끝점에서 LED 중심까지의 거리
        dist1 = math.hypot(x1 - cx, y1 - cy)
        dist2 = math.hypot(x2 - cx, y2 - cy)
        
        # LED 몸체와의 거리 계산 (hull이 있을 경우)
        body_dist1 = float('inf')
        body_dist2 = float('inf')
        min_body_distance = float('inf')
        
        if hull is not None:
            body_dist1 = abs(cv2.pointPolygonTest(hull, (float(x1), float(y1)), True))
            body_dist2 = abs(cv2.pointPolygonTest(hull, (float(x2), float(y2)), True))
            
            # 선 전체에서 LED 몸체까지의 최소 거리 계산
            # 선을 따라 여러 점을 샘플링하여 체크
            num_samples = int(line_length / 5) + 1
            for i in range(num_samples + 1):
                t = i / num_samples
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                dist_to_body = abs(cv2.pointPolygonTest(hull, (float(px), float(py)), True))
                min_body_distance = min(min_body_distance, dist_to_body)
        
        # 선의 방향 벡터
        line_vec = np.array([x2 - x1, y2 - y1])
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        
        # 각 끝점에서 중심으로의 방향 벡터
        to_center1 = np.array([cx - x1, cy - y1])
        to_center2 = np.array([cx - x2, cy - y2])
        
        to_center1_norm = to_center1 / np.linalg.norm(to_center1) if np.linalg.norm(to_center1) > 0 else np.zeros(2)
        to_center2_norm = to_center2 / np.linalg.norm(to_center2) if np.linalg.norm(to_center2) > 0 else np.zeros(2)
        
        # 방향성 체크: 선이 중심을 향하고 있는지
        dot1 = np.dot(line_vec_norm, to_center1_norm)
        dot2 = np.dot(-line_vec_norm, to_center2_norm)
        
        # 조건 체크
        is_pointing_to_center = max(dot1, dot2) > direction_threshold
        is_near_body = min(body_dist1, body_dist2) < body_proximity_threshold or min(dist1, dist2) < body_proximity_threshold * 1.5
        
        if is_pointing_to_center and is_near_body:
            # 중심에서 더 가까운 점과 더 먼 점 결정
            if dist1 < dist2:
                near_point = (x1, y1)
                far_point = (x2, y2)
                far_distance = dist2
                far_body_distance = body_dist2
            else:
                near_point = (x2, y2)
                far_point = (x1, y1)
                far_distance = dist1
                far_body_distance = body_dist1
            
            body_lines.append({
                'line': line,
                'near_point': near_point,
                'far_point': far_point,
                'far_distance': far_distance,
                'far_body_distance': far_body_distance,  # LED 몸체에서 먼 끝점의 거리
                'min_body_distance': min_body_distance,  # 선 전체의 최소 몸체 거리
                'length': line_length,
                'direction_score': max(dot1, dot2)
            })
    
    # LED 몸체에서 가장 멀리 있는 선들을 우선적으로 선택
    # 1차: 먼 끝점이 몸체에서 멀수록 우선
    # 2차: 선 전체가 몸체에서 멀수록 우선
    # 3차: 중심에서 멀수록 우선
    body_lines.sort(key=lambda x: (
        x['far_body_distance'],      # LED 몸체에서 먼 끝점의 거리 (클수록 좋음)
        x['min_body_distance'],      # 선 전체의 최소 몸체 거리 (클수록 좋음)
        x['far_distance']            # 중심에서의 거리 (클수록 좋음)
    ), reverse=True)
    
    return body_lines

def select_best_body_lines(body_lines, max_lines=2, min_body_distance=20):
    """LED 몸체에서 충분히 멀리 있는 최상위 선들만 선택"""
    selected_lines = []
    
    for line_info in body_lines:
        if line_info['far_body_distance'] >= min_body_distance:
            selected_lines.append(line_info)
            
            if len(selected_lines) >= max_lines:
                break
    
    return selected_lines


def find_holes_near_line_endpoints(body_lines, hole_centers, hull, max_distance=25, body_exclusion_distance=50):
    """
    LED 몸체로 향하는 선들의 먼 쪽 끝점 근처에 있는 구멍들을 찾습니다.
    
    Args:
        body_lines: detect_lines_to_body에서 반환된 선들의 정보
        hole_centers: 검출된 구멍들의 중심점 리스트
        hull: LED 몸체의 convex hull
        max_distance: 끝점에서 구멍까지의 최대 거리
        body_exclusion_distance: LED 몸체 경계에서 제외할 거리
    
    Returns:
        tuple: (리드 후보들의 정보, 제외된 구멍들의 정보)
    """
    lead_candidates = []
    excluded_holes = []
    
    for line_info in body_lines:
        far_point = line_info['far_point']
        fx, fy = far_point
        
        # 이 끝점 근처의 구멍들 찾기
        nearby_holes = []
        for hole_pos in hole_centers:
            hx, hy = hole_pos
            distance = math.hypot(fx - hx, fy - hy)
            
            # LED 몸체와의 거리 체크 (제외 조건)
            if hull is not None:
                body_distance = abs(cv2.pointPolygonTest(hull, (float(hx), float(hy)), True))
                if body_distance < body_exclusion_distance:
                    excluded_holes.append({
                        'hole_pos': hole_pos,
                        'body_distance': body_distance,
                        'reason': 'too_close_to_body'
                    })
                    continue
            
            if distance <= max_distance:
                nearby_holes.append({
                    'hole_pos': hole_pos,
                    'distance_to_endpoint': distance,
                    'line_info': line_info
                })
        
        # 이 선에 대해 가장 가까운 구멍 선택
        if nearby_holes:
            nearby_holes.sort(key=lambda x: x['distance_to_endpoint'])
            best_hole = nearby_holes[0]
            
            # 점수 계산
            score = (line_info['length'] * 2 +  # 선의 길이
                    line_info['direction_score'] * 100 +  # 방향성
                    (max_distance - best_hole['distance_to_endpoint']) * 3 +  # 근접성
                    line_info['far_distance'] * 0.5)  # 중심에서의 거리
            
            lead_candidates.append({
                'hole_pos': best_hole['hole_pos'],
                'line_info': line_info,
                'distance_to_endpoint': best_hole['distance_to_endpoint'],
                'score': score
            })
    
    # 점수 순으로 정렬
    lead_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return lead_candidates, excluded_holes

def select_best_symmetric_leads(lead_candidates, centroid, max_leads=2):
    """
    리드 후보들 중에서 대칭성을 고려하여 최적의 리드들을 선택합니다.
    
    Args:
        lead_candidates: find_holes_near_line_endpoints에서 반환된 후보들
        centroid: LED 중심점
        max_leads: 선택할 최대 리드 개수
    
    Returns:
        list: 선택된 리드들
    """
    if not lead_candidates or not centroid:
        return lead_candidates[:max_leads]
    
    if len(lead_candidates) < 2:
        return lead_candidates
    
    cx, cy = centroid
    best_pair = None
    best_score = float('-inf')
    
    # 모든 쌍에 대해 대칭성 검사
    for i in range(len(lead_candidates)):
        for j in range(i + 1, len(lead_candidates)):
            candidate1 = lead_candidates[i]
            candidate2 = lead_candidates[j]
            
            h1x, h1y = candidate1['hole_pos']
            h2x, h2y = candidate2['hole_pos']
            
            # 대칭성 계산
            # 1. 중심에서의 거리 유사성
            dist1 = math.hypot(h1x - cx, h1y - cy)
            dist2 = math.hypot(h2x - cx, h2y - cy)
            distance_similarity = 1.0 / (1.0 + abs(dist1 - dist2))
            
            # 2. 각도 대칭성 (180도에 가까운지)
            angle1 = math.atan2(h1y - cy, h1x - cx)
            angle2 = math.atan2(h2y - cy, h2x - cx)
            angle_diff = abs(abs(angle1 - angle2) - math.pi)
            angle_symmetry = 1.0 / (1.0 + angle_diff)
            
            # 3. 개별 점수 합계
            individual_scores = candidate1['score'] + candidate2['score']
            
            # 종합 점수
            symmetry_score = (distance_similarity * 50 + 
                            angle_symmetry * 100 + 
                            individual_scores * 0.1)
            
            if symmetry_score > best_score:
                best_score = symmetry_score
                best_pair = [candidate1, candidate2]
    
    return best_pair if best_pair else lead_candidates[:max_leads]

# --- 시각화 함수 ---
def visualize_new_results(img, hole_centers, interpolated_points, all_lines, merged_lines,
                         hull, centroid, body_lines, lead_candidates, selected_leads, excluded_holes):
    """새로운 방식의 결과 시각화"""
    vis = img.copy()
    
    # LED 몸체 그리기
    if hull is not None:
        cv2.drawContours(vis, [hull], -1, (0, 255, 255), 2)
        cv2.putText(vis, "LED Body", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if centroid is not None:
        cv2.circle(vis, centroid, 8, (0, 255, 255), -1)
    
    # 모든 검출된 구멍 그리기 (회색)
    for cx, cy in hole_centers:
        cv2.circle(vis, (int(cx), int(cy)), 6, (128, 128, 128), 1)
        cv2.circle(vis, (int(cx), int(cy)), 2, (128, 128, 128), -1)
    
    # 보간된 구멍 그리기 (보라색)
    for cx, cy in interpolated_points:
        cv2.circle(vis, (int(cx), int(cy)), 6, (255, 0, 255), 1)
        cv2.circle(vis, (int(cx), int(cy)), 2, (255, 0, 255), -1)
    
    # 제외된 구멍들 표시 (주황색 X)
    for excluded_info in excluded_holes:
        cx, cy = excluded_info['hole_pos']
        cx, cy = int(cx), int(cy)
        # X 표시
        cv2.line(vis, (cx-8, cy-8), (cx+8, cy+8), (0, 165, 255), 2)
        cv2.line(vis, (cx-8, cy+8), (cx+8, cy-8), (0, 165, 255), 2)
        cv2.circle(vis, (cx, cy), 10, (0, 165, 255), 1)
    
    # 모든 선 그리기 (연한 색)
    for x1, y1, x2, y2 in merged_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)
    
    # LED 몸체로 향하는 선들 강조 (초록색)
    for line_info in body_lines:
        x1, y1, x2, y2 = line_info['line']
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 먼 쪽 끝점 표시
        far_point = line_info['far_point']
        cv2.circle(vis, (int(far_point[0]), int(far_point[1])), 5, (0, 255, 0), -1)
    
    # 리드 후보들 표시 (노란색)
    for i, candidate in enumerate(lead_candidates):
        if candidate not in selected_leads:
            cx, cy = candidate['hole_pos']
            cx, cy = int(cx), int(cy)
            score = candidate['score']
            
            cv2.circle(vis, (cx, cy), 10, (0, 255, 255), 2)
            cv2.putText(vis, f"{score:.0f}", (cx + 12, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # 최종 선택된 LED 리드 강조 (빨간색)
    for i, lead in enumerate(selected_leads):
        cx, cy = lead['hole_pos']
        cx, cy = int(cx), int(cy)
        score = lead['score']
        
        # 구멍 강조
        cv2.circle(vis, (cx, cy), 15, (0, 0, 255), 3)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        
        # 연결된 선 강조
        line_info = lead['line_info']
        x1, y1, x2, y2 = line_info['line']
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # LED 번호와 정보 표시
        cv2.putText(vis, f"LED {i+1}", (cx + 20, cy - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, f"Score: {score:.0f}", (cx + 20, cy + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 중심과의 연결선 그리기
        if centroid is not None:
            cv2.line(vis, centroid, (cx, cy), (255, 0, 0), 2, cv2.LINE_AA)
    
    # 정보 표시
    info_y = vis.shape[0] - 160
    cv2.putText(vis, f"Total Holes: {len(hole_centers)} (gray)", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    cv2.putText(vis, f"Interpolated: {len(interpolated_points)} (purple)", (10, info_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(vis, f"Excluded: {len(excluded_holes)} (orange X)", (10, info_y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(vis, f"Total Lines: {len(merged_lines)}", (10, info_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(vis, f"Body Lines: {len(body_lines)} (green)", (10, info_y + 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, f"Candidates: {len(lead_candidates)} (yellow)", (10, info_y + 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"Selected LEDs: {len(selected_leads)} (red)", (10, info_y + 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 대칭성 정보 표시
    if len(selected_leads) == 2 and centroid is not None:
        cx, cy = centroid
        h1x, h1y = selected_leads[0]['hole_pos']
        h2x, h2y = selected_leads[1]['hole_pos']
        
        dist1 = math.hypot(h1x - cx, h1y - cy)
        dist2 = math.hypot(h2x - cx, h2y - cy)
        angle1 = math.degrees(math.atan2(h1y - cy, h1x - cx))
        angle2 = math.degrees(math.atan2(h2y - cy, h2x - cx))
        angle_diff = abs((angle2 - angle1 + 180) % 360 - 180)
        
        cv2.putText(vis, f"Symmetry: d1={dist1:.1f}, d2={dist2:.1f}, angle={angle_diff:.1f}°", 
                   (10, info_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis

# --- 메인 함수 ---
def main():
    # LED 이미지 찾기
    images = [f for f in os.listdir('.') if 'cap' in f.lower() and 
              f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print('No LED images found in current directory.')
        return
    
    print(f"Found {len(images)} LED images")
    idx = 0
    
    # 첫 번째 이미지 로드
    img = cv2.imread(images[idx])
    if img is None:
        print(f"Failed to load image: {images[idx]}")
        return
    
    # 창 생성
    windows = ['Final Result', 'Holes & Interpolation', 'Lines Detected', 'LED Body']
    for w in windows:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    
    # 컨트롤러 창
    cv2.namedWindow('Controller', cv2.WINDOW_NORMAL)
    
    # 트랙바 생성
    cv2.createTrackbar('Min Length', 'Controller', 30, 100, lambda x: None)
    cv2.createTrackbar('CLAHE Clip x10', 'Controller', 20, 100, lambda x: None)
    cv2.createTrackbar('Min Hole Area', 'Controller', 50, 500, lambda x: None)
    cv2.createTrackbar('Max Hole Area', 'Controller', 500, 2000, lambda x: None)
    cv2.createTrackbar('Direction Threshold x10', 'Controller', 6, 10, lambda x: None)
    cv2.createTrackbar('Body Proximity', 'Controller', 30, 100, lambda x: None)
    cv2.createTrackbar('Hole Distance', 'Controller', 25, 50, lambda x: None)
    cv2.createTrackbar('LED Body Exclusion', 'Controller', 50, 150, lambda x: None)
    cv2.createTrackbar('Use Otsu', 'Controller', 1, 1, lambda x: None)
    cv2.createTrackbar('Enable Interpolation', 'Controller', 1, 1, lambda x: None)
    cv2.createTrackbar('Row/Col Eps', 'Controller', 15, 50, lambda x: None)
    cv2.createTrackbar('Use Symmetry', 'Controller', 1, 1, lambda x: None)
    
    print("\nControls:")
    print("- 'n': Next image")
    print("- 'p': Previous image")
    print("- 's': Save result")
    print("- 'q' or ESC: Quit")
    print("\nNew Detection Logic:")
    print("- Detect lines pointing to LED body")
    print("- Exclude holes near LED body boundary")
    print("- Find holes near far endpoints of those lines")
    print("- Select best symmetric pair")
    
    while True:
        # 트랙바 값 읽기
        min_length = cv2.getTrackbarPos('Min Length', 'Controller')
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip x10', 'Controller') / 10.0
        min_hole_area = cv2.getTrackbarPos('Min Hole Area', 'Controller')
        max_hole_area = cv2.getTrackbarPos('Max Hole Area', 'Controller')
        direction_threshold = cv2.getTrackbarPos('Direction Threshold x10', 'Controller') / 10.0
        body_proximity = cv2.getTrackbarPos('Body Proximity', 'Controller')
        hole_distance = cv2.getTrackbarPos('Hole Distance', 'Controller')
        led_body_exclusion = cv2.getTrackbarPos('LED Body Exclusion', 'Controller')
        use_otsu = cv2.getTrackbarPos('Use Otsu', 'Controller') == 1
        enable_interpolation = cv2.getTrackbarPos('Enable Interpolation', 'Controller') == 1
        eps = cv2.getTrackbarPos('Row/Col Eps', 'Controller')
        use_symmetry = cv2.getTrackbarPos('Use Symmetry', 'Controller') == 1
        
        try:
            # 1. 전처리
            img_clean = remove_red_blue(img)
            gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
            processed = apply_clahe(gray, clipLimit=clahe_clip)
            
            # 2. 구멍 검출
            if use_otsu:
                _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                binary = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                             cv2.THRESH_BINARY, 15, 5)
            
            mask_inv = cv2.bitwise_not(binary)
            hole_centers, hole_info = detect_square_holes(mask_inv, min_hole_area, max_hole_area)
            
            # 2-1. 구멍 보간
            interpolated_points = []
            if enable_interpolation and hole_centers:
                row_y_centers, col_x_centers, grid_map, num_rows, num_cols = cluster_rows_columns(
                    hole_centers, eps, eps, 2)
                
                if row_y_centers and col_x_centers:
                    interpolated_points = grid_based_interpolation(
                        row_y_centers, col_x_centers, grid_map, num_rows, num_cols)
                    interpolated_points = remove_duplicate_points(interpolated_points, 10)
                    
                    # 기존 구멍과 너무 가까운 보간점 제거
                    filtered_interpolated = []
                    for interp_point in interpolated_points:
                        too_close = False
                        for existing_point in hole_centers:
                            if np.hypot(interp_point[0] - existing_point[0], 
                                      interp_point[1] - existing_point[1]) < 10:
                                too_close = True
                                break
                        if not too_close:
                            filtered_interpolated.append(interp_point)
                    interpolated_points = filtered_interpolated
            
            # 모든 구멍 (검출 + 보간)
            all_holes = hole_centers + interpolated_points
            
            # 3. LED 몸체 검출
            hull, centroid, body_mask = detect_capacitor_body(img)
            
            # 4. 직선 검출
            lines_gray = detect_lines_lsd(processed, min_length)
            lines_binary = detect_lines_lsd(binary, min_length)
            all_lines = lines_gray + lines_binary
            merged_lines = merge_lines(all_lines, angle_thresh_deg=10, dist_thresh=20)
            
            # 5. 새로운 LED 리드 검출 로직
            # 5-1. LED 몸체로 향하는 선들 검출
            body_lines = detect_lines_to_body(merged_lines, centroid, hull, 
                                            direction_threshold, body_proximity, min_length)
            
            body_lines = select_best_body_lines(body_lines, max_lines=2, min_body_distance=30)


            # 5-2. 선의 끝점 근처 구멍들 찾기
            lead_candidates, excluded_holes = find_holes_near_line_endpoints(
                body_lines, all_holes, hull, hole_distance, led_body_exclusion)
            
            # 5-3. 최적의 대칭 쌍 선택
            if use_symmetry:
                selected_leads = select_best_symmetric_leads(lead_candidates, centroid, 2)
            else:
                selected_leads = lead_candidates[:2]
            

            if len(selected_leads) < 2 and body_lines:
                print(f"\nOnly {len(selected_leads)} leads found, searching line endpoints...")
                selected_leads = find_additional_leads_from_lines(
                    body_lines, selected_leads, centroid, hull,
                    min_distance_from_body=led_body_exclusion
                )

            # 시각화
            # 구멍 및 보간 시각화
            holes_vis = img.copy()
            for cx, cy in hole_centers:
                cv2.circle(holes_vis, (int(cx), int(cy)), 8, (0, 255, 0), 2)
            for cx, cy in interpolated_points:
                cv2.circle(holes_vis, (int(cx), int(cy)), 8, (255, 0, 255), 2)
                cv2.circle(holes_vis, (int(cx), int(cy)), 2, (255, 0, 255), -1)
            cv2.putText(holes_vis, f"Detected: {len(hole_centers)}, Interpolated: {len(interpolated_points)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 직선 시각화
            lines_vis = img.copy()
            # 일반 선 (연한 회색)
            for x1, y1, x2, y2 in merged_lines:
                cv2.line(lines_vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
            # 몸체로 향하는 선 (초록색)
            for line_info in body_lines:
                x1, y1, x2, y2 = line_info['line']
                cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # 먼 쪽 끝점 표시
                far_point = line_info['far_point']
                cv2.circle(lines_vis, (int(far_point[0]), int(far_point[1])), 6, (0, 255, 0), -1)
            cv2.putText(lines_vis, f"Total: {len(merged_lines)}, To Body: {len(body_lines)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # LED 몸체 시각화
            body_vis = img.copy()
            if hull is not None:
                cv2.drawContours(body_vis, [hull], -1, (0, 255, 255), 2)
                overlay = cv2.cvtColor(body_mask, cv2.COLOR_GRAY2BGR)
                overlay[:,:,0] = 0
                overlay[:,:,1] = body_mask
                overlay[:,:,2] = body_mask
                body_vis = cv2.addWeighted(body_vis, 0.7, overlay, 0.3, 0)
            if centroid is not None:
                cv2.circle(body_vis, centroid, 8, (0, 255, 255), -1)
            
            # 최종 결과
            final_vis = visualize_new_results(img, hole_centers, interpolated_points, 
                                            all_lines, merged_lines, hull, centroid,
                                            body_lines, lead_candidates, selected_leads, excluded_holes)
            
            # 창에 표시
            cv2.imshow('Holes & Interpolation', holes_vis)
            cv2.imshow('Lines Detected', lines_vis)
            cv2.imshow('LED Body', body_vis)
            cv2.imshow('Final Result', final_vis)
            
            # 콘솔 출력
            print(f"\r[{images[idx]}] Body Lines:{len(body_lines)}, Excluded:{len(excluded_holes)}, Candidates:{len(lead_candidates)}, Selected:{len(selected_leads)}", end='')
            
            # 선택된 리드들의 상세 정보 출력
            if selected_leads:
                print()  # 새 줄
                for i, lead in enumerate(selected_leads):
                    cx, cy = lead['hole_pos']
                    score = lead['score']
                    dist_to_endpoint = lead['distance_to_endpoint']
                    print(f"  LED {i+1}: Position({cx:.1f}, {cy:.1f}), Score: {score:.1f}, EndpointDist: {dist_to_endpoint:.1f}")
                
                # 제외된 구멍들 정보 출력
                if excluded_holes:
                    print(f"  Excluded {len(excluded_holes)} holes near LED body (within {led_body_exclusion}px)")
            
        except Exception as e:
            print(f"\nError processing image: {e}")
            import traceback
            traceback.print_exc()
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 또는 ESC
            break
        elif key == ord('n'):  # 다음 이미지
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx])
            print()  # 새 줄
        elif key == ord('p'):  # 이전 이미지
            idx = (idx - 1) % len(images)
            img = cv2.imread(images[idx])
            print()  # 새 줄
        elif key == ord('s'):  # 결과 저장
            if 'final_vis' in locals():
                base_name = os.path.splitext(images[idx])[0]
                save_name = f"{base_name}_new_led_detection_result.png"
                cv2.imwrite(save_name, final_vis)
                print(f"\nSaved: {save_name}")
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()