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

    
def intersect_similar_lines(lines1, lines2, angle_thresh, dist_thresh):
    """lines1과 lines2에서 유사한(=교집합) 선분만 추출"""
    inter = []
    for x1, y1, x2, y2 in lines1:
        angle1 = math.atan2(y2 - y1, x2 - x1)
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        for xx1, yy1, xx2, yy2 in lines2:
            angle2 = math.atan2(yy2 - yy1, xx2 - xx1)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            mid2 = ((xx1 + xx2) / 2, (yy1 + yy2) / 2)
            dist = math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1])
            if angle_diff < angle_thresh and dist < dist_thresh:
                inter.append((x1, y1, x2, y2))
                break
    return inter

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

def merge_lines_dbscan(lines, angle_thresh, dist_thresh, min_samples=2):
    """
    DBSCAN을 이용해 각도와 위치가 유사한 선들을 클러스터링하고,
    각 클러스터의 양 끝단(가장 먼 두 점)을 잇는 선으로 병합해서 반환.
    """
    if not lines:
        return []

    # 1) DBSCAN을 위한 feature 벡터 생성
    w = dist_thresh / angle_thresh
    feats = []
    for x1,y1,x2,y2 in lines:
        angle = math.atan2(y2-y1, x2-x1)
        midx, midy = (x1+x2)/2, (y1+y2)/2
        feats.append([midx, midy, angle * w])
    feats = np.array(feats)

    # 2) 클러스터링
    clustering = DBSCAN(eps=dist_thresh, min_samples=min_samples).fit(feats)
    labels = clustering.labels_

    merged = []
    # 3) 각 클러스터별로 선 재생성
    for lbl in set(labels):
        idxs = np.where(labels == lbl)[0]
        # 노이즈(-1)는 그냥 원본 유지
        if lbl == -1:
            for i in idxs:
                merged.append(lines[i])
            continue

        # 클러스터에 속한 모든 선의 엔드포인트 수집
        pts = []
        for i in idxs:
            x1,y1,x2,y2 = lines[i]
            pts.append((x1,y1))
            pts.append((x2,y2))

        # 가장 먼 두 점 찾기
        max_d = 0
        best = None
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
                if d > max_d:
                    max_d = d
                    best = (pts[i], pts[j])

        if best:
            (sx,sy), (ex,ey) = best
            merged.append((sx,sy,ex,ey))

    return merged

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

# ========== 1. 대칭성 기반 리드 쌍 검출 ==========
def find_symmetric_lead_pairs(hole_centers, led_centroid, symmetry_tolerance=15, 
                            distance_ratio_range=(0.8, 1.2)):
    """LED 중심 기준으로 대칭적인 구멍 쌍 찾기"""
    if not hole_centers or led_centroid is None:
        return []
    
    symmetric_pairs = []
    used_holes = set()
    
    for i, hole1 in enumerate(hole_centers):
        if i in used_holes:
            continue
            
        x1, y1 = hole1
        # LED 중심 기준으로 반대편 예상 위치 계산
        expected_x = 2 * led_centroid[0] - x1
        expected_y = 2 * led_centroid[1] - y1
        
        # 가장 가까운 구멍 찾기
        best_match = None
        best_distance = float('inf')
        best_idx = -1
        
        for j, hole2 in enumerate(hole_centers):
            if j == i or j in used_holes:
                continue
                
            x2, y2 = hole2
            distance_to_expected = math.hypot(x2 - expected_x, y2 - expected_y)
            
            if distance_to_expected < symmetry_tolerance and distance_to_expected < best_distance:
                # 중심으로부터의 거리 비율 확인
                dist1 = math.hypot(x1 - led_centroid[0], y1 - led_centroid[1])
                dist2 = math.hypot(x2 - led_centroid[0], y2 - led_centroid[1])
                
                if dist1 > 0:
                    ratio = dist2 / dist1
                    if distance_ratio_range[0] <= ratio <= distance_ratio_range[1]:
                        best_match = hole2
                        best_distance = distance_to_expected
                        best_idx = j
        
        if best_match is not None:
            symmetric_pairs.append({
                'hole1': hole1,
                'hole2': best_match,
                'hole1_idx': i,
                'hole2_idx': best_idx,
                'symmetry_error': best_distance,
                'pair_indices': (i, best_idx)
            })
            used_holes.add(i)
            used_holes.add(best_idx)
    
    return symmetric_pairs

# ========== 2. 고급 점수 매기기 시스템 ==========
def advanced_lead_scoring(hole_centers, lines, led_hull, led_centroid, 
                        symmetric_pairs, max_distance=30, exclude_distance=10, 
                        min_body_distance=50):
    """다중 기준 기반 고급 점수 매기기"""
    scored_holes = []
    
    # 직선과 구멍 매칭 정보 미리 계산
    line_hole_connections = {}
    for hole_idx, (hx, hy) in enumerate(hole_centers):
        min_line_dist = float('inf')
        best_line = None
        
        for line in lines:
            x1, y1, x2, y2 = line
            # 각 직선의 끝점에서 구멍까지의 거리
            dist1 = math.hypot(hx - x1, hy - y1)
            dist2 = math.hypot(hx - x2, hy - y2)
            min_dist = min(dist1, dist2)
            
            if exclude_distance < min_dist <= max_distance and min_dist < min_line_dist:
                # LED 몸체로부터의 거리 체크
                if led_hull is not None:
                    body_dist = cv2.pointPolygonTest(led_hull, (float(hx), float(hy)), True)
                    if abs(body_dist) >= min_body_distance:
                        min_line_dist = min_dist
                        best_line = line
        
        line_hole_connections[hole_idx] = {
            'min_distance': min_line_dist,
            'best_line': best_line
        }
    
    for hole_idx, (hx, hy) in enumerate(hole_centers):
        score_components = {}
        
        # 1. 기본 거리 점수 (LED 중심 및 몸체로부터의 거리)
        if led_centroid is not None:
            dist_to_center = math.hypot(hx - led_centroid[0], hy - led_centroid[1])
            score_components['center_distance'] = max(0, 100 - dist_to_center * 0.5)
        else:
            score_components['center_distance'] = 0
        
        if led_hull is not None:
            body_dist = abs(cv2.pointPolygonTest(led_hull, (float(hx), float(hy)), True))
            if body_dist < min_body_distance:
                score_components['body_distance'] = 0  # 너무 가까우면 0점
            else:
                score_components['body_distance'] = min(100, body_dist * 2)
        else:
            score_components['body_distance'] = 50
        
        # 2. 대칭성 점수
        score_components['symmetry'] = 0
        for pair in symmetric_pairs:
            if hole_idx in [pair['hole1_idx'], pair['hole2_idx']]:
                # 대칭성이 좋을수록 높은 점수 (최대 100점)
                symmetry_score = max(0, 100 - pair['symmetry_error'] * 5)
                score_components['symmetry'] = symmetry_score
                break
        
        # 3. 직선 연결성 점수
        connection_info = line_hole_connections.get(hole_idx, {})
        if connection_info.get('best_line') is not None:
            line_dist = connection_info['min_distance']
            score_components['line_connectivity'] = max(0, 50 - line_dist)
            
            # 직선 방향성 점수 추가
            line = connection_info['best_line']
            if led_centroid is not None:
                direction_score = calculate_line_direction_score(line, (hx, hy), led_centroid)
                score_components['direction'] = direction_score
            else:
                score_components['direction'] = 0
        else:
            score_components['line_connectivity'] = 0
            score_components['direction'] = 0
        
        # 4. 구멍 품질 점수 (크기, 원형도 등 - hole_info가 있다면)
        score_components['hole_quality'] = 50  # 기본값
        
        # 5. 경계 거리 점수 (이미지 경계에서 너무 가깝지 않은지)
        # 이 부분은 이미지 크기 정보가 필요하므로 생략하거나 별도 구현
        
        # 최종 점수 합계 (가중 평균)
        weights = {
            'center_distance': 0.15,
            'body_distance': 0.25,
            'symmetry': 0.30,
            'line_connectivity': 0.20,
            'direction': 0.10
        }
        
        total_score = sum(score_components[key] * weights[key] for key in weights)
        
        scored_holes.append({
            'hole_idx': hole_idx,
            'hole_pos': (hx, hy),
            'total_score': total_score,
            'score_components': score_components,
            'line_info': connection_info
        })
    
    return scored_holes

def calculate_line_direction_score(line, hole_pos, led_centroid):
    """직선이 LED 중심을 향하는 방향성 점수 계산"""
    x1, y1, x2, y2 = line
    hx, hy = hole_pos
    
    # 구멍에 더 가까운 끝점 찾기
    dist1 = math.hypot(hx - x1, hy - y1)
    dist2 = math.hypot(hx - x2, hy - y2)
    
    if dist1 < dist2:
        line_point = (x1, y1)
        other_point = (x2, y2)
    else:
        line_point = (x2, y2)
        other_point = (x1, y1)
    
    # 직선 방향 벡터 (구멍 쪽 점에서 다른 점으로)
    line_vec = np.array([other_point[0] - line_point[0], other_point[1] - line_point[1]])
    line_mag = np.linalg.norm(line_vec)
    
    if line_mag == 0:
        return 0
    
    line_vec_norm = line_vec / line_mag
    
    # 구멍에서 LED 중심으로의 벡터
    to_center_vec = np.array([led_centroid[0] - hx, led_centroid[1] - hy])
    center_mag = np.linalg.norm(to_center_vec)
    
    if center_mag == 0:
        return 0
    
    to_center_vec_norm = to_center_vec / center_mag
    
    # 코사인 유사도 계산
    cos_angle = np.dot(line_vec_norm, to_center_vec_norm)
    
    # 방향이 일치할수록 높은 점수 (0~50점)
    direction_score = max(0, cos_angle * 50)
    
    return direction_score

# ========== 3. 품질 검증 및 에러 처리 ==========
def validate_lead_detection_quality(selected_leads, led_centroid, 
                                   expected_distance_range=(30, 150),
                                   max_asymmetry_ratio=2.0,
                                   min_confidence_score=30):
    """검출 품질 검증"""
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # 1. 개수 확인
    if len(selected_leads) != 2:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"정확히 2개의 리드가 필요하나 {len(selected_leads)}개 검출됨")
        return validation_results
    
    # 2. 신뢰도 점수 확인
    for i, lead in enumerate(selected_leads):
        if lead['total_score'] < min_confidence_score:
            validation_results['warnings'].append(f"리드 {i+1}의 신뢰도가 낮음: {lead['total_score']:.1f}")
    
    # 3. 두 리드 간 거리 확인
    lead1_pos = selected_leads[0]['hole_pos']
    lead2_pos = selected_leads[1]['hole_pos']
    distance_between_leads = math.hypot(
        lead1_pos[0] - lead2_pos[0],
        lead1_pos[1] - lead2_pos[1]
    )
    
    if not (expected_distance_range[0] <= distance_between_leads <= expected_distance_range[1]):
        validation_results['warnings'].append(
            f"리드 간 거리가 예상 범위를 벗어남: {distance_between_leads:.1f}px "
            f"(예상: {expected_distance_range[0]}-{expected_distance_range[1]}px)"
        )
    
    # 4. 대칭성 확인
    if led_centroid is not None:
        center_to_lead1 = math.hypot(
            lead1_pos[0] - led_centroid[0],
            lead1_pos[1] - led_centroid[1]
        )
        center_to_lead2 = math.hypot(
            lead2_pos[0] - led_centroid[0],
            lead2_pos[1] - led_centroid[1]
        )
        
        if center_to_lead1 > 0 and center_to_lead2 > 0:
            distance_ratio = max(center_to_lead1, center_to_lead2) / min(center_to_lead1, center_to_lead2)
            if distance_ratio > max_asymmetry_ratio:
                validation_results['warnings'].append(
                    f"리드의 비대칭성이 큼: 중심 거리 비율 {distance_ratio:.2f}"
                )
    
    # 5. 점수 구성 요소 분석
    for i, lead in enumerate(selected_leads):
        components = lead.get('score_components', {})
        
        # 대칭성 점수가 너무 낮으면 경고
        if components.get('symmetry', 0) < 20:
            validation_results['warnings'].append(f"리드 {i+1}의 대칭성 점수가 낮음")
        
        # 직선 연결성이 없으면 경고
        if components.get('line_connectivity', 0) < 10:
            validation_results['warnings'].append(f"리드 {i+1}의 직선 연결성이 낮음")
    
    return validation_results

def robust_lead_selection(scored_holes, min_distance_between_leads=20, 
                         confidence_threshold=30, prefer_symmetric_pairs=True):
    """강건한 최종 리드 선택"""
    # 점수순 정렬
    scored_holes.sort(key=lambda x: x['total_score'], reverse=True)
    
    selected_leads = []
    
    # 대칭 쌍을 우선적으로 고려
    if prefer_symmetric_pairs:
        # 대칭성 점수가 높은 구멍들 우선 선택
        symmetric_candidates = [h for h in scored_holes 
                              if h['score_components'].get('symmetry', 0) > 50]
        
        if len(symmetric_candidates) >= 2:
            # 대칭성이 높은 후보들 중에서 선택
            for candidate in symmetric_candidates:
                if candidate['total_score'] < confidence_threshold:
                    continue
                
                too_close = False
                for selected in selected_leads:
                    dist = math.hypot(
                        candidate['hole_pos'][0] - selected['hole_pos'][0],
                        candidate['hole_pos'][1] - selected['hole_pos'][1]
                    )
                    if dist < min_distance_between_leads:
                        too_close = True
                        break
                
                if not too_close:
                    selected_leads.append(candidate)
                    
                if len(selected_leads) >= 2:
                    break
    
    # 대칭 쌍으로 충분하지 않다면 일반적인 방법으로 보완
    if len(selected_leads) < 2:
        for candidate in scored_holes:
            if candidate in selected_leads:
                continue
                
            if candidate['total_score'] < confidence_threshold:
                continue
            
            too_close = False
            for selected in selected_leads:
                dist = math.hypot(
                    candidate['hole_pos'][0] - selected['hole_pos'][0],
                    candidate['hole_pos'][1] - selected['hole_pos'][1]
                )
                if dist < min_distance_between_leads:
                    too_close = True
                    break
            
            if not too_close:
                selected_leads.append(candidate)
                
            if len(selected_leads) >= 2:
                break
    
    return selected_leads

# --- 직선과 구멍 간 거리 계산 (기존 함수 유지) ---
def find_nearest_holes_to_line_endpoints(lines, hole_centers, hull, max_distance=30, 
                                        exclude_distance=10, min_body_distance=50):
    """각 직선의 끝점에서 가장 가까운 구멍 찾기 (너무 가까운 구멍, 몸체에 가까운 구멍 제외)"""
    line_hole_pairs = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        endpoints = [(x1, y1), (x2, y2)]
        
        line_info = {
            'line': line,
            'holes': [],
            'distances': []
        }
        
        for endpoint in endpoints:
            min_dist = float('inf')
            nearest_hole = None
            nearest_idx = -1
            
            for idx, hole in enumerate(hole_centers):
                dist = np.hypot(endpoint[0] - hole[0], endpoint[1] - hole[1])
                
                # LED 몸체로부터의 거리 체크
                if hull is not None:
                    body_dist = cv2.pointPolygonTest(hull, (float(hole[0]), float(hole[1])), True)
                    # 음수면 hull 외부이고 절대값이 거리
                    if abs(body_dist) < min_body_distance:
                        continue  # 몸체에 너무 가까운 구멍은 제외
                
                # 너무 가까운 구멍은 제외하고, 적정 거리의 구멍만 선택
                if exclude_distance < dist < min_dist and dist <= max_distance:
                    min_dist = dist
                    nearest_hole = hole
                    nearest_idx = idx
            
            if nearest_hole is not None:
                line_info['holes'].append((nearest_idx, nearest_hole))
                line_info['distances'].append(min_dist)
        
        if line_info['holes']:
            line_hole_pairs.append(line_info)
    
    return line_hole_pairs

# --- 시각화 함수 (개선: 점수 구성 요소 표시) ---
def visualize_results(img, hole_centers, interpolated_points, lines, hull, centroid, 
                     selected_holes, all_candidates, validation_results=None):
    """최종 결과 시각화 (개선된 버전)"""
    vis = img.copy()
    
    # LED 몸체 그리기
    if hull is not None:
        cv2.drawContours(vis, [hull], -1, (0, 255, 255), 2)
        cv2.putText(vis, "LED Body", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if centroid is not None:
        cv2.circle(vis, centroid, 5, (0, 255, 255), -1)
    
    # 검출된 구멍 그리기 (회색)
    for i, (cx, cy) in enumerate(hole_centers):
        cv2.circle(vis, (int(cx), int(cy)), 8, (128, 128, 128), 1)
        cv2.circle(vis, (int(cx), int(cy)), 2, (128, 128, 128), -1)
    
    # 보간된 구멍 그리기 (보라색)
    for cx, cy in interpolated_points:
        cv2.circle(vis, (int(cx), int(cy)), 8, (255, 0, 255), 1)
        cv2.circle(vis, (int(cx), int(cy)), 2, (255, 0, 255), -1)
    
    # 모든 직선 그리기 (연한 초록)
    for x1, y1, x2, y2 in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 128, 0), 1)
    
    # 후보 구멍들 표시 (파란색) - 선택되지 않은 후보들
    selected_indices = [h['hole_idx'] for h in selected_holes]
    for candidate in all_candidates:
        if candidate['hole_idx'] not in selected_indices:
            cx, cy = candidate['hole_pos']
            cx, cy = int(cx), int(cy)
            # 후보점 표시
            cv2.circle(vis, (cx, cy), 10, (255, 128, 0), 2)  # 주황색
            cv2.circle(vis, (cx, cy), 3, (255, 128, 0), -1)
            # 점수 표시
            score_text = f"{candidate['total_score']:.0f}"
            cv2.putText(vis, score_text, (cx + 12, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
    
    # 최종 선택된 LED 리드 구멍 강조 (빨간색)
    for i, hole_info in enumerate(selected_holes):
        cx, cy = hole_info['hole_pos']
        cx, cy = int(cx), int(cy)
        
        # 구멍 강조
        cv2.circle(vis, (cx, cy), 12, (0, 0, 255), 3)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        
        # 관련 직선 강조 (있다면)
        line_info = hole_info.get('line_info', {})
        if line_info.get('best_line') is not None:
            x1, y1, x2, y2 = line_info['best_line']
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # LED 번호와 점수 표시
        cv2.putText(vis, f"LED {i+1}", (cx + 15, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        score_text = f"Score: {hole_info['total_score']:.1f}"
        cv2.putText(vis, score_text, (cx + 15, cy + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 점수 구성 요소 표시
        components = hole_info.get('score_components', {})
        y_offset = 35
        for key, value in components.items():
            if key in ['symmetry', 'line_connectivity']:  # 주요 점수만 표시
                comp_text = f"{key[:3]}: {value:.0f}"
                cv2.putText(vis, comp_text, (cx + 15, cy + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
    
    # 정보 표시
    info_y = vis.shape[0] - 120
    cv2.putText(vis, f"Detected Holes: {len(hole_centers)} (gray)", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    cv2.putText(vis, f"Interpolated: {len(interpolated_points)} (purple)", (10, info_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(vis, f"Candidates: {len(all_candidates)} (orange), Selected: {len(selected_holes)} (red)", 
               (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Lines: {len(lines)}", (10, info_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 검증 결과 표시
    if validation_results is not None:
        status_color = (0, 255, 0) if validation_results['is_valid'] else (0, 0, 255)
        status_text = "VALID" if validation_results['is_valid'] else "INVALID"
        cv2.putText(vis, f"Status: {status_text}", (10, info_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        if validation_results['warnings']:
            warning_count = len(validation_results['warnings'])
            cv2.putText(vis, f"Warnings: {warning_count}", (10, info_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return vis

# --- 메인 함수 (개선된 버전) ---
def main():
    # LED 이미지 찾기
    images = [f for f in os.listdir('.') if 'led' in f.lower() and 
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
    
    # 트랙바 생성 (개선된 파라미터 포함)
    cv2.createTrackbar('Min Length', 'Controller', 30, 100, lambda x: None)
    cv2.createTrackbar('CLAHE Clip x10', 'Controller', 20, 100, lambda x: None)
    cv2.createTrackbar('Min Hole Area', 'Controller', 50, 500, lambda x: None)
    cv2.createTrackbar('Max Hole Area', 'Controller', 500, 2000, lambda x: None)
    cv2.createTrackbar('Max Line-Hole Dist', 'Controller', 30, 100, lambda x: None)
    cv2.createTrackbar('Min Line-Hole Dist', 'Controller', 10, 50, lambda x: None)
    cv2.createTrackbar('Min Body Distance', 'Controller', 50, 150, lambda x: None)
    cv2.createTrackbar('Merge Angle', 'Controller', 10, 45, lambda x: None)
    cv2.createTrackbar('Merge Distance', 'Controller', 25, 100, lambda x: None)
    cv2.createTrackbar('Use Otsu', 'Controller', 1, 1, lambda x: None)
    cv2.createTrackbar('Enable Interpolation', 'Controller', 1, 1, lambda x: None)
    cv2.createTrackbar('Row/Col Eps', 'Controller', 15, 50, lambda x: None)
    cv2.createTrackbar('Symmetry Tolerance', 'Controller', 15, 50, lambda x: None)  # 새로 추가
    cv2.createTrackbar('Confidence Threshold', 'Controller', 30, 100, lambda x: None)  # 새로 추가
    
    print("\nControls:")
    print("- 'n': Next image")
    print("- 'p': Previous image")
    print("- 's': Save result")
    print("- 'q' or ESC: Quit")
    print("\nImproved Features:")
    print("- Symmetry-based lead pair detection")
    print("- Advanced multi-criteria scoring system")
    print("- Quality validation and error handling")
    print("- Detailed score component analysis")
    
    while True:
        # 트랙바 값 읽기
        min_length = cv2.getTrackbarPos('Min Length', 'Controller')
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip x10', 'Controller') / 10.0
        min_hole_area = cv2.getTrackbarPos('Min Hole Area', 'Controller')
        max_hole_area = cv2.getTrackbarPos('Max Hole Area', 'Controller')
        max_line_hole_dist = cv2.getTrackbarPos('Max Line-Hole Dist', 'Controller')
        min_line_hole_dist = cv2.getTrackbarPos('Min Line-Hole Dist', 'Controller')
        min_body_distance = cv2.getTrackbarPos('Min Body Distance', 'Controller')
        merge_angle = cv2.getTrackbarPos('Merge Angle', 'Controller')
        merge_dist = cv2.getTrackbarPos('Merge Distance', 'Controller')
        use_otsu = cv2.getTrackbarPos('Use Otsu', 'Controller') == 1
        enable_interpolation = cv2.getTrackbarPos('Enable Interpolation', 'Controller') == 1
        eps = cv2.getTrackbarPos('Row/Col Eps', 'Controller')
        symmetry_tolerance = cv2.getTrackbarPos('Symmetry Tolerance', 'Controller')
        confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'Controller')
        
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
            hull, centroid, body_mask = detect_led_body(img)
            
            # 4. 직선 검출
            lines_gray = detect_lines_lsd(processed, min_length)
            lines_binary = detect_lines_lsd(binary, min_length)
            
            # 모든 선 합치기
            all_lines = lines_gray + lines_binary
            
            # 유사한 선 병합
            angle_thresh_rad = np.deg2rad(merge_angle)
            
            angle_thresh = np.deg2rad(10)
            dist_thresh = 20
            merged_lines = intersect_similar_lines(
                lines_gray,
                lines_binary,
                angle_thresh,
                dist_thresh
            )

            merged_lines = merge_lines_dbscan(merged_lines, angle_thresh, dist_thresh, min_samples=2)
            
            # ========== 개선된 LED 리드 검출 파이프라인 ==========
            
            # 5. 대칭성 기반 리드 쌍 검출
            symmetric_pairs = find_symmetric_lead_pairs(all_holes, centroid, symmetry_tolerance)
            
            # 6. 고급 점수 매기기 시스템
            scored_holes = advanced_lead_scoring(
                all_holes, merged_lines, hull, centroid, symmetric_pairs,
                max_line_hole_dist, min_line_hole_dist, min_body_distance)
            
            # 7. 강건한 최종 선택
            selected_holes = robust_lead_selection(
                scored_holes, 
                min_distance_between_leads=20, 
                confidence_threshold=confidence_threshold,
                prefer_symmetric_pairs=True)
            
            # 8. 품질 검증
            validation_results = validate_lead_detection_quality(selected_holes, centroid)
            
            # 시각화
            # 구멍 및 보간 시각화
            holes_vis = img.copy()
            # 검출된 구멍
            for i, (cx, cy) in enumerate(hole_centers):
                cv2.circle(holes_vis, (int(cx), int(cy)), 8, (0, 255, 0), 2)
            # 보간된 구멍
            for cx, cy in interpolated_points:
                cv2.circle(holes_vis, (int(cx), int(cy)), 8, (255, 0, 255), 2)
                cv2.circle(holes_vis, (int(cx), int(cy)), 2, (255, 0, 255), -1)
            # 대칭 쌍 표시
            for pair in symmetric_pairs:
                x1, y1 = pair['hole1']
                x2, y2 = pair['hole2']
                cv2.line(holes_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.putText(holes_vis, f"Sym:{pair['symmetry_error']:.1f}", 
                           (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            cv2.putText(holes_vis, f"Detected: {len(hole_centers)}, Interpolated: {len(interpolated_points)}, Symmetric pairs: {len(symmetric_pairs)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 직선 시각화
            lines_vis = img.copy()
            for x1, y1, x2, y2 in merged_lines:
                cv2.line(lines_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(lines_vis, f"Total Lines: {len(merged_lines)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
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
                cv2.circle(body_vis, centroid, 5, (0, 255, 255), -1)
                cv2.putText(body_vis, "Center", (centroid[0]+10, centroid[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 최종 결과 (개선된 시각화)
            final_vis = visualize_results(img, hole_centers, interpolated_points, 
                                        merged_lines, hull, centroid, selected_holes, 
                                        scored_holes, validation_results)
            
            # 창에 표시
            cv2.imshow('Holes & Interpolation', holes_vis)
            cv2.imshow('Lines Detected', lines_vis)
            cv2.imshow('LED Body', body_vis)
            cv2.imshow('Final Result', final_vis)
            
            # 콘솔 출력 (개선된 정보)
            status_str = "VALID" if validation_results['is_valid'] else "INVALID"
            warning_count = len(validation_results.get('warnings', []))
            
            print(f"\r[{images[idx]}] Holes:{len(hole_centers)}, Interp:{len(interpolated_points)}, "
                  f"Sym_pairs:{len(symmetric_pairs)}, Lines:{len(merged_lines)}, "
                  f"Candidates:{len(scored_holes)}, Selected:{len(selected_holes)}, "
                  f"Status:{status_str}, Warnings:{warning_count}", end='')
            
            # 경고 메시지 출력
            if validation_results.get('warnings'):
                print("\nWarnings:")
                for warning in validation_results['warnings']:
                    print(f"  - {warning}")
            
            if validation_results.get('errors'):
                print("\nErrors:")
                for error in validation_results['errors']:
                    print(f"  - {error}")
            
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
                save_name = f"{base_name}_improved_led_detection_result.png"
                cv2.imwrite(save_name, final_vis)
                print(f"\nSaved: {save_name}")
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()