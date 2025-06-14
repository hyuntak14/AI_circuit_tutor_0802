import cv2
import numpy as np
import math
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Using simple clustering instead.")
    print("For better results, install sklearn: pip install scikit-learn")

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

def cover_holes_excluding_led(binary_img, hole_centers, interpolated_points, led_hull, square_size):
    """
    LED 몸체 범위를 제외한 구멍 위치를 검은 정사각형으로 가림

    Parameters:
    - binary_img: 구멍 검출 이후의 이진 이미지
    - hole_centers: detect_square_holes에서 반환된 구멍 중심 좌표 리스트
    - interpolated_points: 보간된 구멍 좌표 리스트
    - led_hull: detect_led_body에서 반환된 LED 몸체 convex hull
    - square_size: 검은색 정사각형의 한 변 길이 (픽셀)
    """
    masked = binary_img.copy()
    all_points = []
    if hole_centers:
        all_points.extend(hole_centers)
    if interpolated_points:
        all_points.extend(interpolated_points)

    for (cx, cy) in all_points:
        x_int, y_int = int(round(cx)), int(round(cy))
        # LED 몸체 내부는 제외
        if led_hull is not None and cv2.pointPolygonTest(led_hull, (x_int, y_int), False) >= 0:
            continue
        # 정사각형 좌표 계산
        half = square_size // 2
        x0 = max(x_int - half, 0)
        y0 = max(y_int - half, 0)
        x1 = min(x_int + half, masked.shape[1] - 1)
        y1 = min(y_int + half, masked.shape[0] - 1)
        # 해당 영역을 검은색으로 채움
        cv2.rectangle(masked, (x0, y0), (x1, y1), 0, -1)
    return masked


def detect_led_holes_brightness(gray_img, hole_centers, hole_radius, thresh=150):
    """
    밝기 기반 임계치로 LED 리드가 꽂힌 구멍 판별
    """
    led_indices = []
    for i, (cx, cy) in enumerate(hole_centers):
        x, y = int(round(cx)), int(round(cy))
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.circle(mask, (x, y), hole_radius, 255, -1)
        mean_val = cv2.mean(gray_img, mask=mask)[0]
        if mean_val > thresh:
            led_indices.append(i)
    return led_indices

def detect_led_holes_by_lines(gray_img, hole_centers, hole_radius,
                              canny_thresh1=50, canny_thresh2=150,
                              hough_thresh=10, min_line_len=20):
    """
    에지 + 허프 변환으로 LED 리드(직선) 검출
    """
    led_indices = []
    for i, (cx, cy) in enumerate(hole_centers):
        x, y = int(round(cx)), int(round(cy))
        r = hole_radius
        h, w = gray_img.shape[:2]
        x0, y0 = max(x-r,0), max(y-r,0)
        x1, y1 = min(x+r, w), min(y+r, h)
        roi = gray_img[y0:y1, x0:x1]
        edges = cv2.Canny(roi, canny_thresh1, canny_thresh2)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_thresh,
                                minLineLength=min_line_len, maxLineGap=5)
        if lines is not None:
            led_indices.append(i)
    return led_indices




def simple_dbscan(points, eps, min_samples):
    """sklearn이 없을 때 사용하는 간단한 클러스터링"""
    if len(points) == 0:
        return np.array([])
    
    points = np.array(points).reshape(-1, 1)
    labels = np.full(len(points), -1)
    cluster_id = 0
    
    for i, point in enumerate(points):
        if labels[i] != -1:  # 이미 클러스터에 할당됨
            continue
            
        # 이 점 주변의 이웃들 찾기
        neighbors = []
        for j, other_point in enumerate(points):
            if abs(point[0] - other_point[0]) <= eps:
                neighbors.append(j)
        
        if len(neighbors) >= min_samples:
            # 새 클러스터 생성
            for neighbor_idx in neighbors:
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
            cluster_id += 1
    
    return type('', (), {'labels_': labels})()  # sklearn 스타일 객체 모방

def safe_find_contours(image, mode, method):
    """OpenCV 버전에 상관없이 안전하게 contours를 찾는 함수"""
    result = cv2.findContours(image, mode, method)
    if len(result) == 3:
        # OpenCV 3.x
        _, contours, hierarchy = result
    else:
        # OpenCV 4.x
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

def advanced_preprocessing(img_gray, apply_sharpen=False, apply_smooth=False, clahe_clip=2.0):
    """향상된 전처리 파이프라인"""
    processed = img_gray.copy()
    
    # 1) 노이즈 감소
    if apply_smooth:
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
    
    # 2) 선명화
    if apply_sharpen:
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel_sharpen)
    
    # 3) CLAHE 적용
    processed = apply_clahe(processed, clipLimit=clahe_clip)
    
    # 4) 가우시안 블러로 미세한 노이즈 제거
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    
    return processed

def visualize_holes(img, hole_centers, hole_radius, hole_info=None, interpolated_points=None, title="Holes Detected"):
    """구멍들과 보간된 점들을 시각화하는 함수"""
    vis = img.copy()
    
    # 검출된 구멍들 표시
    for i, (cx, cy) in enumerate(hole_centers):
        x_int, y_int = int(round(cx)), int(round(cy))
        
        # 구멍 영역을 원으로 표시 (빨간색)
        cv2.circle(vis, (x_int, y_int), hole_radius, (0, 0, 255), 2)
        
        # 중심점 표시 (파란색)
        cv2.circle(vis, (x_int, y_int), 3, (255, 0, 0), -1)
        
        # 바운딩 박스 표시 (초록색)
        if hole_info and i < len(hole_info):
            x, y, w, h = hole_info[i]['bbox']
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # 번호 표시 (노란색)
        cv2.putText(vis, str(i+1), (x_int+hole_radius+5, y_int-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 좌표 정보 표시
        coord_text = f"({x_int}, {y_int})"
        cv2.putText(vis, coord_text, (x_int+hole_radius+5, y_int+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 상세 정보 표시 (hole_info가 있는 경우)
        if hole_info and i < len(hole_info):
            info = hole_info[i]
            detail_text = f"A:{int(info['area'])} C:{info['circularity']:.2f} R:{info['aspect_ratio']:.2f}"
            cv2.putText(vis, detail_text, (x_int+hole_radius+5, y_int+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    # 보간된 점들 표시 (보라색)
    if interpolated_points:
        for i, (cx, cy) in enumerate(interpolated_points):
            x_int, y_int = int(round(cx)), int(round(cy))
            
            # 보간된 점을 다른 색으로 표시 (보라색)
            cv2.circle(vis, (x_int, y_int), hole_radius//2, (255, 0, 255), 2)
            cv2.circle(vis, (x_int, y_int), 2, (255, 0, 255), -1)
            
            # 보간 표시
            cv2.putText(vis, f"I{i+1}", (x_int+hole_radius//2+3, y_int-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    # 제목과 정보 추가
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis, f"Detected: {len(hole_centers)}", (10, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if interpolated_points:
        cv2.putText(vis, f"Interpolated: {len(interpolated_points)}", (10, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(vis, f"Total: {len(hole_centers) + len(interpolated_points)}", (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return vis

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
    """
    전체 좌표에서 행/열 기준점들을 찾고 격자 정보를 생성
    """
    if not hole_centers:
        return [], [], {}, 0, 0
    
    points = np.array(hole_centers)
    
    # 1. Y좌표 기준으로 행 클러스터링 (행의 기준 Y좌표들 찾기)
    y_coords = points[:, 1].reshape(-1, 1)
    if SKLEARN_AVAILABLE:
        row_clustering = DBSCAN(eps=row_eps, min_samples=min_samples).fit(y_coords)
    else:
        row_clustering = simple_dbscan(y_coords, row_eps, min_samples)
    
    # 2. X좌표 기준으로 열 클러스터링 (열의 기준 X좌표들 찾기)
    x_coords = points[:, 0].reshape(-1, 1)
    if SKLEARN_AVAILABLE:
        col_clustering = DBSCAN(eps=col_eps, min_samples=min_samples).fit(x_coords)
    else:
        col_clustering = simple_dbscan(x_coords, col_eps, min_samples)
    
    # 3. 각 행의 기준 Y좌표 계산
    row_y_centers = []
    row_clusters = {}
    for i, label in enumerate(row_clustering.labels_):
        if label >= 0:  # 노이즈 제외
            if label not in row_clusters:
                row_clusters[label] = []
            row_clusters[label].append(points[i])
    
    # 행별 Y 중심좌표 계산 및 정렬
    for row_id, points_in_row in row_clusters.items():
        y_center = np.mean([p[1] for p in points_in_row])
        row_y_centers.append(y_center)
    row_y_centers = sorted(row_y_centers)  # Y좌표 순으로 정렬
    
    # 4. 각 열의 기준 X좌표 계산
    col_x_centers = []
    col_clusters = {}
    for i, label in enumerate(col_clustering.labels_):
        if label >= 0:  # 노이즈 제외
            if label not in col_clusters:
                col_clusters[label] = []
            col_clusters[label].append(points[i])
    
    # 열별 X 중심좌표 계산 및 정렬
    for col_id, points_in_col in col_clusters.items():
        x_center = np.mean([p[0] for p in points_in_col])
        col_x_centers.append(x_center)
    col_x_centers = sorted(col_x_centers)  # X좌표 순으로 정렬
    
    # 5. 현재 존재하는 구멍들의 격자 위치 맵핑
    grid_map = {}
    for cx, cy in hole_centers:
        # 가장 가까운 행/열 찾기
        closest_row_idx = min(range(len(row_y_centers)), 
                             key=lambda i: abs(row_y_centers[i] - cy))
        closest_col_idx = min(range(len(col_x_centers)), 
                             key=lambda i: abs(col_x_centers[i] - cx))
        
        # 거리 확인 (너무 멀면 해당 격자에 속하지 않음)
        if (abs(row_y_centers[closest_row_idx] - cy) <= row_eps and 
            abs(col_x_centers[closest_col_idx] - cx) <= col_eps):
            grid_map[(closest_row_idx, closest_col_idx)] = (cx, cy)
    
    return row_y_centers, col_x_centers, grid_map, len(row_y_centers), len(col_x_centers)

def grid_based_interpolation(row_y_centers, col_x_centers, grid_map, num_rows, num_cols):
    """
    격자 기반 보간: 모든 격자 교차점에서 누락된 구멍을 찾아 보간
    """
    interpolated_points = []
    
    if not row_y_centers or not col_x_centers:
        return interpolated_points
    
    # 모든 격자 교차점 확인
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            # 해당 격자 위치에 구멍이 없으면 보간점 추가
            if (row_idx, col_idx) not in grid_map:
                x_coord = col_x_centers[col_idx]
                y_coord = row_y_centers[row_idx]
                interpolated_points.append((x_coord, y_coord))
    
    return interpolated_points

def smart_grid_interpolation(row_y_centers, col_x_centers, grid_map, num_rows, num_cols, 
                           min_coverage_ratio=0.3):
    """
    스마트 격자 보간: 각 행/열의 구멍 밀도를 고려한 보간
    """
    interpolated_points = []
    
    if not row_y_centers or not col_x_centers:
        return interpolated_points
    
    # 각 행별 구멍 개수 계산
    row_hole_counts = [0] * num_rows
    for (row_idx, col_idx), _ in grid_map.items():
        row_hole_counts[row_idx] += 1
    
    # 각 열별 구멍 개수 계산
    col_hole_counts = [0] * num_cols
    for (row_idx, col_idx), _ in grid_map.items():
        col_hole_counts[col_idx] += 1
    
    # 각 행/열의 최소 구멍 개수 임계값 계산
    max_holes_per_row = max(row_hole_counts) if row_hole_counts else 0
    max_holes_per_col = max(col_hole_counts) if col_hole_counts else 0
    
    min_holes_per_row = max(1, int(max_holes_per_row * min_coverage_ratio))
    min_holes_per_col = max(1, int(max_holes_per_col * min_coverage_ratio))
    
    # 충분한 구멍이 있는 행/열에서만 보간 수행
    for row_idx in range(num_rows):
        if row_hole_counts[row_idx] >= min_holes_per_row:
            for col_idx in range(num_cols):
                if col_hole_counts[col_idx] >= min_holes_per_col:
                    # 해당 격자 위치에 구멍이 없으면 보간점 추가
                    if (row_idx, col_idx) not in grid_map:
                        x_coord = col_x_centers[col_idx]
                        y_coord = row_y_centers[row_idx]
                        interpolated_points.append((x_coord, y_coord))
    
    return interpolated_points

def statistical_grid_interpolation(row_y_centers, col_x_centers, grid_map, num_rows, num_cols):
    """
    통계적 분석을 통한 격자 보간: 행/열별 구멍 분포를 분석하여 보간 여부 결정
    """
    interpolated_points = []
    
    if not row_y_centers or not col_x_centers:
        return interpolated_points
    
    # 각 행별 구멍 분포 분석
    row_distributions = {}
    for row_idx in range(num_rows):
        row_holes = []
        for col_idx in range(num_cols):
            if (row_idx, col_idx) in grid_map:
                row_holes.append(col_idx)
        row_distributions[row_idx] = sorted(row_holes)
    
    # 각 열별 구멍 분포 분석
    col_distributions = {}
    for col_idx in range(num_cols):
        col_holes = []
        for row_idx in range(num_rows):
            if (row_idx, col_idx) in grid_map:
                col_holes.append(row_idx)
        col_distributions[col_idx] = sorted(col_holes)
    
    # 행별 간격 분석 및 보간
    for row_idx, col_indices in row_distributions.items():
        if len(col_indices) >= 3:  # 최소 3개 구멍이 있어야 분석 가능
            # 간격 계산
            intervals = [col_indices[i+1] - col_indices[i] for i in range(len(col_indices)-1)]
            
            if intervals:
                mean_interval = np.mean(intervals)
                median_interval = np.median(intervals)
                std_interval = np.std(intervals)
                
                # 정상 간격 계산 (이상치 제거)
                normal_intervals = [x for x in intervals if abs(x - median_interval) <= 2 * std_interval]
                robust_mean = np.mean(normal_intervals) if normal_intervals else median_interval
                
                # 큰 간격에서 보간점 추가
                for i in range(len(col_indices)-1):
                    current_interval = col_indices[i+1] - col_indices[i]
                    
                    if current_interval > robust_mean * 1.5:  # 평균보다 1.5배 큰 간격
                        # 보간할 점의 개수 계산
                        num_interpolations = int(round(current_interval / robust_mean)) - 1
                        
                        if num_interpolations > 0:
                            for j in range(1, num_interpolations + 1):
                                interp_col_idx = col_indices[i] + j * (current_interval / (num_interpolations + 1))
                                
                                # 가장 가까운 실제 열 인덱스 찾기
                                closest_col_idx = int(round(interp_col_idx))
                                if 0 <= closest_col_idx < num_cols:
                                    if (row_idx, closest_col_idx) not in grid_map:
                                        x_coord = col_x_centers[closest_col_idx]
                                        y_coord = row_y_centers[row_idx]
                                        interpolated_points.append((x_coord, y_coord))
    
    # 열별 간격 분석 및 보간
    for col_idx, row_indices in col_distributions.items():
        if len(row_indices) >= 3:  # 최소 3개 구멍이 있어야 분석 가능
            # 간격 계산
            intervals = [row_indices[i+1] - row_indices[i] for i in range(len(row_indices)-1)]
            
            if intervals:
                mean_interval = np.mean(intervals)
                median_interval = np.median(intervals)
                std_interval = np.std(intervals)
                
                # 정상 간격 계산 (이상치 제거)
                normal_intervals = [x for x in intervals if abs(x - median_interval) <= 2 * std_interval]
                robust_mean = np.mean(normal_intervals) if normal_intervals else median_interval
                
                # 큰 간격에서 보간점 추가
                for i in range(len(row_indices)-1):
                    current_interval = row_indices[i+1] - row_indices[i]
                    
                    if current_interval > robust_mean * 1.5:  # 평균보다 1.5배 큰 간격
                        # 보간할 점의 개수 계산
                        num_interpolations = int(round(current_interval / robust_mean)) - 1
                        
                        if num_interpolations > 0:
                            for j in range(1, num_interpolations + 1):
                                interp_row_idx = row_indices[i] + j * (current_interval / (num_interpolations + 1))
                                
                                # 가장 가까운 실제 행 인덱스 찾기
                                closest_row_idx = int(round(interp_row_idx))
                                if 0 <= closest_row_idx < num_rows:
                                    if (closest_row_idx, col_idx) not in grid_map:
                                        x_coord = col_x_centers[col_idx]
                                        y_coord = row_y_centers[closest_row_idx]
                                        interpolated_points.append((x_coord, y_coord))
    
    return interpolated_points
    
    return interpolated_points

def detect_square_holes(binary_mask, min_area=50, max_area=2000, 
                       min_circularity=0.3, max_circularity=0.9,
                       min_square_ratio=0.6, max_square_ratio=1.4):
    """
    정사각형 형태의 구멍을 검출하는 함수
    
    Parameters:
    - binary_mask: 이진 마스크 (구멍이 흰색으로 표시됨)
    - min_area, max_area: 면적 범위
    - min_circularity, max_circularity: 원형도 범위 (0~1, 1에 가까울수록 원형)
    - min_square_ratio, max_square_ratio: 가로세로 비율 (1에 가까울수록 정사각형)
    """
    # OpenCV 버전에 안전한 findContours 사용
    contours, _ = safe_find_contours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_centers = []
    hole_info = []
    
    for contour in contours:
        # 1. 면적 필터링
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        # 2. 둘레 계산
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        # 3. 원형도 계산 (4π*면적 / 둘레²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity or circularity > max_circularity:
            continue
        
        # 4. 바운딩 박스를 통한 정사각형 비율 계산
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect_ratio = w / h
        if aspect_ratio < min_square_ratio or aspect_ratio > max_square_ratio:
            continue
        
        # 5. 컨투어의 볼록도(convexity) 검사
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
        if convexity < 0.7:  # 너무 오목한 형태는 제외
            continue
        
        # 6. 중심점 계산
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

def remove_hole_with_visualization(mask, img, hole_radius, min_area, max_area, 
                                 min_circularity, max_circularity, 
                                 min_square_ratio, max_square_ratio):
    """구멍을 제거하고 시각화 정보를 반환하는 함수"""
    centers, hole_info = detect_square_holes(
        mask, min_area, max_area, 
        min_circularity, max_circularity,
        min_square_ratio, max_square_ratio
    )
    
    clean_mask = mask.copy()
    H, W = clean_mask.shape[:2]
    half = hole_radius
    
    removed_regions = []
    
    for cx, cy in centers:
        x_int, y_int = int(round(cx)), int(round(cy))
        x0 = max(x_int - half, 0)
        y0 = max(y_int - half, 0)
        x1 = min(x_int + half, W - 1)
        y1 = min(y_int + half, H - 1)
        
        # 제거된 영역 정보 저장
        removed_regions.append((x0, y0, x1, y1))
        
        # 실제 마스크에서 제거
        cv2.rectangle(clean_mask, (x0, y0), (x1, y1), 0, -1)
    
    return clean_mask, centers, removed_regions, hole_info

def main():
    # LED 이미지 파일들 찾기
    import os
    image_files = [
        f for f in os.listdir('.')
        if ('led' in f.lower()) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_files:
        print("폴더 내에 'led'가 포함된 이미지가 없습니다.")
        print("현재 폴더의 모든 이미지 파일을 찾습니다...")
        image_files = [
            f for f in os.listdir('.')
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not image_files:
            print("이미지 파일이 없습니다.")
            return
    
    print(f"발견된 이미지 파일들: {image_files}")
    
    idx = 0
    img = cv2.imread(image_files[idx])
    if img is None:
        print(f"{image_files[idx]}를 읽을 수 없습니다.")
        return
    
    h, w = img.shape[:2]
    
    # 윈도우 설정
    win_name = "LED Hole Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w, h)
    
    def nothing(x):
        pass
    
    # 트랙바 생성 (구멍 검출 파라미터 추가)
    cv2.createTrackbar("hole_radius", win_name, 8, 50, nothing)
    cv2.createTrackbar("hole_area_thresh", win_name, 100, 500, nothing)
    cv2.createTrackbar("min_hole_area", win_name, 50, 1000, nothing)
    cv2.createTrackbar("max_hole_area", win_name, 500, 3000, nothing)
    cv2.createTrackbar("min_circularity", win_name, 30, 100, nothing)  # 0.3
    cv2.createTrackbar("max_circularity", win_name, 90, 100, nothing)  # 0.9
    cv2.createTrackbar("min_square_ratio", win_name, 60, 200, nothing)  # 0.6
    cv2.createTrackbar("max_square_ratio", win_name, 140, 200, nothing)  # 1.4
    cv2.createTrackbar("Use_Otsu", win_name, 1, 1, nothing)
    cv2.createTrackbar("Apply_Sharpen", win_name, 0, 1, nothing)
    cv2.createTrackbar("Apply_Smooth", win_name, 1, 1, nothing)
    cv2.createTrackbar("CLAHE_clip", win_name, 3, 10, nothing)
    cv2.createTrackbar("block_size", win_name, 15, 51, nothing)
    cv2.createTrackbar("Show_Original", win_name, 0, 1, nothing)
    cv2.createTrackbar("Show_Preprocessing", win_name, 0, 1, nothing)
    cv2.createTrackbar("Show_Binary", win_name, 0, 1, nothing)
    cv2.createTrackbar("Show_Cleaned", win_name, 0, 1, nothing)
    cv2.createTrackbar("Show_Morphology", win_name, 0, 1, nothing)
    cv2.createTrackbar("Show_Holes_Only", win_name, 1, 1, nothing)
    
    # 보간 관련 트랙바
    cv2.createTrackbar("Enable_Interpolation", win_name, 1, 1, nothing)
    cv2.createTrackbar("Grid_Basic_Interpolation", win_name, 1, 1, nothing)
    cv2.createTrackbar("Grid_Smart_Interpolation", win_name, 0, 1, nothing)
    cv2.createTrackbar("Statistical_Grid_Interpolation", win_name, 1, 1, nothing)
    cv2.createTrackbar("Row_Eps", win_name, 15, 50, nothing)
    cv2.createTrackbar("Col_Eps", win_name, 15, 50, nothing)
    cv2.createTrackbar("Min_Samples", win_name, 2, 10, nothing)
    cv2.createTrackbar("Duplicate_Threshold", win_name, 10, 30, nothing)
    cv2.createTrackbar("Min_Coverage_Ratio", win_name, 30, 100, nothing)  # 0.3 = 30%
    
    # OpenCV 버전 출력
    print(f"OpenCV version: {cv2.__version__}")
    
    print("조작법:")
    print("- 'q': 종료")
    print("- 'n': 다음 이미지")
    print("- 's': 현재 결과 저장")
    print("- 트랙바로 파라미터 조정 가능")
    print("- Show_* 트랙바로 다양한 단계 시각화 가능")
    print("- 보간 관련 트랙바:")
    print("  * Enable_Interpolation: 보간 기능 전체 on/off")
    print("  * Grid_Basic_Interpolation: 기본 격자 보간 (모든 교차점 보간)")
    print("  * Grid_Smart_Interpolation: 스마트 격자 보간 (밀도 고려)")
    print("  * Statistical_Grid_Interpolation: 통계적 격자 보간 (간격 분석)")
    print("  * Row_Eps/Col_Eps: DBSCAN 클러스터링 거리 임계값")
    print("  * Min_Samples: 클러스터 형성 최소 샘플 수")
    print("  * Min_Coverage_Ratio: 스마트 보간 최소 밀도 비율 (0.3 = 30%)")
    print("  * Duplicate_Threshold: 중복 제거 거리 임계값")
    
    while True:
        # 트랙바 값 읽기
        hole_r = cv2.getTrackbarPos("hole_radius", win_name)
        hole_area = cv2.getTrackbarPos("hole_area_thresh", win_name)
        min_hole_area = cv2.getTrackbarPos("min_hole_area", win_name)
        max_hole_area = cv2.getTrackbarPos("max_hole_area", win_name)
        min_circularity = cv2.getTrackbarPos("min_circularity", win_name) / 100.0
        max_circularity = cv2.getTrackbarPos("max_circularity", win_name) / 100.0
        min_square_ratio = cv2.getTrackbarPos("min_square_ratio", win_name) / 100.0
        max_square_ratio = cv2.getTrackbarPos("max_square_ratio", win_name) / 100.0
        use_otsu = cv2.getTrackbarPos("Use_Otsu", win_name) == 1
        apply_sharpen = cv2.getTrackbarPos("Apply_Sharpen", win_name) == 1
        apply_smooth = cv2.getTrackbarPos("Apply_Smooth", win_name) == 1
        clahe_clip = cv2.getTrackbarPos("CLAHE_clip", win_name)
        blk = cv2.getTrackbarPos("block_size", win_name)
        show_original = cv2.getTrackbarPos("Show_Original", win_name) == 1
        show_preprocessing = cv2.getTrackbarPos("Show_Preprocessing", win_name) == 1
        show_binary = cv2.getTrackbarPos("Show_Binary", win_name) == 1
        show_cleaned = cv2.getTrackbarPos("Show_Cleaned", win_name) == 1
        show_morphology = cv2.getTrackbarPos("Show_Morphology", win_name) == 1
        show_holes_only = cv2.getTrackbarPos("Show_Holes_Only", win_name) == 1
        
        # 보간 관련 파라미터
        enable_interpolation = cv2.getTrackbarPos("Enable_Interpolation", win_name) == 1
        grid_basic_interpolation = cv2.getTrackbarPos("Grid_Basic_Interpolation", win_name) == 1
        grid_smart_interpolation = cv2.getTrackbarPos("Grid_Smart_Interpolation", win_name) == 1
        statistical_interp_enabled = cv2.getTrackbarPos("Statistical_Grid_Interpolation", win_name) == 1

        row_eps = cv2.getTrackbarPos("Row_Eps", win_name)
        col_eps = cv2.getTrackbarPos("Col_Eps", win_name)
        min_samples = max(cv2.getTrackbarPos("Min_Samples", win_name), 1)
        duplicate_threshold = cv2.getTrackbarPos("Duplicate_Threshold", win_name)
        min_coverage_ratio = cv2.getTrackbarPos("Min_Coverage_Ratio", win_name) / 100.0
        
        if blk < 3: blk = 3
        if blk % 2 == 0: blk += 1
        
        try:
            # 1) 향상된 전처리 (원본 코드와 동일)
            img_clean = remove_red_blue(img)
            gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
            processed = advanced_preprocessing(gray, apply_sharpen, apply_smooth, clahe_clip)
            
            # 2) 이진화 (원본 코드와 동일)
            if use_otsu:
                _, thr = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask_inv = cv2.bitwise_not(thr)
            else:
                thr = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, blk, 5)
                mask_inv = cv2.bitwise_not(thr)
            
            # 3) 작은 노이즈 제거 (안전한 findContours 사용)
            conts, _ = safe_find_contours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            small_mask = np.zeros_like(mask_inv)
            for cnt in conts:
                if cv2.contourArea(cnt) < hole_area:
                    cv2.drawContours(small_mask, [cnt], -1, 255, -1)
            mask_small_removed = cv2.bitwise_and(mask_inv, cv2.bitwise_not(small_mask))
            
            # 4) 구멍 검출 및 제거
            mask_no_holes, hole_centers, removed_regions, hole_info = remove_hole_with_visualization(
                mask_small_removed, img_clean, hole_r, min_hole_area, max_hole_area,
                min_circularity, max_circularity, min_square_ratio, max_square_ratio)
            
            # 6) 보간 처리
            interpolated_points = []
            if enable_interpolation and hole_centers:
                try:
                    # 새로운 격자 기반 클러스터링
                    row_y_centers, col_x_centers, grid_map, num_rows, num_cols = cluster_rows_columns(
                        hole_centers, row_eps, col_eps, min_samples)
                    
                    print(f"Grid Analysis: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} total positions")
                    print(f"Existing holes: {len(hole_centers)}, Missing: {num_rows * num_cols - len(hole_centers)}")
                    
                    # 기본 격자 보간
                    if grid_basic_interpolation and row_y_centers and col_x_centers:
                        basic_points = grid_based_interpolation(row_y_centers, col_x_centers, grid_map, num_rows, num_cols)
                        interpolated_points.extend(basic_points)
                        print(f"Basic grid interpolation added: {len(basic_points)} points")
                    
                    # 스마트 격자 보간
                    if grid_smart_interpolation and row_y_centers and col_x_centers:
                        smart_points = smart_grid_interpolation(row_y_centers, col_x_centers, grid_map, 
                                                              num_rows, num_cols, min_coverage_ratio)
                        interpolated_points.extend(smart_points)
                        print(f"Smart grid interpolation added: {len(smart_points)} points")
                    
                    # 통계적 격자 보간
                    if statistical_interp_enabled and row_y_centers and col_x_centers:
                        stat_points = statistical_grid_interpolation(row_y_centers, col_x_centers, grid_map, 
                                                                   num_rows, num_cols)
                        interpolated_points.extend(stat_points)
                        print(f"Statistical grid interpolation added: {len(stat_points)} points")
                    
                    # 중복 제거
                    interpolated_points = remove_duplicate_points(interpolated_points, duplicate_threshold)
                    
                    # 기존 점들과 너무 가까운 보간점 제거
                    filtered_interpolated = []
                    for interp_point in interpolated_points:
                        too_close = False
                        for existing_point in hole_centers:
                            if np.hypot(interp_point[0] - existing_point[0], 
                                      interp_point[1] - existing_point[1]) < duplicate_threshold:
                                too_close = True
                                break
                        if not too_close:
                            filtered_interpolated.append(interp_point)
                    
                    interpolated_points = filtered_interpolated
                    print(f"Final interpolated points after filtering: {len(interpolated_points)}")
                    
                except Exception as e:
                    print(f"Interpolation error: {e}")
                    import traceback
                    traceback.print_exc()
                    interpolated_points = []
                    
            # 5) 모폴로지 처리
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(mask_no_holes, cv2.MORPH_OPEN, k, iterations=1)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)
            connected = cv2.dilate(closed, k, iterations=1)
            processed_morph = cv2.erode(connected, k, iterations=1)
            
            # 시각화
            if show_holes_only:
                # 구멍만 강조해서 표시
                display_img = visualize_holes(img, hole_centers, hole_r, hole_info, interpolated_points, "LED Hole Detection")
            elif show_original:
                display_img = img.copy()
                cv2.putText(display_img, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            elif show_preprocessing:
                display_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                cv2.putText(display_img, "After Preprocessing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            elif show_binary:
                display_img = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                cv2.putText(display_img, "Binary Mask (Inverted)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            elif show_cleaned:
                display_img = cv2.cvtColor(mask_small_removed, cv2.COLOR_GRAY2BGR)
                cv2.putText(display_img, "After Noise Removal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            elif show_morphology:
                display_img = cv2.cvtColor(processed_morph, cv2.COLOR_GRAY2BGR)
                cv2.putText(display_img, "After Morphology", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                display_img = visualize_holes(img, hole_centers, hole_r, hole_info, interpolated_points, "LED Hole Detection")
            
            # 전처리 단계별 시각화 창 생성
            preprocessing_steps = {
                "1. Original": img,
                "2. Red/Blue Removed": img_clean,
                "3. Preprocessed": cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR),
                "4. Binary": cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR),
                "5. Noise Removed": cv2.cvtColor(mask_small_removed, cv2.COLOR_GRAY2BGR),
                "6. After Morphology": cv2.cvtColor(processed_morph, cv2.COLOR_GRAY2BGR),
                "7. Holes Detected": visualize_holes(img, hole_centers, hole_r, hole_info, interpolated_points, "Final Result")
            }
            
            # 전처리 단계들을 2x4 그리드로 배치
            resize_w, resize_h = w // 3, h // 3
            step_images = []
            
            for title, step_img in preprocessing_steps.items():
                # 이미지 크기 조정
                resized = cv2.resize(step_img, (resize_w, resize_h))
                
                # 제목 추가
                labeled = resized.copy()
                cv2.putText(labeled, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                step_images.append(labeled)
            
            # 2x4 그리드 생성 (7개 이미지 + 1개 빈 공간)
            if len(step_images) == 7:
                # 빈 이미지 추가
                empty_img = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
                cv2.putText(empty_img, "Processing Steps", (10, resize_h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                step_images.append(empty_img)
            
            # 2x4 그리드 배치
            top_row = np.hstack(step_images[0:4])
            bottom_row = np.hstack(step_images[4:8])
            preprocessing_grid = np.vstack((top_row, bottom_row))
            
            # 구멍 정보를 별도 창에 표시
            info_img = np.zeros((450, 700, 3), dtype=np.uint8)
            y_pos = 30
            
            # 제목
            cv2.putText(info_img, f"File: {image_files[idx]}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 40
            
            # 파라미터 정보
            cv2.putText(info_img, f"Parameters:", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
            cv2.putText(info_img, f"  Hole Radius: {hole_r}px", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Area Range: {min_hole_area}-{max_hole_area}px", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Circularity: {min_circularity:.2f}-{max_circularity:.2f}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Square Ratio: {min_square_ratio:.2f}-{max_square_ratio:.2f}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 30
            
            # 보간 정보
            cv2.putText(info_img, f"Interpolation:", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y_pos += 25
            cv2.putText(info_img, f"  Enabled: {enable_interpolation}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Basic Grid: {grid_basic_interpolation}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Smart Grid: {grid_smart_interpolation}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Statistical Grid: {statistical_grid_interpolation}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Coverage Ratio: {min_coverage_ratio:.2f}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(info_img, f"  Row Eps: {row_eps}, Col Eps: {col_eps}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 30
            
            # 구멍 검출 결과
            cv2.putText(info_img, f"Results:", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
            cv2.putText(info_img, f"  Detected Holes: {len(hole_centers)}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20
            if enable_interpolation:
                cv2.putText(info_img, f"  Interpolated Points: {len(interpolated_points)}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                y_pos += 20
                cv2.putText(info_img, f"  Total Points: {len(hole_centers) + len(interpolated_points)}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_pos += 20
            
            # 상세 정보 (처음 몇 개만)
            max_display = 8
            if len(hole_centers) > 0:
                cv2.putText(info_img, f"Detected Holes (showing first {min(max_display, len(hole_centers))}):", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 20
                
                for i, (cx, cy) in enumerate(hole_centers[:max_display]):
                    if y_pos > 420:
                        break
                    text = f"  H{i+1}: ({int(cx)}, {int(cy)})"
                    cv2.putText(info_img, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_pos += 16
                    
                if len(hole_centers) > max_display:
                    cv2.putText(info_img, f"  ... and {len(hole_centers) - max_display} more", (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # 이미지 표시
            cv2.imshow(win_name, display_img)
            cv2.imshow("Hole Detection Info", info_img)
            led_hull, led_centroid, led_body_mask = detect_led_body(img_clean)
            square_size = hole_r * 2  # 원하는 크기로 조정 가능
            covered_mask = cover_holes_excluding_led(mask_small_removed,
                                            hole_centers,
                                            interpolated_points,
                                            led_hull,
                                            square_size)
            cv2.imshow("Covered Holes Mask", covered_mask)


            #cv2.imshow("Preprocessing Steps (2x4)", preprocessing_grid)
            
            # 창 크기 설정
            #cv2.namedWindow("Preprocessing Steps (2x4)", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("Preprocessing Steps (2x4)", resize_w * 10, resize_h * 5)

            # 창 이름 정의
            window_name = "Preprocessing Steps (2x4)"

            preprocessing_grid_resized = cv2.resize(
            preprocessing_grid, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST
            )

            # 리사이즈 가능한 창으로 설정
            cv2.namedWindow("Preprocessing Steps (2x4)", cv2.WINDOW_NORMAL)
            cv2.imshow("Preprocessing Steps (2x4)", preprocessing_grid_resized)

            
            # 콘솔 출력
            print(f"\n=== Hole Detection Results ===")
            print(f"Image: {image_files[idx]}")
            print(f"Holes detected: {len(hole_centers)}")
            print(f"Hole coordinates:")
            for i, (cx, cy) in enumerate(hole_centers):
                print(f"  Hole {i+1}: ({int(cx)}, {int(cy)})")
            print(f"Removed regions:")
            for i, (x0, y0, x1, y1) in enumerate(removed_regions):
                print(f"  Region {i+1}: ({x0}, {y0}) to ({x1}, {y1})")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()  # 상세한 오류 정보 출력
            
            # 에러 발생 시 원본 이미지라도 표시
            try:
                display_img = img.copy()
                cv2.putText(display_img, f"Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(win_name, display_img)
                
                # 빈 정보 창 표시
                info_img = np.zeros((450, 700, 3), dtype=np.uint8)
                cv2.putText(info_img, "Error occurred during processing", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(info_img, f"Error: {str(e)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Hole Detection Info", info_img)
            except:
                pass  # 최악의 경우에도 프로그램이 멈추지 않도록
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(image_files)
            new_img = cv2.imread(image_files[idx])
            if new_img is None:
                print(f"{image_files[idx]}를 읽을 수 없습니다.")
                # 다음 이미지를 시도
                for i in range(len(image_files)):
                    test_idx = (idx + i) % len(image_files)
                    test_img = cv2.imread(image_files[test_idx])
                    if test_img is not None:
                        idx = test_idx
                        img = test_img
                        h, w = img.shape[:2]
                        cv2.resizeWindow(win_name, w, h)
                        print(f"Switched to image: {image_files[idx]}")
                        break
                else:
                    print("읽을 수 있는 이미지가 없습니다.")
            else:
                img = new_img
                h, w = img.shape[:2]
                cv2.resizeWindow(win_name, w, h)
                print(f"Switched to image: {image_files[idx]}")
        elif key == ord('s'):
            # 현재 결과 저장
            try:
                if 'display_img' in locals():
                    save_name = f"hole_detection_result_{image_files[idx]}"
                    cv2.imwrite(save_name, display_img)
                    print(f"Result saved as: {save_name}")
                else:
                    print("No image to save")
            except Exception as e:
                print(f"Error saving image: {e}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()