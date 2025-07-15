import cv2
import numpy as np
import math
from collections import deque

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Using simple clustering instead.")


def safe_find_contours(image, mode, method):
    """OpenCV 버전에 상관없이 안전하게 contours를 찾는 함수"""
    result = cv2.findContours(image, mode, method)
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result
    return contours, hierarchy


class CapEndpointDetector:
    def __init__(self,
                 min_length=30,
                 clahe_clip=2.0,
                 min_hole_area=50,
                 max_hole_area=500,
                 direction_threshold=0.6,
                 body_proximity=30,
                 hole_distance=25,
                 led_body_exclusion=50,
                 use_otsu=True,
                 enable_interpolation=True,
                 row_col_eps=15,
                 use_symmetry=True):
        # 초기 파라미터 설정
        self.min_length = min_length
        self.clahe_clip = clahe_clip
        self.min_hole_area = min_hole_area
        self.max_hole_area = max_hole_area
        self.direction_threshold = direction_threshold
        self.body_proximity = body_proximity
        self.hole_distance = hole_distance
        self.led_body_exclusion = led_body_exclusion
        self.use_otsu = use_otsu
        self.enable_interpolation = enable_interpolation
        self.row_col_eps = row_col_eps
        self.use_symmetry = use_symmetry

    # --- 전처리 및 유틸리티 함수들 ---
    def remove_red_blue(self, img):
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

    def apply_clahe(self, gray, clipLimit=None):
        """CLAHE 적용 함수"""
        if clipLimit is None:
            clipLimit = self.clahe_clip
        clahe = cv2.createCLAHE(clipLimit=clipLimit)
        return clahe.apply(gray)

    def safe_find_contours(self, image, mode, method):
        """OpenCV 버전에 상관없이 안전하게 contours를 찾는 함수"""
        res = cv2.findContours(image, mode, method)
        if len(res) == 3:
            _, contours, hierarchy = res
        else:
            contours, hierarchy = res
        return contours, hierarchy

    def simple_dbscan(self, points, eps, min_samples):
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

    def detect_square_holes(self, binary, min_area=None, max_area=None):
        """정사각형 형태의 구멍을 검출하는 함수"""
        if min_area is None:
            min_area = self.min_hole_area
        if max_area is None:
            max_area = self.max_hole_area
            
        contours, _ = self.safe_find_contours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        info = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            # 원형도 체크
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3 or circularity > 0.9:
                continue
            
            # 종횡비 체크
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            ratio = float(w) / h
            if ratio < 0.6 or ratio > 1.4:
                continue
            
            # 볼록성 체크
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            convexity = area / hull_area
            if convexity < 0.7:
                continue
            
            # 중심점 계산
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centers.append((cx, cy))
                info.append({
                    'rect': (x, y, w, h), 
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': ratio,
                    'convexity': convexity,
                    'contour': cnt
                })
        
        return centers, info

    def remove_duplicate_points(self, points, threshold=10):
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

    def cluster_rows_columns(self, hole_centers, row_eps=None, col_eps=None, min_samples=2):
        """전체 좌표에서 행/열 기준점들을 찾고 격자 정보를 생성"""
        if not hole_centers:
            return [], [], {}, 0, 0
        
        if row_eps is None:
            row_eps = self.row_col_eps
        if col_eps is None:
            col_eps = self.row_col_eps
        
        points = np.array(hole_centers)
        
        # Y좌표 기준으로 행 클러스터링
        y_coords = points[:, 1].reshape(-1, 1)
        if SKLEARN_AVAILABLE:
            row_clustering = DBSCAN(eps=row_eps, min_samples=min_samples).fit(y_coords)
        else:
            row_clustering = self.simple_dbscan(y_coords, row_eps, min_samples)
        
        # X좌표 기준으로 열 클러스터링
        x_coords = points[:, 0].reshape(-1, 1)
        if SKLEARN_AVAILABLE:
            col_clustering = DBSCAN(eps=col_eps, min_samples=min_samples).fit(x_coords)
        else:
            col_clustering = self.simple_dbscan(x_coords, col_eps, min_samples)
        
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
            if row_y_centers and col_x_centers:
                closest_row_idx = min(range(len(row_y_centers)), 
                                     key=lambda i: abs(row_y_centers[i] - cy))
                closest_col_idx = min(range(len(col_x_centers)), 
                                     key=lambda i: abs(col_x_centers[i] - cx))
                
                if (abs(row_y_centers[closest_row_idx] - cy) <= row_eps and 
                    abs(col_x_centers[closest_col_idx] - cx) <= col_eps):
                    grid_map[(closest_row_idx, closest_col_idx)] = (cx, cy)
        
        return row_y_centers, col_x_centers, grid_map, len(row_y_centers), len(col_x_centers)

    def grid_based_interpolation(self, row_y_centers, col_x_centers, grid_map, num_rows, num_cols):
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

    def detect_lines_lsd(self, img_gray, min_length=None):
        """LSD를 사용하여 선을 검출하고 최소 길이 필터링"""
        if min_length is None:
            min_length = self.min_length
            
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



    def detect_capacitor_body(self, img, min_hull_area=350):
        """Capacitor body detection using Otsu and convex hull"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask_otsu = cv2.threshold(gray,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) if self.use_otsu else cv2.threshold(gray,127,255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_otsu)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask_clean = cv2.morphologyEx(mask_inv,cv2.MORPH_CLOSE,kernel,iterations=3)
        mask_clean = cv2.morphologyEx(mask_clean,cv2.MORPH_OPEN,kernel,iterations=2)
        contours, _ = safe_find_contours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None, mask_clean
        kept=[]
        for cnt in contours:
            hull=cv2.convexHull(cnt)
            if cv2.contourArea(hull)>=min_hull_area:
                kept.append(hull)
        if not kept: return None, None, mask_clean
        merged=cv2.convexHull(np.vstack(kept))
        M=cv2.moments(merged)
        centroid=(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"]!=0 else None
        return merged, centroid, mask_clean

    def merge_lines(self, lines, angle_thresh_deg=10, dist_thresh=20):
        """비슷한 각도와 가까운 거리에 있는 선분들을 병합합니다."""
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

        # BFS를 사용하여 연결된 그룹 찾기
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
            
            if best_pair:
                p_start, p_end = best_pair
                merged_lines.append((int(p_start[0]), int(p_start[1]), int(p_end[0]), int(p_end[1])))

        return merged_lines

    def detect_lines_to_body(self, lines, centroid, hull, direction_threshold=None, 
                            body_proximity_threshold=None, min_line_length=None):
        """LED 몸체로 향하는 선들을 검출합니다."""
        if not lines or not centroid:
            return []
        
        if direction_threshold is None:
            direction_threshold = self.direction_threshold
        if body_proximity_threshold is None:
            body_proximity_threshold = self.body_proximity
        if min_line_length is None:
            min_line_length = self.min_length
        
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
            
            # LED 몸체와의 거리 계산
            body_dist1 = float('inf')
            body_dist2 = float('inf')
            min_body_distance = float('inf')
            
            if hull is not None:
                body_dist1 = abs(cv2.pointPolygonTest(hull, (float(x1), float(y1)), True))
                body_dist2 = abs(cv2.pointPolygonTest(hull, (float(x2), float(y2)), True))
                
                # 선 전체에서 LED 몸체까지의 최소 거리 계산
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
                    'far_body_distance': far_body_distance,
                    'min_body_distance': min_body_distance,
                    'length': line_length,
                    'direction_score': max(dot1, dot2)
                })
        
        # LED 몸체에서 가장 멀리 있는 선들을 우선적으로 선택
        body_lines.sort(key=lambda x: (
            x['far_body_distance'],
            x['min_body_distance'],
            x['far_distance']
        ), reverse=True)
        
        return body_lines

    def select_best_body_lines(self, body_lines, max_lines=2, min_body_distance=20):
        """LED 몸체에서 충분히 멀리 있는 최상위 선들만 선택"""
        selected_lines = []
        
        for line_info in body_lines:
            if line_info['far_body_distance'] >= min_body_distance:
                selected_lines.append(line_info)
                
                if len(selected_lines) >= max_lines:
                    break
        
        return selected_lines

    def find_holes_near_line_endpoints(self, body_lines, hole_centers, hull, 
                                      max_distance=None, body_exclusion_distance=None):
        """LED 몸체로 향하는 선들의 먼 쪽 끝점 근처에 있는 구멍들을 찾습니다."""
        if max_distance is None:
            max_distance = self.hole_distance
        if body_exclusion_distance is None:
            body_exclusion_distance = self.led_body_exclusion
            
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
                score = (line_info['length'] * 2 +
                        line_info['direction_score'] * 100 +
                        (max_distance - best_hole['distance_to_endpoint']) * 3 +
                        line_info['far_distance'] * 0.5)
                
                lead_candidates.append({
                    'hole_pos': best_hole['hole_pos'],
                    'line_info': line_info,
                    'distance_to_endpoint': best_hole['distance_to_endpoint'],
                    'score': score
                })
        
        # 점수 순으로 정렬
        lead_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return lead_candidates, excluded_holes

    def select_best_symmetric_leads(self, lead_candidates, centroid, max_leads=2):
        """리드 후보들 중에서 대칭성을 고려하여 최적의 리드들을 선택합니다."""
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
                dist1 = math.hypot(h1x - cx, h1y - cy)
                dist2 = math.hypot(h2x - cx, h2y - cy)
                distance_similarity = 1.0 / (1.0 + abs(dist1 - dist2))
                
                angle1 = math.atan2(h1y - cy, h1x - cx)
                angle2 = math.atan2(h2y - cy, h2x - cx)
                angle_diff = abs(abs(angle1 - angle2) - math.pi)
                angle_symmetry = 1.0 / (1.0 + angle_diff)
                
                individual_scores = candidate1['score'] + candidate2['score']
                
                symmetry_score = (distance_similarity * 50 + 
                                angle_symmetry * 100 + 
                                individual_scores * 0.1)
                
                if symmetry_score > best_score:
                    best_score = symmetry_score
                    best_pair = [candidate1, candidate2]
        
        return best_pair if best_pair else lead_candidates[:max_leads]

    def find_best_symmetric_endpoint(self, existing_lead, candidates, centroid):
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

    def find_additional_leads_from_lines(self, body_lines, selected_leads, centroid, hull, 
                                       min_distance_from_body=None, max_total_leads=2):
        """이미 선택된 리드가 부족할 때, Body Lines의 끝점에서 추가 리드 찾기"""
        if len(selected_leads) >= max_total_leads:
            return selected_leads
        
        if min_distance_from_body is None:
            min_distance_from_body = self.hole_distance
        
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
                'source': 'line_endpoint'
            })
        
        # 거리 순으로 정렬 (멀수록 우선)
        line_endpoints.sort(key=lambda x: x['distance_from_center'], reverse=True)
        
        # 필요한 만큼 추가
        additional_leads = []
        needed = max_total_leads - len(selected_leads)
        
        for i in range(min(needed, len(line_endpoints))):
            endpoint = line_endpoints[i]
            additional_leads.append({
                'hole_pos': endpoint['position'],
                'line_info': endpoint['line_info'],
                'distance_to_endpoint': 0,
                'score': endpoint['distance_from_center'] * 2,
                'source': 'line_endpoint'
            })
        
        # 대칭성 고려하여 최종 선택
        if len(selected_leads) == 1 and len(additional_leads) > 0:
            best_symmetric = self.find_best_symmetric_endpoint(selected_leads[0], additional_leads, centroid)
            if best_symmetric:
                return selected_leads + [best_symmetric]
        
        return selected_leads + additional_leads

    def extract(self, img, box, holes=None):
        """
        주어진 이미지(img)와 bounding box(box)에 대해
        LED 리드 위치 두 개를 항상 리스트 형태로 반환합니다.
        """
        # 1) ROI 추출 및 전처리
        x1, y1, x2, y2 = box
        roi = img[y1:y2, x1:x2]
        clean = self.remove_red_blue(roi)
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        proc = self.apply_clahe(gray, clipLimit=self.clahe_clip)

        # 2) 이진화 및 구멍 검출
        if self.use_otsu:
            _, binary = cv2.threshold(proc, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(proc, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 15, 5)
        mask_inv = cv2.bitwise_not(binary)

        hole_centers, hole_info = self.detect_square_holes(
            mask_inv, self.min_hole_area, self.max_hole_area)

        # 3) 구멍 보간 (선택적)
        interpolated = []
        if self.enable_interpolation and hole_centers:
            rows, cols, grid_map, n_rows, n_cols = self.cluster_rows_columns(
                hole_centers, self.row_col_eps, self.row_col_eps, 2)
            if rows and cols:
                interpolated = self.grid_based_interpolation(
                    rows, cols, grid_map, n_rows, n_cols)
                interpolated = self.remove_duplicate_points(
                    interpolated, self.row_col_eps)
                interpolated = [pt for pt in interpolated if all(
                    math.hypot(pt[0] - hc[0], pt[1] - hc[1]) > self.row_col_eps 
                    for hc in hole_centers)]
        all_holes = hole_centers + interpolated

        # 4) LED 몸체 및 선 검출
        hull, centroid, body_mask = self.detect_capacitor_body(roi)
        lines1 = self.detect_lines_lsd(proc, self.min_length)
        lines2 = self.detect_lines_lsd(binary, self.min_length)
        merged = self.merge_lines(lines1 + lines2)
        body_lines = self.detect_lines_to_body(
            merged, centroid, hull,
            self.direction_threshold, self.body_proximity, self.min_length)

        # 5) 리드 후보 생성 및 선택
        body_lines = self.select_best_body_lines(body_lines, max_lines=2, min_body_distance=30)
        lead_candidates, excluded_holes = self.find_holes_near_line_endpoints(
            body_lines, all_holes, hull,
            self.hole_distance, self.led_body_exclusion)
        
        if self.use_symmetry:
            selected_leads = self.select_best_symmetric_leads(
                lead_candidates, centroid, 2)
        else:
            selected_leads = lead_candidates[:2]
            
        if len(selected_leads) < 2:
            selected_leads = self.find_additional_leads_from_lines(
                body_lines, selected_leads, centroid, hull,
                self.hole_distance, 2)

        # 6) 결과 위치 조정 및 반환
        positions = [lead['hole_pos'] for lead in selected_leads]
        positions = [(int(x + x1), int(y + y1)) for x, y in positions]
        return positions


if __name__ == '__main__':
    # VS Code에서 바로 실행 시 이 부분만 수정하세요
    # 테스트할 이미지 경로와 박스 좌표를 직접 지정합니다.
    IMAGE_PATH = r'D:\Hyuntak\lab\AR_circuit_tutor\breadboard_project\temp\led25.JPG'  # <-- 여기에 테스트 이미지 경로
    BOX = None  # 전체 이미지 사용: None 또는 (x1, y1, x2, y2)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"이미지 로드 실패: {IMAGE_PATH}")
        exit(1)

    h, w = img.shape[:2]
    box = BOX if BOX else (0, 0, w, h)

    detector = LedEndpointDetector()
    endpoints = detector.extract(img, box)

    # 결과 시각화
    for idx, (x, y) in enumerate(endpoints):
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(idx+1), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('LED Endpoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()