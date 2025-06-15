import cv2
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict
import os

def radial_scan_for_leads(img, centroid, hull, num_rays=72, min_distance=50, debug=False):
    """
    중심에서 방사형으로 스캔하여 리드 특징 찾기
    """
    if not centroid:
        return []
    
    cx, cy = centroid
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # 디버그용 시각화
    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, (int(cx), int(cy)), 5, (0, 255, 255), -1)
    
    lead_candidates = []
    ray_profiles = []
    
    for i in range(num_rays):
        angle = i * 2 * np.pi / num_rays
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # LED 몸체 밖에서 시작 지점 찾기
        start_dist = 20
        if hull is not None:
            for d in range(10, 100, 2):
                px = cx + d * dx
                py = cy + d * dy
                if cv2.pointPolygonTest(hull, (float(px), float(py)), False) < 0:
                    start_dist = d + 5
                    break
        
        # 방사선을 따라 스캔
        intensity_profile = []
        gradient_profile = []
        positions = []
        
        max_dist = int(min(
            abs(cx / dx) if dx != 0 else float('inf'),
            abs(cy / dy) if dy != 0 else float('inf'),
            abs((width - cx) / dx) if dx != 0 else float('inf'),
            abs((height - cy) / dy) if dy != 0 else float('inf')
        ))
        
        for dist in range(start_dist, min(max_dist, 900)):
            x = int(cx + dist * dx)
            y = int(cy + dist * dy)
            
            if 0 <= x < width and 0 <= y < height:
                intensity = gray[y, x]
                intensity_profile.append(intensity)
                positions.append((x, y))
                
                if len(intensity_profile) > 1:
                    gradient = intensity - intensity_profile[-2]
                    gradient_profile.append(gradient)
            else:
                break
        
        if len(intensity_profile) < 20:
            continue
        
        # 프로파일 분석
        intensity_array = np.array(intensity_profile)
        gradient_array = np.array(gradient_profile) if gradient_profile else np.array([])
        
        # 급격한 어두워짐 찾기 (구멍)
        if len(gradient_array) > 0:
            dark_peaks, properties = find_peaks(-gradient_array, 
                                              height=15, 
                                              distance=10)
            
            for peak_idx in dark_peaks:
                if peak_idx + start_dist > min_distance:
                    hole_width = estimate_hole_width(intensity_array, peak_idx)
                    
                    if 3 <= hole_width <= 20:
                        pos = positions[peak_idx]
                        lead_candidates.append({
                            'pos': pos,
                            'angle': angle,
                            'distance': peak_idx + start_dist,
                            'confidence': min(1.0, abs(gradient_array[peak_idx]) / 100),
                            'method': 'radial_hole',
                            'hole_width': hole_width
                        })
        
        # 연속적인 어두운 선 찾기 (리드 와이어)
        dark_threshold = np.mean(intensity_array) - np.std(intensity_array)
        dark_regions = find_continuous_dark_regions(intensity_array, dark_threshold, min_length=15)
        
        for start_idx, end_idx in dark_regions:
            if start_idx + start_dist > min_distance:
                end_pos = positions[min(end_idx, len(positions)-1)]
                lead_candidates.append({
                    'pos': end_pos,
                    'angle': angle,
                    'distance': end_idx + start_dist,
                    'confidence': 0.7,
                    'method': 'radial_wire',
                    'wire_length': end_idx - start_idx
                })
        
        # 디버그 시각화
        if debug and positions:
            cv2.line(debug_img, 
                    (int(cx + start_dist * dx), int(cy + start_dist * dy)),
                    positions[-1], (0, 255, 0), 1)
            
            for candidate in lead_candidates[-2:]:
                if candidate['angle'] == angle:
                    cv2.circle(debug_img, candidate['pos'], 3, (0, 0, 255), -1)
        
        ray_profiles.append({
            'angle': angle,
            'profile': intensity_profile,
            'positions': positions
        })
    
    # 근접한 후보들 병합
    merged_candidates = merge_nearby_candidates(lead_candidates, threshold=10)
    
    # 대칭성 기반 필터링
    symmetric_candidates = filter_by_symmetry(merged_candidates, centroid)
    
    if debug:
        return symmetric_candidates, debug_img, ray_profiles
    
    return symmetric_candidates

def estimate_hole_width(intensity_profile, center_idx):
    """어두운 영역의 너비 추정"""
    if center_idx >= len(intensity_profile):
        return 0
    
    center_val = intensity_profile[center_idx]
    threshold = center_val + 20
    
    left = center_idx
    while left > 0 and intensity_profile[left] < threshold:
        left -= 1
    
    right = center_idx
    while right < len(intensity_profile) - 1 and intensity_profile[right] < threshold:
        right += 1
    
    return right - left

def find_continuous_dark_regions(profile, threshold, min_length=10):
    """연속적인 어두운 영역 찾기"""
    regions = []
    start = None
    
    for i, val in enumerate(profile):
        if val < threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_length:
                regions.append((start, i))
            start = None
    
    if start is not None and len(profile) - start >= min_length:
        regions.append((start, len(profile)))
    
    return regions

def merge_nearby_candidates(candidates, threshold=10):
    """근접한 후보들을 병합"""
    if not candidates:
        return []
    
    groups = defaultdict(list)
    used = set()
    
    for i, cand1 in enumerate(candidates):
        if i in used:
            continue
        
        group = [cand1]
        used.add(i)
        
        for j, cand2 in enumerate(candidates[i+1:], i+1):
            if j not in used:
                dist = np.hypot(cand1['pos'][0] - cand2['pos'][0],
                              cand1['pos'][1] - cand2['pos'][1])
                if dist < threshold:
                    group.append(cand2)
                    used.add(j)
        
        if group:
            avg_x = np.mean([c['pos'][0] for c in group])
            avg_y = np.mean([c['pos'][1] for c in group])
            max_confidence = max(c['confidence'] for c in group)
            methods = list(set(c['method'] for c in group))
            
            groups[(int(avg_x), int(avg_y))].append({
                'pos': (int(avg_x), int(avg_y)),
                'confidence': max_confidence,
                'method': methods,
                'support': len(group)
            })
    
    return [group[0] for group in groups.values()]

def detect_yellow_hull_and_centroid(img):
    """
    노란색 영역을 HSV 마스크로 추출한 뒤,
    가장 큰 컨투어의 convex hull과 그 중심점을 계산
    """
    # 1) HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2) 노란색 범위 마스크 (확실하지 않음: 환경에 따라 범위 조정 필요)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 3) 노이즈 제거를 위한 모폴로지
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 4) 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    # 5) 면적이 가장 큰 컨투어 선택
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    
    # 6) hull의 모멘트로 중심점 계산
    M = cv2.moments(hull)
    if M["m00"] == 0:
        return hull, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = (cx, cy)
    
    return hull, centroid


def filter_by_symmetry(candidates, centroid):
    """대칭성을 고려하여 후보 필터링"""
    if len(candidates) < 2 or not centroid:
        return candidates
    
    cx, cy = centroid
    scored_pairs = []
    
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            
            angle1 = np.arctan2(c1['pos'][1] - cy, c1['pos'][0] - cx)
            angle2 = np.arctan2(c2['pos'][1] - cy, c2['pos'][0] - cx)
            angle_diff = abs(angle1 - angle2)
            
            symmetry_score = np.cos(angle_diff - np.pi) ** 2
            
            dist1 = np.hypot(c1['pos'][0] - cx, c1['pos'][1] - cy)
            dist2 = np.hypot(c2['pos'][0] - cx, c2['pos'][1] - cy)
            dist_score = 1.0 / (1.0 + abs(dist1 - dist2) / max(dist1, dist2))
            
            total_score = symmetry_score * 0.7 + dist_score * 0.3
            scored_pairs.append((total_score, c1, c2))
    
    if scored_pairs:
        # 유사도(score)만 기준으로 내림차순 정렬
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        best_score, best_c1, best_c2 = scored_pairs[0]
        
        if best_score > 0.6:
            return [best_c1, best_c2]
    
    # 기본: confidence 기준 상위 2개 반환
    return sorted(candidates, key=lambda x: x['confidence'], reverse=True)[:2]

def trace_along_edge(edges, start_point, initial_direction, max_length=200, step_size=2):
    """엣지를 따라 추적"""
    height, width = edges.shape
    trace = [np.array(start_point)]
    current_point = np.array(start_point, dtype=float)
    current_direction = np.array(initial_direction)
    
    search_angles = np.array([-45, -30, -15, 0, 15, 30, 45]) * np.pi / 180
    
    for _ in range(max_length // step_size):
        found_next = False
        best_score = -1
        best_point = None
        best_direction = None
        
        for angle_offset in search_angles:
            cos_a = np.cos(angle_offset)
            sin_a = np.sin(angle_offset)
            
            new_direction = np.array([
                current_direction[0] * cos_a - current_direction[1] * sin_a,
                current_direction[0] * sin_a + current_direction[1] * cos_a
            ])
            
            next_point = current_point + new_direction * step_size
            x, y = int(next_point[0]), int(next_point[1])
            
            if 0 <= x < width and 0 <= y < height:
                x_min = max(0, x-1)
                x_max = min(width, x+2)
                y_min = max(0, y-1)
                y_max = min(height, y+2)
                
                edge_strength = np.sum(edges[y_min:y_max, x_min:x_max]) / 255.0
                direction_score = 1.0 - abs(angle_offset) / (np.pi / 4)
                score = edge_strength * 0.7 + direction_score * 0.3
                
                if score > best_score and edge_strength > 0:
                    best_score = score
                    best_point = next_point
                    best_direction = new_direction
                    found_next = True
        
        if not found_next:
            break
        
        current_point = best_point
        current_direction = best_direction
        trace.append(current_point.copy())
        
        if np.linalg.norm(current_point - trace[0]) > max_length:
            break
    
    return np.array(trace)

def evaluate_trace_quality(trace, edges):
    """추적 경로의 품질 평가"""
    if len(trace) < 2:
        return 0.0
    
    height, width = edges.shape
    
    edge_match_count = 0
    for point in trace:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            if edges[y, x] > 0:
                edge_match_count += 1
    
    edge_score = edge_match_count / len(trace)
    
    straightness = 0.5
    if len(trace) >= 3:
        directions = []
        for i in range(len(trace) - 1):
            d = trace[i+1] - trace[i]
            if np.linalg.norm(d) > 0:
                directions.append(d / np.linalg.norm(d))
        
        if len(directions) > 1:
            angle_changes = []
            for i in range(len(directions) - 1):
                dot = np.clip(np.dot(directions[i], directions[i+1]), -1, 1)
                angle_changes.append(np.arccos(dot))
            
            straightness = 1.0 - np.mean(angle_changes) / np.pi
    
    quality = edge_score * 0.6 + straightness * 0.4
    return quality

def trace_lead_from_body(img, hull, centroid, debug=False):
    """LED 몸체에서 시작하여 엣지를 따라 리드를 추적"""
    if hull is None or centroid is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    edges1 = cv2.Canny(gray, 30, 100)
    edges2 = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    if debug:
        debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [hull], -1, (0, 255, 255), 2)
    
    contour_points = hull.reshape(-1, 2)
    cx, cy = centroid
    
    lead_traces = []
    
    for i in range(0, len(contour_points), 5):
        point = contour_points[i]
        
        direction = point - np.array([cx, cy])
        direction = direction / np.linalg.norm(direction)
        
        start_point = point + direction * 5
        
        trace = trace_along_edge(edges, start_point, direction, 
                               max_length=200, step_size=2)
        
        if len(trace) > 20:
            quality = evaluate_trace_quality(trace, edges)
            
            lead_traces.append({
                'trace': trace,
                'start': trace[0],
                'end': trace[-1],
                'length': len(trace),
                'quality': quality,
                'direction': direction
            })
            
            if debug:
                for j in range(len(trace) - 1):
                    cv2.line(debug_img, 
                           tuple(trace[j].astype(int)), 
                           tuple(trace[j+1].astype(int)), 
                           (0, 255, 0), 2)
                cv2.circle(debug_img, tuple(trace[-1].astype(int)), 5, (0, 0, 255), -1)
    
    lead_traces.sort(key=lambda x: x['quality'] * x['length'], reverse=True)
    
    selected_traces = select_best_traces(lead_traces, centroid)
    
    if debug:
        return selected_traces, debug_img
    
    return selected_traces

def select_best_traces(traces, centroid):
    """최적의 추적 경로 선택"""
    if len(traces) <= 2:
        return traces
    
    cx, cy = centroid
    
    for trace in traces:
        end_x, end_y = trace['end']
        trace['angle'] = np.arctan2(end_y - cy, end_x - cx)
        trace['end_distance'] = np.hypot(end_x - cx, end_y - cy)
    
    best_pair = None
    best_score = -1
    
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            t1, t2 = traces[i], traces[j]
            
            angle_diff = abs(abs(t1['angle'] - t2['angle']) - np.pi)
            angle_score = 1.0 - angle_diff / np.pi
            
            dist_ratio = min(t1['end_distance'], t2['end_distance']) / \
                        max(t1['end_distance'], t2['end_distance'])
            
            quality_score = (t1['quality'] + t2['quality']) / 2
            
            total_score = angle_score * 0.4 + dist_ratio * 0.3 + quality_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_pair = [t1, t2]
    
    if best_pair and best_score > 0.6:
        return best_pair
    
    return sorted(traces, key=lambda x: x['quality'] * x['length'], reverse=True)[:2]

def detect_led_area(img):
    """LED 영역 검출 (간단 버전)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 가장 큰 밝은 영역 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        
        # 중심점
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
        else:
            centroid = None
        
        return hull, centroid
    
    return None, None

def process_led_image(image_path):
    """LED 이미지를 처리하여 리드 검출"""
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # LED 영역 검출
    hull, centroid = detect_led_area(img)
    
    if hull is None or centroid is None:
        print("Error: Cannot detect LED area")
        return
    
    print(f"LED centroid: {centroid}")
    
    # 결과 시각화를 위한 이미지들
    result_radial = img.copy()
    result_edge = img.copy()
    result_combined = img.copy()
    
    hull, centroid = detect_yellow_hull_and_centroid(img)
    if hull is None or centroid is None:
        print("Error: Cannot detect yellow region")
        return

    # 1. 방사형 스캔
    print("\n1. Radial Scanning...")
    radial_candidates, radial_debug, ray_profiles = radial_scan_for_leads(
        img, centroid, hull, num_rays=72, min_distance=30, debug=True
    )
    
    print(f"Found {len(radial_candidates)} lead candidates using radial scan")
    
    # 방사형 스캔 결과 시각화
    cv2.drawContours(result_radial, [hull], -1, (255, 255, 0), 2)
    cv2.circle(result_radial, centroid, 5, (0, 255, 255), -1)
    
    for i, candidate in enumerate(radial_candidates):
        cv2.circle(result_radial, candidate['pos'], 8, (0, 0, 255), -1)
        cv2.putText(result_radial, f"R{i+1}", 
                   (candidate['pos'][0] + 10, candidate['pos'][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.line(result_radial, centroid, candidate['pos'], (0, 255, 0), 1)
        
        print(f"  Lead {i+1}: pos={candidate['pos']}, "
              f"confidence={candidate['confidence']:.2f}, "
              f"method={candidate['method']}")
    
    # 2. 엣지 기반 추적
    print("\n2. Edge-based Tracing...")
    edge_traces, edge_debug = trace_lead_from_body(img, hull, centroid, debug=True)
    
    print(f"Found {len(edge_traces)} lead traces using edge tracking")
    
    # 엣지 추적 결과 시각화
    cv2.drawContours(result_edge, [hull], -1, (255, 255, 0), 2)
    cv2.circle(result_edge, centroid, 5, (0, 255, 255), -1)
    
    for i, trace in enumerate(edge_traces):
        # 추적 경로 그리기
        for j in range(len(trace['trace']) - 1):
            cv2.line(result_edge, 
                    tuple(trace['trace'][j].astype(int)), 
                    tuple(trace['trace'][j+1].astype(int)), 
                    (0, 255, 0), 2)
        
        # 끝점 표시
        end_point = tuple(trace['end'].astype(int))
        cv2.circle(result_edge, end_point, 8, (255, 0, 0), -1)
        cv2.putText(result_edge, f"E{i+1}", 
                   (end_point[0] + 10, end_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        print(f"  Trace {i+1}: end={end_point}, "
              f"length={trace['length']}, "
              f"quality={trace['quality']:.2f}")
    
    # 3. 결합된 결과
    cv2.drawContours(result_combined, [hull], -1, (255, 255, 0), 2)
    cv2.circle(result_combined, centroid, 5, (0, 255, 255), -1)
    
    # 모든 검출 결과 표시
    for i, candidate in enumerate(radial_candidates):
        cv2.circle(result_combined, candidate['pos'], 6, (0, 0, 255), -1)
        cv2.putText(result_combined, "R", 
                   (candidate['pos'][0] - 15, candidate['pos'][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    for i, trace in enumerate(edge_traces):
        end_point = tuple(trace['end'].astype(int))
        cv2.circle(result_combined, end_point, 6, (255, 0, 0), -1)
        cv2.putText(result_combined, "E", 
                   (end_point[0] - 15, end_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 결과 표시
    # 창 크기 조정
    scale = 800 / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    result_radial_resized = cv2.resize(result_radial, (new_width, new_height))
    result_edge_resized = cv2.resize(result_edge, (new_width, new_height))
    result_combined_resized = cv2.resize(result_combined, (new_width, new_height))
    radial_debug_resized = cv2.resize(radial_debug, (new_width, new_height))
    edge_debug_resized = cv2.resize(edge_debug, (new_width, new_height))
    
    # 결과 창 표시
    cv2.imshow('Radial Scan Result', result_radial_resized)
    cv2.imshow('Edge Trace Result', result_edge_resized)
    cv2.imshow('Combined Result', result_combined_resized)
    cv2.imshow('Radial Debug', radial_debug_resized)
    cv2.imshow('Edge Debug', edge_debug_resized)
    
    # 결과 저장
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f'{base_name}_radial_result.jpg', result_radial)
    cv2.imwrite(f'{base_name}_edge_result.jpg', result_edge)
    cv2.imwrite(f'{base_name}_combined_result.jpg', result_combined)
    cv2.imwrite(f'{base_name}_radial_debug.jpg', radial_debug)
    cv2.imwrite(f'{base_name}_edge_debug.jpg', edge_debug)
    
    print(f"\nResults saved as {base_name}_*.jpg")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """메인 함수"""
    # 테스트할 이미지 경로
    test_images = [
        'led_image1.jpg',  # 여기에 실제 이미지 경로 입력
        'led_image2.jpg',
        # 더 많은 이미지 추가 가능
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Processing: {image_path}")
            print(f"{'='*50}")
            
            process_led_image(image_path)
        else:
            print(f"Warning: {image_path} not found")

if __name__ == "__main__":
    # 단일 이미지 테스트
    image_path = 'led1.jpg'  # 여기에 실제 이미지 경로 입력
    
    if os.path.exists(image_path):
        process_led_image(image_path)
    else:
        print(f"Please provide a valid image path. '{image_path}' not found.")
        print("\nUsage: Change 'image_path' variable to your LED image path")