import cv2
import numpy as np
from hole_detector2 import HoleDetector
from skimage.morphology import skeletonize
from scipy import ndimage

# --- 조명 정규화 함수들 ---
def normalize_illumination(img):
    """여러 방법으로 조명 정규화"""
    # LAB 색공간에서 L 채널 정규화
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE를 L 채널에만 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # 병합 후 BGR로 변환
    lab_clahe = cv2.merge([l_clahe, a, b])
    img_normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return img_normalized

def local_contrast_normalization(gray, kernel_size=31):
    """지역적 명암 정규화"""
    # 지역 평균
    local_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # 지역 표준편차
    gray_squared = gray.astype(np.float32) ** 2
    local_mean_squared = cv2.GaussianBlur(gray_squared, (kernel_size, kernel_size), 0)
    local_std = np.sqrt(np.maximum(local_mean_squared - local_mean.astype(np.float32)**2, 0))
    
    # 정규화
    normalized = np.where(local_std > 10, 
                         (gray - local_mean) / (local_std + 1e-8) * 127 + 127,
                         gray)
    
    return np.clip(normalized, 0, 255).astype(np.uint8)

# --- 다중 검출 방법들 ---
def detect_by_morphology_gradient(gray):
    """모폴로지 그래디언트로 엣지 검출"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
    return binary

def detect_by_directional_filter(gray):
    """방향성 필터로 수직/수평 구조 강조"""
    # Sobel 필터
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 방향과 크기 계산
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    # 주요 방향 선택 (수직/수평이 아닌 대각선 방향)
    mask_diagonal = (np.abs(direction) > np.pi/6) & (np.abs(direction) < 5*np.pi/6)
    binary = np.zeros_like(gray)
    binary[mask_diagonal & (magnitude > 50)] = 255
    
    return binary.astype(np.uint8)

def detect_by_ridge_detection(gray):
    """Ridge detection으로 선형 구조 검출"""
    # Hessian 행렬의 고유값으로 ridge 검출
    hxx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=5)
    hyy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=5)
    hxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    
    # 고유값 계산
    trace = hxx + hyy
    det = hxx * hyy - hxy * hxy
    discriminant = trace * trace - 4 * det
    
    # Ridge 강도
    ridge_strength = np.zeros_like(gray, dtype=np.float64)
    mask = discriminant >= 0
    ridge_strength[mask] = np.abs(trace[mask] - np.sqrt(discriminant[mask])) / 2
    
    # 이진화
    _, binary = cv2.threshold(ridge_strength, np.percentile(ridge_strength, 90), 255, cv2.THRESH_BINARY)
    
    return binary.astype(np.uint8)

# --- LED 몸체와 리드 분리 ---
def separate_body_and_leads(img, body_mask):
    """LED 몸체를 제외한 리드 영역 추출"""
    # 몸체 마스크 확장
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    body_dilated = cv2.dilate(body_mask, kernel, iterations=1)
    
    # 전체에서 몸체 제외
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    leads_only = cv2.bitwise_and(binary, cv2.bitwise_not(body_dilated))
    
    return leads_only

# --- 리드 후보 평가 ---
def evaluate_lead_candidate(contour, img_shape, hull_centroid):
    """리드 후보의 점수 계산"""
    score = 0.0
    
    # 1. 길이
    _, _, w, h = cv2.boundingRect(contour)
    length = max(w, h)
    if 20 <= length <= 200:
        score += 0.2
    
    # 2. 직선성
    if len(contour) >= 5:
        line_result = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = line_result.flatten()
        # 컨투어 점들과 직선의 거리
        points = contour.reshape(-1, 2)
        distances = []
        for pt in points:
            d = abs((pt[1] - y) * vx - (pt[0] - x) * vy) / np.sqrt(vx*vx + vy*vy)
            distances.append(d)
        avg_dist = np.mean(distances)
        if avg_dist < 2.0:  # 직선에 가까움
            score += 0.3
    
    # 3. 폭 일관성
    if len(contour) >= 10:
        # 컨투어의 바운딩 박스로 간단히 평가
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(max(w, h)) / max(min(w, h), 1)
        if aspect_ratio > 3:  # 길쭉한 형태
            score += 0.2
    
    # 4. 위치 (hull 중심에서의 방향)
    if hull_centroid is not None:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # 중심에서 멀어지는 방향인지 확인
            dist_to_center = np.sqrt((cx - hull_centroid[0])**2 + (cy - hull_centroid[1])**2)
            if dist_to_center > 30:
                score += 0.3
    
    return score

# --- 메인 검출 함수 ---
def detect_led_leads_robust(img, debug=False):
    """조명 변화에 강인한 LED 리드 검출"""
    h, w = img.shape[:2]
    
    # 1. 조명 정규화
    img_normalized = normalize_illumination(img)
    gray = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY)
    gray_lcn = local_contrast_normalization(gray)
    
    # 2. LED 몸체 검출 (색상 기반)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 빨강, 초록, 노랑 마스크
    masks = []
    # 빨강
    masks.append(cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255])))
    # 초록
    masks.append(cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255])))
    # 노랑
    masks.append(cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255])))
    
    body_mask = np.zeros_like(gray)
    for mask in masks:
        body_mask = cv2.bitwise_or(body_mask, mask)
    
    # 몸체 정제
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Convex hull 계산
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = None
    hull_centroid = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            hull_centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    # 3. 다중 방법으로 리드 검출
    all_masks = []
    
    # 방법 1: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray_lcn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    all_masks.append(('Adaptive', adaptive))
    
    # 방법 2: Otsu threshold
    _, otsu = cv2.threshold(gray_lcn, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    all_masks.append(('Otsu', otsu))
    
    # 방법 3: Morphology gradient
    morph_grad = detect_by_morphology_gradient(gray_lcn)
    all_masks.append(('Morphology', morph_grad))
    
    # 방법 4: Directional filter
    directional = detect_by_directional_filter(gray_lcn)
    all_masks.append(('Directional', directional))
    
    # 방법 5: Ridge detection
    ridge = detect_by_ridge_detection(gray_lcn)
    all_masks.append(('Ridge', ridge))
    
    # 4. 각 방법에서 리드 후보 추출
    all_candidates = []
    
    for method_name, mask in all_masks:
        # 몸체 영역 제거
        mask_no_body = cv2.bitwise_and(mask, cv2.bitwise_not(body_mask))
        
        # 노이즈 제거
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_no_body, cv2.MORPH_OPEN, kernel_small)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_small)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # 너무 작은 것 제외
                continue
            
            # 직선 근사
            if len(cnt) >= 5:
                line_result = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = line_result.flatten()
                
                # 끝점 계산
                t_values = []
                for pt in cnt.reshape(-1, 2):
                    t = (pt[0] - x0) * vx + (pt[1] - y0) * vy
                    t_values.append(float(t))
                
                if t_values:
                    t_min, t_max = min(t_values), max(t_values)
                    p1 = (int(x0 + vx * t_min), int(y0 + vy * t_min))
                    p2 = (int(x0 + vx * t_max), int(y0 + vy * t_max))
                    
                    length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    if length > 20:  # 최소 길이
                        score = evaluate_lead_candidate(cnt, (h, w), hull_centroid)
                        all_candidates.append({
                            'method': method_name,
                            'p1': p1,
                            'p2': p2,
                            'score': score,
                            'contour': cnt
                        })
    
    # 5. 후보들을 클러스터링하고 최적 선택
    final_leads = cluster_and_select_best_leads(all_candidates, hull_centroid)
    
    if debug:
        return final_leads, all_masks, hull, hull_centroid, all_candidates
    else:
        return final_leads, hull, hull_centroid

def cluster_and_select_best_leads(candidates, hull_centroid, distance_threshold=20):
    """유사한 위치의 후보들을 클러스터링하고 최적 리드 선택"""
    if not candidates:
        return []
    
    # 클러스터링
    clusters = []
    used = [False] * len(candidates)
    
    for i, cand_i in enumerate(candidates):
        if used[i]:
            continue
        
        cluster = [cand_i]
        used[i] = True
        
        # 유사한 위치의 후보들을 같은 클러스터로
        for j, cand_j in enumerate(candidates[i+1:], i+1):
            if used[j]:
                continue
            
            # 두 선분의 중점 거리
            mid_i = ((cand_i['p1'][0] + cand_i['p2'][0]) / 2, 
                    (cand_i['p1'][1] + cand_i['p2'][1]) / 2)
            mid_j = ((cand_j['p1'][0] + cand_j['p2'][0]) / 2,
                    (cand_j['p1'][1] + cand_j['p2'][1]) / 2)
            
            dist = np.sqrt((mid_i[0] - mid_j[0])**2 + (mid_i[1] - mid_j[1])**2)
            
            if dist < distance_threshold:
                cluster.append(cand_j)
                used[j] = True
        
        clusters.append(cluster)
    
    # 각 클러스터에서 최고 점수 선택
    best_candidates = []
    for cluster in clusters:
        # 클러스터 내에서 점수가 가장 높은 것 선택
        best = max(cluster, key=lambda x: x['score'])
        # 여러 방법에서 검출된 경우 보너스 점수
        best['score'] += len(cluster) * 0.1
        best_candidates.append(best)
    
    # 점수 순으로 정렬
    best_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 최종 2개 선택 (각도 조건 확인)
    final_leads = []
    if best_candidates:
        # 첫 번째 리드
        lead1 = best_candidates[0]
        final_leads.append((lead1['p1'], lead1['p2']))
        
        # 두 번째 리드 (각도 조건)
        if hull_centroid and len(best_candidates) > 1:
            for cand in best_candidates[1:]:
                # 끝점 결정 (hull에서 먼 점)
                d1 = np.sqrt((cand['p1'][0] - hull_centroid[0])**2 + 
                           (cand['p1'][1] - hull_centroid[1])**2)
                d2 = np.sqrt((cand['p2'][0] - hull_centroid[0])**2 + 
                           (cand['p2'][1] - hull_centroid[1])**2)
                
                tip1 = lead1['p2'] if np.sqrt((lead1['p2'][0] - hull_centroid[0])**2 + 
                                             (lead1['p2'][1] - hull_centroid[1])**2) > \
                                     np.sqrt((lead1['p1'][0] - hull_centroid[0])**2 + 
                                             (lead1['p1'][1] - hull_centroid[1])**2) else lead1['p1']
                tip2 = cand['p2'] if d2 > d1 else cand['p1']
                
                # 각도 계산
                vec1 = np.array(tip1) - np.array(hull_centroid)
                vec2 = np.array(tip2) - np.array(hull_centroid)
                
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                if 60 <= angle <= 180:  # 각도 조건 완화
                    final_leads.append((cand['p1'], cand['p2']))
                    break
    
    return final_leads

# --- 시각화 함수 ---
def visualize_robust_detection(img, final_leads, all_masks, hull, hull_centroid, all_candidates):
    """검출 결과 시각화"""
    h, w = img.shape[:2]
    
    # 2x4 그리드로 시각화
    vis_images = []
    
    # 1. 원본
    vis_images.append(img)
    
    # 2-6. 각 검출 방법 결과
    for i, (name, mask) in enumerate(all_masks[:5]):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vis_images.append(vis)
    
    # 7. 모든 후보
    candidates_vis = np.zeros_like(img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, cand in enumerate(all_candidates):
        color = colors[i % len(colors)]
        cv2.line(candidates_vis, cand['p1'], cand['p2'], color, 2)
    vis_images.append(candidates_vis)
    
    # 8. 최종 결과
    final_vis = img.copy()
    if hull is not None:
        cv2.drawContours(final_vis, [hull], -1, (0, 255, 0), 2)
    for p1, p2 in final_leads:
        cv2.line(final_vis, p1, p2, (0, 255, 255), 3)
        # 끝점 표시
        cv2.circle(final_vis, p1, 5, (255, 0, 0), -1)
        cv2.circle(final_vis, p2, 5, (255, 0, 0), -1)
    vis_images.append(final_vis)
    
    # 그리드 생성
    grid = []
    for i in range(0, 8, 4):
        row = np.hstack([cv2.resize(vis_images[j], (w//2, h//2)) 
                        for j in range(i, min(i+4, len(vis_images)))])
        grid.append(row)
    
    return np.vstack(grid)

# --- 메인 실행 ---
if __name__ == "__main__":
    import os
    
    image_files = [
        f for f in os.listdir('.')
        if any(k in f.lower() for k in ['led', 'cap']) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_files:
        print("LED 이미지를 찾을 수 없습니다.")
        exit()
    
    idx = 0
    cv2.namedWindow("Robust LED Detection", cv2.WINDOW_NORMAL)
    
    while True:
        img = cv2.imread(image_files[idx])
        if img is None:
            print(f"{image_files[idx]}를 읽을 수 없습니다.")
            idx = (idx + 1) % len(image_files)
            continue
        
        # 검출 수행
        final_leads, all_masks, hull, hull_centroid, all_candidates = \
            detect_led_leads_robust(img, debug=True)
        
        # 시각화
        vis = visualize_robust_detection(img, final_leads, all_masks, 
                                       hull, hull_centroid, all_candidates)
        
        # 정보 표시
        info = f"Image: {image_files[idx]} | Leads: {len(final_leads)}"
        cv2.putText(vis, info, (10, vis.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Robust LED Detection", vis)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(image_files)
        elif key == ord('p'):
            idx = (idx - 1) % len(image_files)
    
    cv2.destroyAllWindows()