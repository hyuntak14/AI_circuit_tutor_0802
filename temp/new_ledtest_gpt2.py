import cv2
import numpy as np
from hole_detector2 import HoleDetector
from skimage.morphology import skeletonize
import math

# --- Import or define LSD detector that returns only raw line coordinates ---
def lsd_detect_raw(img_gray):
    """
    LSD로 선분을 검출하고, 선분 좌표 리스트만 반환합니다.
    """
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(img_gray)
    raw_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            raw_lines.append((int(x1), int(y1), int(x2), int(y2)))
    return raw_lines

def draw_lsd_lines(img_color, lines_to_draw):
    """
    img_color 위에 lines_to_draw 리스트의 선분만 초록색으로 그려서 반환합니다.
    """
    out = img_color.copy()
    for (x1, y1, x2, y2) in lines_to_draw:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return out

# --- 향상된 전처리 함수들 ---
def remove_red_blue(img):
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

# --- 향상된 스켈레톤 함수들 ---
def prune_skeleton_neighborhood(skel_bin):
    pruned = skel_bin.copy()
    H, W = pruned.shape
    result = np.zeros_like(pruned)
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if pruned[y, x] == 1:
                nb = np.count_nonzero(pruned[y-1:y+2, x-1:x+2]) - 1
                if nb >= 2:
                    result[y, x] = 1
    return result

def advanced_skeleton_pruning(skel_bin, min_length=10):
    """향상된 스켈레톤 가지치기"""
    # 기본 가지치기
    pruned = prune_skeleton_neighborhood(skel_bin)
    
    # 짧은 가지 제거
    H, W = pruned.shape
    result = pruned.copy()
    
    # 연결된 구성요소 분석
    num_labels, labels = cv2.connectedComponents(pruned.astype(np.uint8))
    
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        component_points = np.where(component == 1)
        
        if len(component_points[0]) < min_length:
            result[component == 1] = 0
    
    return result

# --- 향상된 클러스터링 함수 ---
def advanced_cluster_endpoints(endpoints, thresh=5, min_cluster_size=1):
    """향상된 끝점 클러스터링"""
    if not endpoints:
        return []
    
    clusters = []
    for pt in endpoints:
        placed = False
        for cluster in clusters:
            # 클러스터 중심까지의 거리 계산
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            if distance((cx, cy), pt) <= thresh:
                cluster.append(pt)
                placed = True
                break
        if not placed:
            clusters.append([pt])
    
    # 클러스터 크기 필터링 및 중심점 계산
    clustered = []
    for cluster in clusters:
        if len(cluster) >= min_cluster_size:
            # 가중 평균 사용 (더 많은 점이 있는 곳에 가중치)
            avg_x = int(round(sum(p[0] for p in cluster) / len(cluster)))
            avg_y = int(round(sum(p[1] for p in cluster) / len(cluster)))
            clustered.append((avg_x, avg_y))
    
    return clustered

# --- 향상된 끝점 검출 함수들 ---
def get_skeleton_endpoints(binary_img, enable_prune, min_skel_length=10):
    """향상된 스켈레톤 기반 끝점 검출"""
    bw = (binary_img // 255).astype(np.uint8)
    skel_bin = skeletonize(bw).astype(np.uint8)
    
    if enable_prune:
        skel_bin = advanced_skeleton_pruning(skel_bin, min_skel_length)
    
    endpoints = []
    H, W = skel_bin.shape
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skel_bin[y, x] == 1:
                nb = np.count_nonzero(skel_bin[y - 1:y + 2, x - 1:x + 2]) - 1
                if nb == 1:  # 끝점 조건
                    endpoints.append((x, y))
    
    return advanced_cluster_endpoints(endpoints, thresh=5), skel_bin

def get_gray_mask_endpoints(img, s_low=0, s_high=50, v_low=180, v_high=255, min_area=20):
    """향상된 그레이 마스크 기반 끝점 검출"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, s_low, v_low), (180, s_high, v_high))

    # 향상된 모폴로지 처리
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_gray = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, k, iterations=2)

    conts, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    endpoints = []

    for cnt in conts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # 컨투어의 형태 분석
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 너무 원형인 것은 제외 (구멍일 가능성)
        if circularity > 0.8:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        roi = np.zeros_like(mask_gray[y:y+h, x:x+w])
        cnt_shifted = cnt - [x, y]
        cv2.drawContours(roi, [cnt_shifted], -1, 255, thickness=cv2.FILLED)

        bw_roi = (roi // 255).astype(np.uint8)
        skel_roi = skeletonize(bw_roi).astype(np.uint8)

        hh, ww = skel_roi.shape
        for yy in range(1, hh - 1):
            for xx in range(1, ww - 1):
                if skel_roi[yy, xx] == 1:
                    nb = np.count_nonzero(skel_roi[yy - 1:yy + 2, xx - 1:xx + 2]) - 1
                    if nb == 1:
                        endpoints.append((x + xx, y + yy))

    return advanced_cluster_endpoints(endpoints, thresh=5), mask_gray

def get_contour_endpoints(binary_img, min_area=50, max_area=5000):
    """컨투어 기반 끝점 검출 (새로운 방법)"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    endpoints = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            # 컨투어의 극값점들 찾기
            hull = cv2.convexHull(cnt, returnPoints=True)
            
            # 가장 바깥쪽 점들 찾기
            leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
            rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
            topmost = tuple(hull[hull[:, :, 1].argmin()][0])
            bottommost = tuple(hull[hull[:, :, 1].argmax()][0])
            
            # 중심점 계산
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 중심에서 가장 먼 점들을 끝점 후보로
                extreme_points = [leftmost, rightmost, topmost, bottommost]
                for pt in extreme_points:
                    dist_from_center = distance(pt, (cx, cy))
                    if dist_from_center > 10:  # 최소 거리 조건
                        endpoints.append(pt)
    
    return advanced_cluster_endpoints(endpoints, thresh=8), None

def get_lsd_endpoints(filtered_lsd_lines):
    """LSD 선분의 끝점들을 추출"""
    endpoints = []
    for (x1, y1, x2, y2) in filtered_lsd_lines:
        endpoints.extend([(x1, y1), (x2, y2)])
    return advanced_cluster_endpoints(endpoints, thresh=5)

# --- 향상된 라인 검출 및 필터링 ---
def filter_lines_by_angle(lines, min_angle=0, max_angle=150):
    """각도 기반 라인 필터링 (0~150도 범위)"""
    filtered_lines = []
    for line in lines:
        # 라인 형태가 ((x1, y1), (x2, y2)) 또는 (x1, y1, x2, y2)인지 확인
        if len(line) == 2:  # ((x1, y1), (x2, y2)) 형태
            (x1, y1), (x2, y2) = line
        else:  # (x1, y1, x2, y2) 형태
            x1, y1, x2, y2 = line
            
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = abs(angle) % 180  # 0~180도로 정규화
        
        # 허용 각도 범위 내에 있는지 확인 (0~150도)
        if min_angle <= angle <= max_angle:
            if len(line) == 2:
                filtered_lines.append(((x1, y1), (x2, y2)))
            else:
                filtered_lines.append((x1, y1, x2, y2))
    
    return filtered_lines

def get_candidate_lines_from_endpoints(endpoints,
                                       filtered_lsd_lines,
                                       hough_lines,
                                       match_dist_thresh,
                                       filter_angles,
                                       min_angle,
                                       max_angle,
                                       merge_angle_thresh=math.radians(5),
                                       merge_dist_thresh=30):
    """
    endpoints: list of (x, y) endpoint tuples from lsd + gray mask
    filtered_lsd_lines: list of (x1, y1, x2, y2) from initial LSD filtering
    hough_lines: numpy array (N,1,4) or list of (x1,y1,x2,y2) from Hough transform
    match_dist_thresh: float, max distance between endpoint and line
    filter_angles: unused in this context but kept for compatibility
    min_angle, max_angle: angle range to filter lines (in radians)
    merge_angle_thresh, merge_dist_thresh: thresholds for merging similar lines

    Returns list of candidate lines as pairs of endpoints ((x1, y1), (x2, y2))
    """
    # 0) 원시 선분 리스트 결합: flatten hough_lines if numpy array
    all_lines = []
    all_lines.extend(filtered_lsd_lines)
    if hough_lines is not None:
        if isinstance(hough_lines, np.ndarray):
            try:
                flat = hough_lines.reshape(-1, 4)
                hough_list = [ (int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in flat ]
            except Exception:
                hough_list = []
        else:
            hough_list = list(hough_lines)
        all_lines.extend(hough_list)

    candidates = []
    for ex, ey in endpoints:
        for line in all_lines:
            x1, y1, x2, y2 = line
            theta = math.atan2(y2 - y1, x2 - x1)
            if not (min_angle <= theta <= max_angle):
                continue
            if distance_point_to_line((ex, ey), (x1, y1), (x2, y2)) > match_dist_thresh:
                continue
            candidates.append((x1, y1, x2, y2))

    merged_candidates = merge_similar_lines(candidates,
                                            angle_thresh=merge_angle_thresh,
                                            dist_thresh=merge_dist_thresh)

    # 병합된 선분을 ((x1, y1), (x2, y2)) 형태로 변환하여 반환
    return [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in merged_candidates]

# --- 향상된 최종 끝점 선택 ---
def validate_tip_pair(tip1, tip2, hull_centroid, min_distance=10, max_distance=200):
    """끝점 쌍의 유효성 검증"""
    if hull_centroid is None:
        return True
        
    # 거리 조건
    tip_distance = distance(tip1, tip2)
    if tip_distance < min_distance or tip_distance > max_distance:
        return False
    
    # 중심점과의 관계 확인
    center = hull_centroid
    dist1 = distance(tip1, center)
    dist2 = distance(tip2, center)
    
    # 비슷한 거리에 있어야 함 (대칭성)
    ratio = min(dist1, dist2) / max(dist1, dist2)
    if ratio < 0.7:  # 거리 비율이 너무 차이나면 제외
        return False
    
    return True

def get_final_tips_from_lines(merged_lines, hull, hull_centroid, hull_proximity, match_dist_thresh, enable_validation=True, min_tip_dist=30, max_tip_dist=150):
    """향상된 최종 끝점 선택"""
    final_tips = []
    if hull is not None and hull_centroid is not None:
        candidate_tips = []
        for line in merged_lines:
            # 라인 형태 처리
            if len(line) == 2:  # ((x1, y1), (x2, y2)) 형태
                p1, p2 = line
            else:  # (x1, y1, x2, y2) 형태
                x1, y1, x2, y2 = line
                p1, p2 = (x1, y1), (x2, y2)
                
            d1 = cv2.pointPolygonTest(hull, p1, True)
            d2 = cv2.pointPolygonTest(hull, p2, True)
            
            # 더 엄격한 조건
            if -hull_proximity <= d1 <= hull_proximity and d2 < -hull_proximity:
                dist_from_center = distance(p2, hull_centroid)
                candidate_tips.append((p2, p1, dist_from_center))
            elif -hull_proximity <= d2 <= hull_proximity and d1 < -hull_proximity:
                dist_from_center = distance(p1, hull_centroid)
                candidate_tips.append((p1, p2, dist_from_center))
        
        # 거리순 정렬
        candidate_tips.sort(key=lambda x: x[2], reverse=True)
        
        if len(candidate_tips) >= 1:
            final_tips.append(candidate_tips[0][0])
            
            # 두 번째 끝점 선택
            for i in range(1, len(candidate_tips)):
                tip_candidate = candidate_tips[i][0]
                
                # 고급 검증 활성화 여부에 따라 처리
                if enable_validation:
                    # 첫 번째 끝점과의 유효성 검증
                    if validate_tip_pair(final_tips[0], tip_candidate, hull_centroid, min_tip_dist, max_tip_dist):
                        # 기존 끝점들과 너무 가깝지 않은지 확인
                        too_close = False
                        for existing_tip in final_tips:
                            if distance(existing_tip, tip_candidate) <= match_dist_thresh:
                                too_close = True
                                break
                        
                        if not too_close:
                            final_tips.append(tip_candidate)
                            break  # 2개만 선택
                else:
                    # 간단한 거리 검증만
                    too_close = False
                    for existing_tip in final_tips:
                        if distance(existing_tip, tip_candidate) <= match_dist_thresh:
                            too_close = True
                            break
                    
                    if not too_close:
                        final_tips.append(tip_candidate)
                        break  # 2개만 선택
    
    return final_tips

# --- 거리 계산 함수 ---
def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

import numpy as np
import math

def distance_point_to_line(pt, line_p1, line_p2):
    """pt가 line(p1→p2)에 수직 투영됐을 때의 거리"""
    x0, y0 = pt
    x1, y1 = line_p1
    x2, y2 = line_p2
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = math.hypot(y2-y1, x2-x1)
    return num/den if den else float('inf')

def merge_similar_lines(lines,
                        angle_thresh=math.radians(5),   # 각도 차 허용치 (5°)
                        dist_thresh=30):                 # 거리 차 허용치 (픽셀)  # 추측입니다.
    merged = []
    used = [False] * len(lines)
    
    for i, (x1, y1, x2, y2) in enumerate(lines):
        if used[i]:
            continue
        used[i] = True
        theta = math.atan2(y2 - y1, x2 - x1)
        group = [ (x1, y1, x2, y2) ]
        
        # 같은 그룹 찾기
        for j in range(i+1, len(lines)):
            if used[j]:
                continue
            x3, y3, x4, y4 = lines[j]
            theta_j = math.atan2(y4 - y3, x4 - x3)
            # 각도 차 계산 (−π→π)
            delta = abs(math.atan2(math.sin(theta - theta_j),
                                   math.cos(theta - theta_j)))
            if delta > angle_thresh:
                continue
            
            # 끝점들 간 최소 거리 검사
            if (distance_point_to_line((x3,y3),(x1,y1),(x2,y2)) < dist_thresh or
                distance_point_to_line((x4,y4),(x1,y1),(x2,y2)) < dist_thresh):
                used[j] = True
                group.append((x3, y3, x4, y4))
        
        # 그룹 내 모든 끝점 수집
        pts = [(x,y) for (x,y,_,_) in group] + [(x,y) for (_,_,x,y) in group]
        # 대표 방향 단위벡터
        ux, uy = math.cos(theta), math.sin(theta)
        # 투영값 리스트
        projs = [px*ux + py*uy for (px,py) in pts]
        
        # 최소/최대 투영값에 해당하는 실제 점을 꺼내 병합 선분 생성
        min_pt = pts[int(np.argmin(projs))]
        max_pt = pts[int(np.argmax(projs))]
        merged.append((min_pt[0], min_pt[1], max_pt[0], max_pt[1]))
    
    return merged



# --- 메인 실행부 ---
if __name__ == "__main__":
    image_files = [
        f for f in __import__('os').listdir('.')
        if ('led' in f.lower()) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not image_files:
        print("폴더 내에 'led'가 포함된 이미지가 없습니다.")
        exit()

    idx = 0
    img = cv2.imread(image_files[idx])
    if img is None:
        print(f"{image_files[idx]}를 읽을 수 없습니다.")
        exit()
    h, w = img.shape[:2]
    
    # HSV 범위 세팅
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r1_low = np.array([0, 100, 100], dtype=np.uint8)
    r1_high = np.array([10, 255, 255], dtype=np.uint8)
    r2_low = np.array([160, 100, 100], dtype=np.uint8)
    r2_high = np.array([180, 255, 255], dtype=np.uint8)
    g_low = np.array([40, 50, 50], dtype=np.uint8)
    g_high = np.array([80, 255, 255], dtype=np.uint8)
    y_low = np.array([18, 80, 100], dtype=np.uint8)
    y_high = np.array([35, 255, 255], dtype=np.uint8)

    # 윈도우와 트랙바 설정 (개선된 기본값)
    win_name = "Advanced Lead Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w * 2, h * 2)
    
    def nothing(x):
        pass

    # 기본 파라미터들 (더 최적화된 기본값)
    cv2.createTrackbar("hole_radius", win_name, 8, 50, nothing)
    cv2.createTrackbar("hole_area_thresh", win_name, 100, 500, nothing)
    cv2.createTrackbar("Canny_th1", win_name, 30, 300, nothing)
    cv2.createTrackbar("Canny_th2", win_name, 100, 300, nothing)
    cv2.createTrackbar("min_length", win_name, 30, max(w, h), nothing)
    cv2.createTrackbar("hull_proximity", win_name, 25, 100, nothing)
    cv2.createTrackbar("match_dist_thresh", win_name, 15, 50, nothing)
    cv2.createTrackbar("Use_Otsu", win_name, 1, 1, nothing)
    cv2.createTrackbar("Apply_Sharpen", win_name, 0, 1, nothing)
    cv2.createTrackbar("Apply_Smooth", win_name, 1, 1, nothing)
    cv2.createTrackbar("Enable_Pruning", win_name, 1, 1, nothing)
    cv2.createTrackbar("CLAHE_clip", win_name, 3, 10, nothing)
    cv2.createTrackbar("block_size", win_name, 15, 51, nothing)
    cv2.createTrackbar("Silver_S_low", win_name, 0, 100, nothing)
    cv2.createTrackbar("Silver_S_high", win_name, 40, 100, nothing)
    cv2.createTrackbar("Silver_V_low", win_name, 160, 255, nothing)
    cv2.createTrackbar("Silver_V_high", win_name, 255, 255, nothing)
    cv2.createTrackbar("Filter_Angles", win_name, 1, 1, nothing)
    cv2.createTrackbar("Min_Angle", win_name, 0, 180, nothing)
    cv2.createTrackbar("Max_Angle", win_name, 150, 180, nothing)
    cv2.createTrackbar("Enable_Validation", win_name, 1, 1, nothing)
    cv2.createTrackbar("Min_Tip_Distance", win_name, 30, 100, nothing)
    cv2.createTrackbar("Max_Tip_Distance", win_name, 150, 300, nothing)

    while True:
        # 트랙바 값 읽기
        hole_r = cv2.getTrackbarPos("hole_radius", win_name)
        hole_area = cv2.getTrackbarPos("hole_area_thresh", win_name)
        canny1 = cv2.getTrackbarPos("Canny_th1", win_name)
        canny2 = cv2.getTrackbarPos("Canny_th2", win_name)
        min_length = max(cv2.getTrackbarPos("min_length", win_name), 1)
        hull_proximity = cv2.getTrackbarPos("hull_proximity", win_name)
        match_dist_thresh = cv2.getTrackbarPos("match_dist_thresh", win_name)
        use_otsu = cv2.getTrackbarPos("Use_Otsu", win_name) == 1
        apply_sharpen = cv2.getTrackbarPos("Apply_Sharpen", win_name) == 1
        apply_smooth = cv2.getTrackbarPos("Apply_Smooth", win_name) == 1
        enable_prune = cv2.getTrackbarPos("Enable_Pruning", win_name) == 1
        clahe_clip = cv2.getTrackbarPos("CLAHE_clip", win_name)
        blk = cv2.getTrackbarPos("block_size", win_name)
        s_low = cv2.getTrackbarPos("Silver_S_low", win_name)
        s_high = cv2.getTrackbarPos("Silver_S_high", win_name)
        v_low = cv2.getTrackbarPos("Silver_V_low", win_name)
        v_high = cv2.getTrackbarPos("Silver_V_high", win_name)
        filter_angles = cv2.getTrackbarPos("Filter_Angles", win_name) == 1
        min_angle = cv2.getTrackbarPos("Min_Angle", win_name)
        max_angle = cv2.getTrackbarPos("Max_Angle", win_name)
        enable_validation = cv2.getTrackbarPos("Enable_Validation", win_name) == 1
        min_tip_dist = cv2.getTrackbarPos("Min_Tip_Distance", win_name)
        max_tip_dist = cv2.getTrackbarPos("Max_Tip_Distance", win_name)
        
        if blk < 3: blk = 3
        if blk % 2 == 0: blk += 1

        # 1) 향상된 전처리
        img_clean = remove_red_blue(img)
        gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
        processed = advanced_preprocessing(gray, apply_sharpen, apply_smooth, clahe_clip)

        # 2) 이진화
        if use_otsu:
            _, thr = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_inv = cv2.bitwise_not(thr)
        else:
            thr = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, blk, 5)
            mask_inv = cv2.bitwise_not(thr)

        # 3) 노이즈 제거
        conts, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_mask = np.zeros_like(mask_inv)
        for cnt in conts:
            if cv2.contourArea(cnt) < hole_area:
                cv2.drawContours(small_mask, [cnt], -1, 255, -1)
        mask_small_removed = cv2.bitwise_and(mask_inv, cv2.bitwise_not(small_mask))

        # 4) 구멍 제거
        hole_detector = HoleDetector()
        def remove_hole(mask, img, hole_radius):
            centers = hole_detector.detect_holes_raw(img)
            clean_mask = mask.copy()
            H, W = clean_mask.shape[:2]
            half = hole_radius
            for cx, cy in centers:
                x_int, y_int = int(round(cx)), int(round(cy))
                x0 = max(x_int - half, 0)
                y0 = max(y_int - half, 0)
                x1 = min(x_int + half, W - 1)
                y1 = min(y_int + half, H - 1)
                cv2.rectangle(clean_mask, (x0, y0), (x1, y1), 0, -1)
            return clean_mask

        mask_no_holes = remove_hole(mask_small_removed, img_clean, hole_r)

        # 5) 모폴로지 연산
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(mask_no_holes, cv2.MORPH_OPEN, k, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)
        connected = cv2.dilate(closed, k, iterations=1)
        processed_morph = cv2.erode(connected, k, iterations=1)

        # 6) 다양한 방법으로 끝점 검출
        skel_endpoints, skel_bin = get_skeleton_endpoints(processed_morph, enable_prune)
        gray_endpoints, mask_gray = get_gray_mask_endpoints(img_clean, s_low, s_high, v_low, v_high)
        contour_endpoints, _ = get_contour_endpoints(processed_morph)
        
        # 7) LSD 및 Hough 선분 검출
        raw_lsd_lines = lsd_detect_raw(processed)
        filtered_lsd_lines = [
            (x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_lsd_lines
            if np.hypot(x2 - x1, y2 - y1) > 40
        ]
        merged_lsd_lines = merge_similar_lines(filtered_lsd_lines,
                                       angle_thresh=math.radians(10),
                                       dist_thresh=10)  # 추측입니다
        merged_lsd_lines = filtered_lsd_lines
        lsd_endpoints = get_lsd_endpoints(filtered_lsd_lines)
        
        
        edges = cv2.Canny(processed_morph, canny1, canny2)
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                    minLineLength=min_length, maxLineGap=10)
        
        hough_list = []
        if hough_lines is not None:
            for l in hough_lines:
                x1,y1,x2,y2 = l[0]
                hough_list.append((x1,y1,x2,y2))
        # 2) 비슷한 선분 병합
            merged_hough = merge_similar_lines(hough_list,
                                            angle_thresh=math.radians(10),
                                            dist_thresh=10)  # 필요에 따라 조정
        else:
            merged_hough = []

        if merged_hough:
            # 배열 → (N,1,4) 형태로 reshape
            hough_lines = np.array(merged_hough, dtype=np.int32).reshape(-1, 1, 4)
        else:
            hough_lines = None


        # 8) Convex Hull 생성
        mask_body = cv2.bitwise_or(
            cv2.bitwise_or(cv2.inRange(hsv_full, r1_low, r1_high),
                          cv2.inRange(hsv_full, r2_low, r2_high)),
            cv2.bitwise_or(cv2.inRange(hsv_full, g_low, g_high),
                          cv2.inRange(hsv_full, y_low, y_high))
        )
        kernel_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body_clean = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel_body, iterations=2)
        mask_body_clean = cv2.morphologyEx(mask_body_clean, cv2.MORPH_CLOSE, kernel_body, iterations=2)
        mask_body_clean = cv2.dilate(mask_body_clean, kernel_body, iterations=2)
        
        conts_body, _ = cv2.findContours(mask_body_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = None
        hull_centroid = None
        if conts_body:
            largest_cont = max(conts_body, key=lambda c: cv2.contourArea(c))
            hull = cv2.convexHull(largest_cont)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                hull_centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])

        # 9) 향상된 4가지 조합으로 끝점 검출
        # 1) LSD + Gray Mask  
        endpoints_lsd_gray = lsd_endpoints + gray_endpoints
        lines_lsd_gray = get_candidate_lines_from_endpoints(endpoints_lsd_gray, filtered_lsd_lines, 
                                                           hough_lines, match_dist_thresh, filter_angles, min_angle, max_angle)
        tips_lsd_gray = get_final_tips_from_lines(lines_lsd_gray, hull, hull_centroid, 
                                                 hull_proximity, match_dist_thresh, enable_validation, min_tip_dist, max_tip_dist)
        
        # 2) LSD + Skeleton
        endpoints_lsd_skel = lsd_endpoints + skel_endpoints  
        lines_lsd_skel = get_candidate_lines_from_endpoints(endpoints_lsd_skel, filtered_lsd_lines,
                                                           hough_lines, match_dist_thresh, filter_angles, min_angle, max_angle)
        tips_lsd_skel = get_final_tips_from_lines(lines_lsd_skel, hull, hull_centroid,
                                                 hull_proximity, match_dist_thresh, enable_validation, min_tip_dist, max_tip_dist)
        
        # 3) Skeleton + Gray + Contour (기존 방법을 개선)
        endpoints_enhanced = skel_endpoints + gray_endpoints + contour_endpoints
        lines_enhanced = get_candidate_lines_from_endpoints(endpoints_enhanced, filtered_lsd_lines,
                                                           hough_lines, match_dist_thresh, filter_angles, min_angle, max_angle)
        tips_enhanced = get_final_tips_from_lines(lines_enhanced, hull, hull_centroid,
                                                 hull_proximity, match_dist_thresh, enable_validation, min_tip_dist, max_tip_dist)
        
        # 4) All Methods Combined (최고 성능)
        endpoints_all = lsd_endpoints + gray_endpoints + skel_endpoints + contour_endpoints
        lines_all = get_candidate_lines_from_endpoints(endpoints_all, filtered_lsd_lines,
                                                      hough_lines, match_dist_thresh, filter_angles, min_angle, max_angle)
        tips_all = get_final_tips_from_lines(lines_all, hull, hull_centroid,
                                            hull_proximity, match_dist_thresh, enable_validation, min_tip_dist, max_tip_dist)

        # --- 향상된 시각화 ---
        def create_result_visualization(base_img, lines, tips, title_color):
            """결과 시각화 이미지를 생성합니다."""
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR) if len(base_img.shape) == 2 else base_img.copy()
            
            # 선분 그리기
            for (p1, p2) in lines:
                cv2.line(vis, p1, p2, (0, 255, 0), 2)
            
            # Hull 그리기 (반투명)
            if hull is not None:
                overlay = vis.copy()
                cv2.drawContours(overlay, [hull], -1, (255, 255, 0), 2)
                vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
            
            # 최종 끝점 그리기 (더 강조)
            for i, (x, y) in enumerate(tips):
                cv2.circle(vis, (x, y), 10, (0, 0, 0), 3)  # 검은색 테두리
                cv2.circle(vis, (x, y), 8, title_color, -1)  # 색상 원
                cv2.circle(vis, (x, y), 3, (255, 255, 255), -1)  # 흰색 중심
                cv2.putText(vis, str(i+1), (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(vis, str(i+1), (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2)
            
            return vis

        # 각 방법별 결과 생성
        vis_lsd_gray = create_result_visualization(processed_morph, lines_lsd_gray, tips_lsd_gray, (0, 255, 255))
        vis_lsd_skel = create_result_visualization(processed_morph, lines_lsd_skel, tips_lsd_skel, (255, 0, 255))
        vis_enhanced = create_result_visualization(processed_morph, lines_enhanced, tips_enhanced, (255, 255, 0))
        vis_all = create_result_visualization(processed_morph, lines_all, tips_all, (0, 0, 255))

        def to_bgr(x):
            return x if len(x.shape) == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

        resize_size = (w // 2, h // 2)
        
        # 기본 정보 창들
        basic_imgs = [
            to_bgr(img),
            to_bgr(processed),
            to_bgr(processed_morph),
            to_bgr((skel_bin * 255).astype(np.uint8))
        ]
        basic_tiles = [cv2.resize(v, resize_size) for v in basic_imgs]
        
        # 4가지 조합 결과 창들  
        result_imgs = [vis_lsd_gray, vis_lsd_skel, vis_enhanced, vis_all]
        result_tiles = [cv2.resize(v, resize_size) for v in result_imgs]
        
        # 텍스트 라벨 추가 (폰트 크기도 증가)
        def add_label(img, text, color=(255, 255, 255)):
            labeled = img.copy()
            cv2.putText(labeled, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            return labeled
        
        basic_tiles[0] = add_label(basic_tiles[0], "Original")
        basic_tiles[1] = add_label(basic_tiles[1], "Preprocessed")
        basic_tiles[2] = add_label(basic_tiles[2], "Binary Mask")
        basic_tiles[3] = add_label(basic_tiles[3], "Skeleton")
        
        result_tiles[0] = add_label(result_tiles[0], f"1.LSD+Gray ({len(tips_lsd_gray)})", (0, 255, 255))
        result_tiles[1] = add_label(result_tiles[1], f"2.LSD+Skel ({len(tips_lsd_skel)})", (255, 0, 255))
        result_tiles[2] = add_label(result_tiles[2], f"3.Enhanced ({len(tips_enhanced)})", (255, 255, 0))
        result_tiles[3] = add_label(result_tiles[3], f"4.All ({len(tips_all)})", (0, 0, 255))
        
        # 2x2 그리드로 배치
        basic_top_row = np.hstack(basic_tiles[0:2])
        basic_bottom_row = np.hstack(basic_tiles[2:4])
        basic_grid = np.vstack((basic_top_row, basic_bottom_row))
        
        result_top_row = np.hstack(result_tiles[0:2])
        result_bottom_row = np.hstack(result_tiles[2:4])
        result_grid = np.vstack((result_top_row, result_bottom_row))
        
        cv2.imshow("Basic Info (2x2)", basic_grid)
        cv2.imshow("4 Detection Methods (2x2)", result_grid)
        
        # 윈도우 크기 조정
        cv2.namedWindow("Basic Info (2x2)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("4 Detection Methods (2x2)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Basic Info (2x2)", w, h)
        cv2.resizeWindow("4 Detection Methods (2x2)", w, h)
        
        # 개별 창들
        cv2.imshow("1. LSD + Gray Mask", vis_lsd_gray)
        cv2.imshow("2. LSD + Skeleton", vis_lsd_skel) 
        cv2.imshow("3. Enhanced Method", vis_enhanced)
        cv2.imshow("4. All Combined", vis_all)
        
        # 향상된 결과 정보 출력
        print(f"\n=== Advanced Detection Results ===")
        print(f"Filter Settings: Angles={filter_angles} ({min_angle}°-{max_angle}°), Validation={enable_validation}")
        print(f"LSD Endpoints: {len(lsd_endpoints)}")
        print(f"Gray Mask Endpoints: {len(gray_endpoints)}")
        print(f"Skeleton Endpoints: {len(skel_endpoints)}")
        print(f"Contour Endpoints: {len(contour_endpoints)}")
        print(f"")
        print(f"1. LSD + Gray Tips: {len(tips_lsd_gray)} -> {tips_lsd_gray}")
        print(f"2. LSD + Skeleton Tips: {len(tips_lsd_skel)} -> {tips_lsd_skel}")
        print(f"3. Enhanced Tips: {len(tips_enhanced)} -> {tips_enhanced}")
        print(f"4. All Combined Tips: {len(tips_all)} -> {tips_all}")
        
        # 끝점 품질 평가
        all_methods = [
            ("LSD + Gray", tips_lsd_gray),
            ("LSD + Skeleton", tips_lsd_skel),
            ("Enhanced", tips_enhanced),
            ("All Combined", tips_all)
        ]
        
        # 2개 끝점을 정확히 검출한 방법들
        perfect_methods = [name for name, tips in all_methods if len(tips) == 2]
        if perfect_methods:
            print(f"\nPerfect Detection (2 tips): {', '.join(perfect_methods)}")
        
        # 가장 많은 끝점을 검출한 방법
        best_method = max(all_methods, key=lambda x: len(x[1]))
        print(f"Best Method: {best_method[0]} with {len(best_method[1])} tips")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(image_files)
            img = cv2.imread(image_files[idx])
            if img is None:
                print(f"{image_files[idx]}를 읽을 수 없습니다.")
            else:
                h, w = img.shape[:2]
                cv2.resizeWindow(win_name, w * 2, h * 2)
                hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
    cv2.destroyAllWindows()