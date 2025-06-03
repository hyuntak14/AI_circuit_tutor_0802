import cv2
import numpy as np
from hole_detector2 import HoleDetector
from skimage.morphology import skeletonize

# --- 전처리 함수들 ---
def remove_red_blue(img):
    """
    빨강/파랑 영역을 주변 픽셀 색상으로 보간하여 제거합니다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 빨강 범위
    lr1 = np.array([0, 100, 100], dtype=np.uint8)
    ur1 = np.array([10, 255, 255], dtype=np.uint8)
    lr2 = np.array([160, 100, 100], dtype=np.uint8)
    ur2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lr1, ur1),
        cv2.inRange(hsv, lr2, ur2)
    )
    # 파랑 범위
    lb = np.array([100, 150, 50], dtype=np.uint8)
    ub = np.array([140, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lb, ub)

    mask_rb = cv2.bitwise_or(mask_red, mask_blue)
    img_inpainted = cv2.inpaint(img, mask_rb, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img_inpainted

def apply_clahe(gray, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def gray_mask_adaptive(gray_clahe, block_size=15, C=5, hole_area_thresh=15):
    thr = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        block_size, C
    )
    mask = cv2.bitwise_not(thr)
    hole_mask = np.zeros_like(mask)
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in conts:
        if cv2.contourArea(cnt) < hole_area_thresh:
            cv2.drawContours(hole_mask, [cnt], -1, 255, -1)
    mask_clean = cv2.bitwise_and(mask, cv2.bitwise_not(hole_mask))
    return mask_clean

def remove_holes(mask, img, hole_radius=7):
    hd = HoleDetector()
    centers = hd.detect_holes_raw(img)
    clean_mask = mask.copy()
    h, w = clean_mask.shape[:2]
    half = hole_radius
    for cx, cy in centers:
        x_int = int(round(cx))
        y_int = int(round(cy))
        x0 = max(x_int - half, 0)
        y0 = max(y_int - half, 0)
        x1 = min(x_int + half, w - 1)
        y1 = min(y_int + half, h - 1)
        cv2.rectangle(clean_mask, (x0, y0), (x1, y1), 0, -1)
    return clean_mask

# --- 거리 계산 함수 ---
def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

# --- 빈 콜백 함수 ---
def nothing(x):
    pass

# --- 리드 검증 함수들 (개선사항 1, 2) ---
def validate_lead_pair(tip1, tip2, hull_centroid, min_angle=10, max_angle=170):
    """두 리드 끝점이 유효한 쌍인지 검증"""
    if hull_centroid is None:
        return False
    
    # 두 리드가 hull 중심에서 보는 각도 계산
    vec1 = np.array(tip1) - np.array(hull_centroid)
    vec2 = np.array(tip2) - np.array(hull_centroid)
    
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    # LED 리드는 보통 90-180도 사이의 각도를 가짐 (범위 확대)
    return min_angle <= angle <= max_angle

def calculate_angle_between_leads(tip1, tip2, hull_centroid):
    """두 리드 사이의 각도 계산"""
    if hull_centroid is None:
        return 0
    
    vec1 = np.array(tip1) - np.array(hull_centroid)
    vec2 = np.array(tip2) - np.array(hull_centroid)
    
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    return angle

def calculate_line_thickness(cnt, line_params):
    """컨투어의 평균 두께 계산"""
    vx, vy, x0, y0 = line_params
    # 법선 벡터
    nx, ny = -vy, vx
    
    pts = cnt.reshape(-1, 2)
    # 직선으로부터의 거리들
    distances = []
    for pt in pts:
        d = abs((pt[0] - x0) * nx + (pt[1] - y0) * ny)
        distances.append(d)
    
    return np.mean(distances) * 2  # 평균 두께

# --- 컨투어에서 fitLine 후 끝점 및 직선 파라미터 추출 (수직 제외 로직 포함) ---
def fit_line_endpoints(cnt, w, h, min_length, max_ratio=0.9, min_thickness=2, max_thickness=15):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    # 거의 수직인 경우 제외
    if abs(vx) < 0.1:
        return None

    # 두께 체크 (개선사항 2)
    params = (vx, vy, x0, y0)
    thickness = calculate_line_thickness(cnt, params)
    if thickness < min_thickness or thickness > max_thickness:
        return None

    diffs = pts - np.array([[x0, y0]], dtype=np.float32)
    ts = diffs.dot(np.array([vx, vy], dtype=np.float32))
    t_min, t_max = ts.min(), ts.max()

    p1 = (int(round(x0 + vx * t_min)), int(round(y0 + vy * t_min)))
    p2 = (int(round(x0 + vx * t_max)), int(round(y0 + vy * t_max)))
    length = distance(p1, p2)
    if length < min_length:
        return None

    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if dx >= max_ratio * w or dy >= max_ratio * h:
        return None

    for (x, y) in (p1, p2):
        if x <= 1 or x >= w - 2 or y <= 1 or y >= h - 2:
            return None

    return p1, p2, params, thickness

# --- 스켈레톤 + 엔드포인트 추출 ---
def skeleton_and_endpoints(binary_img):
    bw = (binary_img // 255).astype(np.uint8)
    skel = skeletonize(bw).astype(np.uint8) * 255
    endpoints = []
    H, W = skel.shape
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skel[y, x] == 255:
                nb = np.count_nonzero(skel[y-1:y+2, x-1:x+2]) - 1
                if nb == 1:
                    endpoints.append((x, y))
    return skel, endpoints

# --- 스켈레톤 끝점 클러스터링 (5픽셀 이내) ---
def cluster_endpoints(endpoints, thresh=5):
    clusters = []
    for pt in endpoints:
        placed = False
        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            if distance((cx, cy), pt) <= thresh:
                cluster.append(pt)
                placed = True
                break
        if not placed:
            clusters.append([pt])
    clustered = []
    for cluster in clusters:
        avg_x = int(round(sum(p[0] for p in cluster) / len(cluster)))
        avg_y = int(round(sum(p[1] for p in cluster) / len(cluster)))
        clustered.append((avg_x, avg_y))
    return clustered

# --- 메인 실행부 ---
if __name__ == "__main__":
    image_files = [
        f for f in __import__('os').listdir('.')
        if any(k in f.lower() for k in ['led', 'cappp']) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not image_files:
        print("폴더 내에 'led' 또는 'cap'이 포함된 이미지가 없습니다.")
        exit()

    idx = 0
    img = cv2.imread(image_files[idx])
    if img is None:
        print(f"{image_files[idx]}를 읽을 수 없습니다.")
        exit()
    h, w = img.shape[:2]

    # --- LED 몸체 검출: HSV에서 빨강·초록·노랑 영역 이진화 & 모폴로지 정제 ---
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨강 범위
    r1_low = np.array([0, 100, 100], dtype=np.uint8)
    r1_high = np.array([10, 255, 255], dtype=np.uint8)
    r2_low = np.array([160, 100, 100], dtype=np.uint8)
    r2_high = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv_full, r1_low, r1_high),
        cv2.inRange(hsv_full, r2_low, r2_high)
    )
    # 초록 범위
    g_low = np.array([40,  50,  50], dtype=np.uint8)
    g_high = np.array([80, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv_full, g_low, g_high)
    # 노랑 범위
    y_low = np.array([20, 100, 100], dtype=np.uint8)
    y_high = np.array([30, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_full, y_low, y_high)

    mask_body = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green), mask_yellow)
    # 모폴로지 정제
    kernel_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_body_clean = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel_body, iterations=2)
    mask_body_clean = cv2.morphologyEx(mask_body_clean, cv2.MORPH_CLOSE, kernel_body, iterations=2)

    # 가장 큰 컨투어 검출 후 convex hull 계산
    conts_body, _ = cv2.findContours(mask_body_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = None
    hull_centroid = None
    if conts_body:
        largest_cont = max(conts_body, key=lambda c: cv2.contourArea(c))
        hull = cv2.convexHull(largest_cont)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            hull_centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        else:
            hull_centroid = None

    win_name = "Contour + Skeleton Visualization"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w * 2, h * 2)

    # 트랙바: 전처리 파라미터 + 최소 길이 + 두께 파라미터 (개선사항 6)
    cv2.createTrackbar("block_size",       win_name, 15, 51, nothing)
    cv2.createTrackbar("C",                win_name, 5, 20, nothing)
    cv2.createTrackbar("CLAHE_clip",       win_name, 2, 10, nothing)
    cv2.createTrackbar("hole_radius",      win_name, 7, 20, nothing)
    cv2.createTrackbar("hole_area_thresh", win_name, 15, 100, nothing)
    cv2.createTrackbar("Canny_th1",        win_name, 50, 200, nothing)
    cv2.createTrackbar("Canny_th2",        win_name, 150, 300, nothing)
    cv2.createTrackbar("min_length",       win_name, 20, max(w, h), nothing)
    cv2.createTrackbar("lead_thickness_min", win_name, 2, 10, nothing)
    cv2.createTrackbar("lead_thickness_max", win_name, 15, 30, nothing)
    cv2.createTrackbar("hull_proximity", win_name, 30, 100, nothing)

    while True:
        h, w = img.shape[:2]

        blk = cv2.getTrackbarPos("block_size", win_name)
        if blk < 3: blk = 3
        if blk % 2 == 0: blk += 1
        Cval = cv2.getTrackbarPos("C", win_name)
        clahe_clip = cv2.getTrackbarPos("CLAHE_clip", win_name)
        hole_r = cv2.getTrackbarPos("hole_radius", win_name)
        hole_area = cv2.getTrackbarPos("hole_area_thresh", win_name)
        canny1 = cv2.getTrackbarPos("Canny_th1", win_name)
        canny2 = cv2.getTrackbarPos("Canny_th2", win_name)
        min_length = cv2.getTrackbarPos("min_length", win_name)
        if min_length < 1: min_length = 1
        lead_thickness_min = cv2.getTrackbarPos("lead_thickness_min", win_name)
        lead_thickness_max = cv2.getTrackbarPos("lead_thickness_max", win_name)
        hull_proximity = cv2.getTrackbarPos("hull_proximity", win_name)

        # 1) 빨강/파랑 제거 → 그레이스케일
        img_clean = remove_red_blue(img)
        gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

        # 2) Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3) CLAHE
        gray_clahe = apply_clahe(blurred, clipLimit=float(clahe_clip), tileGridSize=(8, 8))

        # 4) AdaptiveThreshold + 반전
        thr = cv2.adaptiveThreshold(
            gray_clahe, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blk, Cval
        )
        mask_inv = cv2.bitwise_not(thr)

        # 5) 작은 노이즈 컨투어 제거
        conts, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_mask = np.zeros_like(mask_inv)
        for cnt in conts:
            if cv2.contourArea(cnt) < hole_area:
                cv2.drawContours(small_mask, [cnt], -1, 255, -1)
        mask_small_removed = cv2.bitwise_and(mask_inv, cv2.bitwise_not(small_mask))

        # 6) HoleDetector 기반 구멍 제거
        mask_no_holes = remove_holes(mask_small_removed, img_clean, hole_radius=hole_r)

        # 7) Closing (3×3 커널) + Dilation (끊어진 선 연결)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(mask_no_holes, cv2.MORPH_CLOSE, kernel, iterations=1)
        connected = cv2.dilate(closed, kernel, iterations=1)

        # 8) Canny 엣지 (확인용)
        edges = cv2.Canny(connected, canny1, canny2)

        # 9) Skeleton + 엔드포인트 + 엔드포인트 클러스터링
        skel, skel_eps = skeleton_and_endpoints(connected)
        clustered_eps = cluster_endpoints(skel_eps, thresh=5)
        skel_vis = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        for (x, y) in clustered_eps:
            cv2.circle(skel_vis, (x, y), 4, (255, 0, 0), -1)

        # 10) 컨투어 검출 → 직선 근사 후 끝점 & 파라미터 추출
        line_vis = cv2.cvtColor(connected, cv2.COLOR_GRAY2BGR)
        contours2, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        raw_lines = []  # (p1, p2, (vx, vy, x0, y0), length, thickness)
        for cnt in contours2:
            res = fit_line_endpoints(cnt, w, h, min_length, min_thickness=lead_thickness_min, max_thickness=lead_thickness_max)
            if res:
                p1, p2, params, thickness = res
                length = distance(p1, p2)
                raw_lines.append((p1, p2, params, length, thickness))

        # 11) 기울기-위치 유사 선분 병합
        merged_lines = []
        used = [False] * len(raw_lines)
        angle_thresh = 0.1  # 라디안 약 5.7도
        dist_thresh = 5     # 픽셀 거리 기준

        def line_params_from_endpoints(p1, p2):
            A = p2[1] - p1[1]
            B = -(p2[0] - p1[0])
            C = p2[0] * p1[1] - p2[1] * p1[0]
            norm = np.hypot(A, B)
            if norm == 0:
                return A, B, C
            return A / norm, B / norm, C / norm

        for i, (p1_i, p2_i, line_i, _, _) in enumerate(raw_lines):
            if used[i]:
                continue
            vx_i, vy_i, x0_i, y0_i = line_i
            ang_i = np.arctan2(vy_i, vx_i)
            group_points = [p1_i, p2_i]
            used[i] = True

            A_i, B_i, C_i = line_params_from_endpoints(p1_i, p2_i)

            for j in range(i + 1, len(raw_lines)):
                if used[j]:
                    continue
                p1_j, p2_j, line_j, _, _ = raw_lines[j]
                vx_j, vy_j, x0_j, y0_j = line_j
                ang_j = np.arctan2(vy_j, vx_j)
                if abs(ang_i - ang_j) < angle_thresh:
                    d1 = abs(A_i * p1_j[0] + B_i * p1_j[1] + C_i)
                    d2 = abs(A_i * p2_j[0] + B_i * p2_j[1] + C_i)
                    if d1 < dist_thresh and d2 < dist_thresh:
                        used[j] = True
                        group_points.extend([p1_j, p2_j])

            ref = np.array([x0_i, y0_i], dtype=np.float32)
            dir_vec = np.array([vx_i, vy_i], dtype=np.float32)
            ts = []
            for (x, y) in group_points:
                t = dir_vec.dot(np.array([x, y], dtype=np.float32) - ref)
                ts.append((t, (x, y)))
            t_min, p_min = min(ts, key=lambda x: x[0])
            t_max, p_max = max(ts, key=lambda x: x[0])
            merged_lines.append((p_min, p_max))

        # 12) 검출된 merged_lines 중 가장 긴 선분들 순으로 정렬
        merged_with_length = []
        for (p1, p2) in merged_lines:
            merged_with_length.append((p1, p2, distance(p1, p2)))
        merged_with_length.sort(key=lambda x: x[2], reverse=True)

        # --- Convex Hull 기반: 몸체 근처에서 뻗어나가는 선 끝점 2개 선택 (개선사항 1) ---
        final_tips = []
        if hull is not None and hull_centroid is not None:
            # 리드가 hull 주변에서 시작하는 경우를 포함하여 검출
            # 한 점이 hull 근처(일정 거리 이내)에 있고 다른 점이 멀리 있는 경우
            candidate_tips = []
            hull_proximity_threshold = hull_proximity  # 트랙바에서 조절 가능
            
            for (p1, p2, length) in merged_with_length:
                d1 = cv2.pointPolygonTest(hull, p1, True)  # True로 실제 거리 계산
                d2 = cv2.pointPolygonTest(hull, p2, True)
                
                # p1이 hull 근처(내부 포함)이고 p2가 멀리 있는 경우
                if -hull_proximity_threshold <= d1 <= hull_proximity_threshold and d2 < -hull_proximity_threshold:
                    # 더 멀리 있는 점을 tip으로, 가까운 점에서 centroid까지의 거리도 고려
                    candidate_tips.append((p2, p1, length, distance(p2, hull_centroid)))
                # p2가 hull 근처(내부 포함)이고 p1이 멀리 있는 경우
                elif -hull_proximity_threshold <= d2 <= hull_proximity_threshold and d1 < -hull_proximity_threshold:
                    candidate_tips.append((p1, p2, length, distance(p1, hull_centroid)))
            
            # 길이와 centroid로부터의 거리를 모두 고려하여 정렬
            # 긴 리드이면서 끝점이 centroid로부터 먼 것을 우선시
            candidate_tips.sort(key=lambda x: x[2] * 0.3 + x[3] * 0.7, reverse=True)
            
            # 각도 검증을 통해 유효한 쌍 찾기
            if len(candidate_tips) >= 2:
                # 첫 번째 tip 선택
                final_tips.append(candidate_tips[0][0])
                
                # 두 번째 tip은 첫 번째와 적절한 각도를 이루는 것 선택
                found_second = False
                for i in range(1, len(candidate_tips)):
                    if validate_lead_pair(final_tips[0], candidate_tips[i][0], hull_centroid):
                        final_tips.append(candidate_tips[i][0])
                        found_second = True
                        break
                
                # 각도 조건을 만족하는 두 번째 리드가 없으면, 가장 좋은 후보를 선택
                if not found_second and len(candidate_tips) >= 2:
                    # 첫 번째와 가장 각도가 큰 것을 선택
                    best_angle = 0
                    best_idx = 1
                    for i in range(1, min(len(candidate_tips), 5)):
                        angle = calculate_angle_between_leads(final_tips[0], candidate_tips[i][0], hull_centroid)
                        if angle > best_angle:
                            best_angle = angle
                            best_idx = i
                    if best_angle > 60:  # 최소 60도 이상이면 선택
                        final_tips.append(candidate_tips[best_idx][0])
                        
            elif len(candidate_tips) == 1:
                final_tips.append(candidate_tips[0][0])

        # --- 시각화: 스켈레톤 엔드포인트, 원본 합쳐진 선분, convex hull, 선택된 두 점 ---
        # 스켈레톤 + 클러스터링된 엔드포인트
        skel_vis = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        for (x, y) in clustered_eps:
            cv2.circle(skel_vis, (x, y), 4, (255, 0, 0), -1)

        # 라인 시각화
        line_vis = cv2.cvtColor(connected, cv2.COLOR_GRAY2BGR)
        # 모든 병합 라인(연한 회색)
        for (p1, p2) in merged_lines:
            cv2.line(line_vis, p1, p2, (200, 200, 200), 1)
        
        # 후보 리드들 시각화 (파란색) - 디버깅용
        if 'candidate_tips' in locals():
            for i, (tip, base, _, _) in enumerate(candidate_tips[:5]):  # 상위 5개만
                cv2.line(line_vis, base, tip, (255, 100, 0), 2)
                cv2.putText(line_vis, str(i+1), tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        
        # convex hull (초록)
        if hull is not None:
            cv2.drawContours(line_vis, [hull], -1, (0, 255, 0), 2)
        # 선택된 최종 두 점 (노랑)
        for (x, y) in final_tips:
            cv2.circle(line_vis, (x, y), 7, (0, 255, 255), -1)

        # --- 전체 시각화: 2×4 타일 ---
        def to_bgr(x):
            return x if len(x.shape) == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

        resize_size = (w // 2, h // 2)
        imgs = [
            to_bgr(img),                # 1. Original
            to_bgr(gray),               # 2. Gray
            to_bgr(blurred),            # 3. Blurred
            to_bgr(gray_clahe),         # 4. CLAHE
            to_bgr(mask_inv),           # 5. Thresh Inv
            to_bgr(mask_small_removed), # 6. Small Removed
            to_bgr(skel_vis),           # 7. Skeleton + Clustered Endpoints
            to_bgr(line_vis)            # 8. Lines + Hull + Final Tips
        ]
        tiles = [cv2.resize(v, resize_size) for v in imgs]
        top_row = np.hstack(tiles[0:4])
        bot_row = np.hstack(tiles[4:8])
        canvas = np.vstack((top_row, bot_row))

        # --- 디버깅 정보 추가 (개선사항 7) ---
        info_text = f"Image: {image_files[idx]}"
        info_text += f" | Candidates: {len(candidate_tips) if 'candidate_tips' in locals() else 0}"
        if final_tips:
            info_text += f" | Leads found: {len(final_tips)}"
            if len(final_tips) == 2:
                angle = calculate_angle_between_leads(final_tips[0], final_tips[1], hull_centroid)
                info_text += f" | Angle: {angle:.1f} deg"
            elif len(final_tips) == 1 and 'candidate_tips' in locals() and len(candidate_tips) > 1:
                # 두 번째 리드를 찾지 못한 이유 표시
                info_text += " | 2nd lead angle mismatch"
        
        # 정보 텍스트 배경 추가
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(canvas, (5, canvas.shape[0] - 35), (15 + text_size[0], canvas.shape[0] - 5), (0, 0, 0), -1)
        cv2.putText(canvas, info_text, (10, canvas.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win_name, canvas)

        key = cv2.waitKey(50) & 0xFF
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
                # 새로운 이미지가 로드될 때마다 hull 정보도 갱신
                hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask_red = cv2.bitwise_or(
                    cv2.inRange(hsv_full, r1_low, r1_high),
                    cv2.inRange(hsv_full, r2_low, r2_high)
                )
                mask_green = cv2.inRange(hsv_full, g_low, g_high)
                mask_yellow = cv2.inRange(hsv_full, y_low, y_high)
                mask_body = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green), mask_yellow)
                mask_body_clean = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel_body, iterations=2)
                mask_body_clean = cv2.morphologyEx(mask_body_clean, cv2.MORPH_CLOSE, kernel_body, iterations=2)
                conts_body, _ = cv2.findContours(mask_body_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if conts_body:
                    largest_cont = max(conts_body, key=lambda c: cv2.contourArea(c))
                    hull = cv2.convexHull(largest_cont)
                    M = cv2.moments(hull)
                    if M["m00"] != 0:
                        hull_centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                    else:
                        hull_centroid = None
                else:
                    hull = None
                    hull_centroid = None
            continue

    cv2.destroyAllWindows()