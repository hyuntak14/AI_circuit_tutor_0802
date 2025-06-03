import cv2
import numpy as np
from hole_detector2 import HoleDetector
from skimage.morphology import skeletonize

# --- Import or define LSD detector that returns only raw line coordinates ---
def lsd_detect_raw(img_gray):
    """
    LSD로 선분을 검출하고, 선분 좌표 리스트만 반환합니다.
    (화면에 그려진 이미지는 여기서 생성하지 않습니다.)
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
    lines_to_draw: [(x1,y1,x2,y2), ...]
    """
    out = img_color.copy()
    for (x1, y1, x2, y2) in lines_to_draw:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return out

# --- 전처리 함수들 ---
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

# --- 스켈레톤 가지치기 함수 (이웃 기반) ---
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

# --- 엔드포인트 클러스터링 함수 ---
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

# --- 스켈레톤 기반 끝점 추출 함수 ---
def get_skeleton_endpoints(binary_img, enable_prune):
    bw = (binary_img // 255).astype(np.uint8)
    skel_bin = skeletonize(bw).astype(np.uint8)
    if enable_prune:
        skel_bin = prune_skeleton_neighborhood(skel_bin)
    endpoints = []
    H, W = skel_bin.shape
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skel_bin[y, x] == 1:
                nb = np.count_nonzero(skel_bin[y - 1:y + 2, x - 1:x + 2]) - 1
                if nb == 1:
                    endpoints.append((x, y))
    return cluster_endpoints(endpoints, thresh=5), skel_bin

# --- 그레이 마스크 기반 끝점 추출 함수 ---
def get_gray_mask_endpoints(img, sat_thresh=40, val_low=60, val_high=200):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = cv2.inRange(sat, sat_thresh, 255)
    mask = cv2.bitwise_and(mask, cv2.inRange(val, val_low, val_high))
    # 모폴로지 처리
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_gray = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, k, iterations=1)
    conts, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    endpoints = []
    for cnt in conts:
        if cv2.contourArea(cnt) < 20:
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
    return cluster_endpoints(endpoints, thresh=5), mask_gray

# --- 거리 계산 함수 ---
def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

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
    # HSV 범위 세팅 (색상별 분리)
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r1_low = np.array([0, 100, 100], dtype=np.uint8)
    r1_high = np.array([10, 255, 255], dtype=np.uint8)
    r2_low = np.array([160, 100, 100], dtype=np.uint8)
    r2_high = np.array([180, 255, 255], dtype=np.uint8)
    g_low = np.array([40, 50, 50], dtype=np.uint8)
    g_high = np.array([80, 255, 255], dtype=np.uint8)
    y_low = np.array([18, 80, 100], dtype=np.uint8)
    y_high = np.array([35, 255, 255], dtype=np.uint8)

    # 윈도우와 트랙바 설정
    win_name = "Lead Detection Improved"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w * 2, h * 2)
    # 트랙바 콜백
    def nothing(x):
        pass

    cv2.createTrackbar("hole_radius", win_name, 5, 50, nothing)
    cv2.createTrackbar("hole_area_thresh", win_name, 50, 500, nothing)
    cv2.createTrackbar("Canny_th1", win_name, 50, 300, nothing)
    cv2.createTrackbar("Canny_th2", win_name, 150, 300, nothing)
    cv2.createTrackbar("min_length", win_name, 20, max(w, h), nothing)
    cv2.createTrackbar("lead_thickness_min", win_name, 2, 10, nothing)
    cv2.createTrackbar("lead_thickness_max", win_name, 15, 30, nothing)
    cv2.createTrackbar("hull_proximity", win_name, 30, 100, nothing)
    cv2.createTrackbar("match_dist_thresh", win_name, 10, 50, nothing)
    cv2.createTrackbar("Use_Otsu", win_name, 0, 1, nothing)
    cv2.createTrackbar("Apply_Sharpen", win_name, 0, 1, nothing)
    cv2.createTrackbar("Apply_Smooth", win_name, 0, 1, nothing)
    cv2.createTrackbar("Enable_Pruning", win_name, 0, 1, nothing)
    cv2.createTrackbar("Gray_S", win_name, 30, 100, nothing)
    cv2.createTrackbar("Gray_V_low", win_name, 60, 255, nothing)
    cv2.createTrackbar("Gray_V_high", win_name, 200, 255, nothing)
    cv2.createTrackbar("Open_iter", win_name, 1, 5, nothing)
    cv2.createTrackbar("Close_iter", win_name, 1, 5, nothing)
    cv2.createTrackbar("BB_Open", win_name, 1, 5, nothing)
    cv2.createTrackbar("BB_Close", win_name, 1, 5, nothing)
    cv2.createTrackbar("BB_Erode", win_name, 1, 5, nothing)
    cv2.createTrackbar("BB_Dilate", win_name, 1, 5, nothing)
    cv2.createTrackbar("CLAHE_clip", win_name, 2, 10, nothing)
    cv2.createTrackbar("block_size", win_name, 11, 51, nothing)

    while True:
        # 트랙바 값 읽기
        hole_r = cv2.getTrackbarPos("hole_radius", win_name)
        hole_area = cv2.getTrackbarPos("hole_area_thresh", win_name)
        canny1 = cv2.getTrackbarPos("Canny_th1", win_name)
        canny2 = cv2.getTrackbarPos("Canny_th2", win_name)
        min_length = cv2.getTrackbarPos("min_length", win_name)
        if min_length < 1: min_length = 1
        lead_thickness_min = cv2.getTrackbarPos("lead_thickness_min", win_name)
        lead_thickness_max = cv2.getTrackbarPos("lead_thickness_max", win_name)
        hull_proximity = cv2.getTrackbarPos("hull_proximity", win_name)
        match_dist_thresh = cv2.getTrackbarPos("match_dist_thresh", win_name)
        use_otsu = cv2.getTrackbarPos("Use_Otsu", win_name) == 1
        apply_sharpen = cv2.getTrackbarPos("Apply_Sharpen", win_name) == 1
        apply_smooth = cv2.getTrackbarPos("Apply_Smooth", win_name) == 1
        enable_prune = cv2.getTrackbarPos("Enable_Pruning", win_name) == 1
        sat_thresh = cv2.getTrackbarPos("Gray_S", win_name)
        val_low = cv2.getTrackbarPos("Gray_V_low", win_name)
        val_high = cv2.getTrackbarPos("Gray_V_high", win_name)
        if val_low > val_high:
            val_low = val_high
        open_iter = cv2.getTrackbarPos("Open_iter", win_name)
        close_iter = cv2.getTrackbarPos("Close_iter", win_name)
        bb_open_iter = cv2.getTrackbarPos("BB_Open", win_name)
        bb_close_iter = cv2.getTrackbarPos("BB_Close", win_name)
        bb_erode_iter = cv2.getTrackbarPos("BB_Erode", win_name)
        bb_dilate_iter = cv2.getTrackbarPos("BB_Dilate", win_name)
        clahe_clip = cv2.getTrackbarPos("CLAHE_clip", win_name)
        blk = cv2.getTrackbarPos("block_size", win_name)
        if blk < 3: blk = 3
        if blk % 2 == 0: blk += 1

        # 1) 빨강/파랑 제거 → 그레이스케일
        img_clean = remove_red_blue(img)
        gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

        # 2) BilateralFilter 적용
        processed = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3) CLAHE 적용
        gray_clahe = apply_clahe(processed, clipLimit=clahe_clip)

        # 4) Adaptive Thresholding / Otsu 선택
        if use_otsu:
            _, thr = cv2.threshold(
                gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            mask_inv = cv2.bitwise_not(thr)
        else:
            thr = cv2.adaptiveThreshold(
                gray_clahe,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blk,
                5
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
                y1 = min(y_int + half, W - 1)
                cv2.rectangle(clean_mask, (x0, y0), (x1, y1), 0, -1)
            return clean_mask

        mask_no_holes = remove_hole(mask_small_removed, img_clean, hole_r)

        # 7) Morphological: Opening, Closing (반복 횟수 조정)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(mask_no_holes, cv2.MORPH_OPEN, k, iterations=open_iter)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=close_iter)

        # 8) Dilation → Erosion
        connected = cv2.dilate(closed, k, iterations=1)
        processed_morph = cv2.erode(connected, k, iterations=1)

        # 9-A) 스켈레톤 기반 끝점 검출
        skel_endpoints, skel_bin = get_skeleton_endpoints(processed_morph, enable_prune)

        # 9-B) 은색/회색 마스크 기반 끝점 검출
        gray_endpoints, mask_gray = get_gray_mask_endpoints(
            img_clean,
            sat_thresh=sat_thresh,
            val_low=val_low,
            val_high=val_high
        )

        # 10) LSD를 이용한 직선 검출 (raw) → 길이 10픽셀 이하 제거 → 필터된 선 시각화
        raw_lsd_lines = lsd_detect_raw(gray)  # 모든 선분 좌표
        filtered_lsd_lines = [
            (x1, y1, x2, y2)
            for (x1, y1, x2, y2) in raw_lsd_lines
            if np.hypot(x2 - x1, y2 - y1) > 50  # 길이가 10 픽셀보다 큰 경우만 남김 (> 10)
        ]
        # 필터링된 선분만 img_clean 위에 그려서 보여줌
        lsd_img = draw_lsd_lines(img_clean, filtered_lsd_lines)

        # === HoughLinesP로 직선 검출 (기존) ===
        edges = cv2.Canny(processed_morph, canny1, canny2)
        hough_lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50,
            minLineLength=min_length, maxLineGap=5
        )

        # 11) 후보 라인 필터링: 스켈레톤/그레이 끝점 근처에 있는 Hough 및 LSD 선분 합침
        candidate_lines = []
        # Hough 기반 선분
        if hough_lines is not None:
            for l in hough_lines:
                x1, y1, x2, y2 = l[0]
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                keep = False
                for (sx, sy) in skel_endpoints + gray_endpoints:
                    if distance(p1, (sx, sy)) <= match_dist_thresh or \
                       distance(p2, (sx, sy)) <= match_dist_thresh:
                        keep = True
                        break
                if keep:
                    candidate_lines.append((p1, p2))
        # LSD 기반 선분 (필터된 것만)
        for (x1, y1, x2, y2) in filtered_lsd_lines:
            p1 = (x1, y1)
            p2 = (x2, y2)
            keep = False
            for (sx, sy) in skel_endpoints + gray_endpoints:
                if distance(p1, (sx, sy)) <= match_dist_thresh or \
                   distance(p2, (sx, sy)) <= match_dist_thresh:
                    keep = True
                    break
            if keep:
                candidate_lines.append((p1, p2))

        merged_lines = candidate_lines  # 병합 없이 단순 합침

        # 12) Convex Hull 기반 최종 리드 끝점 후보 선택
        final_tips = []
        mask_body = cv2.bitwise_or(
            cv2.bitwise_or(
                cv2.inRange(hsv_full, r1_low, r1_high),
                cv2.inRange(hsv_full, r2_low, r2_high)
            ),
            cv2.bitwise_or(
                cv2.inRange(hsv_full, g_low, g_high),
                cv2.inRange(hsv_full, y_low, y_high)
            )
        )
        kernel_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_body_iter = 2
        close_body_iter = 2
        dilate_body_iter = 2
        mask_body_clean = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel_body, iterations=open_body_iter)
        mask_body_clean = cv2.morphologyEx(mask_body_clean, cv2.MORPH_CLOSE, kernel_body, iterations=close_body_iter)
        mask_body_clean = cv2.dilate(mask_body_clean, kernel_body, iterations=dilate_body_iter)
        conts_body, _ = cv2.findContours(mask_body_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = None
        hull_centroid = None
        if conts_body:
            largest_cont = max(conts_body, key=lambda c: cv2.contourArea(c))
            hull = cv2.convexHull(largest_cont)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                hull_centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])

        if hull is not None and hull_centroid is not None:
            candidate_tips = []
            for (p1, p2) in merged_lines:
                d1 = cv2.pointPolygonTest(hull, p1, True)
                d2 = cv2.pointPolygonTest(hull, p2, True)
                if -hull_proximity <= d1 <= hull_proximity and d2 < -hull_proximity:
                    candidate_tips.append((p2, p1, distance(p2, hull_centroid)))
                elif -hull_proximity <= d2 <= hull_proximity and d1 < -hull_proximity:
                    candidate_tips.append((p1, p2, distance(p1, hull_centroid)))
            candidate_tips.sort(key=lambda x: x[2], reverse=True)
            if len(candidate_tips) >= 2:
                final_tips.append(candidate_tips[0][0])
                for i in range(1, len(candidate_tips)):
                    if distance(final_tips[0], candidate_tips[i][0]) <= match_dist_thresh:
                        final_tips.append(candidate_tips[i][0])

        # --- 시각화 ---
        # 회색 마스크 + 끝점
        gray_display = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        for (x, y) in gray_endpoints:
            cv2.circle(gray_display, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Gray Mask + Endpoints", gray_display)

        # 스켈레톤 이진 영상
        cv2.imshow("Skeleton Binary", (skel_bin * 255).astype(np.uint8))
        # 스켈레톤 + 끝점
        skel_display = cv2.cvtColor((skel_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for (x, y) in skel_endpoints:
            cv2.circle(skel_display, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("Skeleton + Endpoints", skel_display)

        # **필터된 LSD 선분만 그려진 이미지**를 보여줍니다.
        cv2.imshow("LSD Lines (Filtered >10px)", lsd_img)

        # 최종 시각화: 합쳐진 선분 + 최종 끝점
        vis_final = cv2.cvtColor(processed_morph, cv2.COLOR_GRAY2BGR)
        for (p1, p2) in merged_lines:
            cv2.line(vis_final, p1, p2, (0, 255, 0), 1)
        for (x, y) in final_tips:
            cv2.circle(vis_final, (x, y), 7, (0, 255, 255), -1)

        def to_bgr(x):
            return x if len(x.shape) == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

        resize_size = (w // 2, h // 2)
        imgs = [
            to_bgr(img),          # 원본
            to_bgr(gray_display), # 회색 마스크 + 끝점
            to_bgr(skel_display), # 스켈레톤 + 끝점
            to_bgr(vis_final)     # LSD+Hough 기반 합쳐진 선분 + 최종 끝점
        ]
        tiles = [cv2.resize(v, resize_size) for v in imgs]
        top_row = np.hstack(tiles[0:2])
        bottom_row = np.hstack(tiles[2:4])
        grid = np.vstack((top_row, bottom_row))
        cv2.imshow("2x2 Grid", grid)

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
                # HSV 범위 재설정
                # (여기서는 그대로 두되, 필요시 트랙바를 통해 수정 가능)
    cv2.destroyAllWindows()
