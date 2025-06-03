import cv2
import numpy as np
import os
from hole_detector2 import HoleDetector
from skimage.morphology import skeletonize
from skimage.filters import frangi, sato

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
    return cv2.inpaint(img, mask_rb, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def apply_clahe(gray, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def remove_holes(mask, img, hole_radius=7):
    hd = HoleDetector()
    centers = hd.detect_holes_raw(img)
    clean_mask = mask.copy()
    h, w = clean_mask.shape[:2]
    half = hole_radius
    for cx, cy in centers:
        x_int, y_int = int(round(cx)), int(round(cy))
        x0 = max(x_int - half, 0)
        y0 = max(y_int - half, 0)
        x1 = min(x_int + half, w - 1)
        y1 = min(y_int + half, h - 1)
        cv2.rectangle(clean_mask, (x0, y0), (x1, y1), 0, -1)
    return clean_mask

def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_skeleton(binary_img):
    bw = (binary_img // 255).astype(np.uint8)
    skel = skeletonize(bw).astype(np.uint8) * 255
    return skel

def get_gray_mask(img_bgr, sat_thresh, val_low, val_high, open_iter, close_iter):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, val_low], dtype=np.uint8)
    upper_gray = np.array([179, sat_thresh, val_high], dtype=np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if open_iter > 0:
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return mask_gray

# --- 메인 ---
if __name__ == "__main__":
    # 이미지 파일 목록: 이름에 "led" 포함
    image_files = [
        f for f in os.listdir('.')
        if 'led' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not image_files:
        print("폴더 내에 'led'가 포함된 이미지가 없습니다.")
        exit()

    # Line Segment Detector 초기화
    lsd = cv2.createLineSegmentDetector(0)

    idx = 0
    while True:
        filename = image_files[idx]
        img = cv2.imread(filename)
        if img is None:
            idx = (idx + 1) % len(image_files)
            continue
        h, w = img.shape[:2]

        # --- 1) LED 몸체(노랑/빨강/초록) 마스크 → Convex Hull ---
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 빨강
        r1_low = np.array([0, 100, 100], dtype=np.uint8)
        r1_high = np.array([10, 255, 255], dtype=np.uint8)
        r2_low = np.array([160, 100, 100], dtype=np.uint8)
        r2_high = np.array([180, 255, 255], dtype=np.uint8)
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv_full, r1_low, r1_high),
            cv2.inRange(hsv_full, r2_low, r2_high)
        )
        # 초록
        g_low = np.array([40, 50, 50], dtype=np.uint8)
        g_high = np.array([80, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv_full, g_low, g_high)
        # 노랑
        y_low = np.array([18, 80, 100], dtype=np.uint8)
        y_high = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv_full, y_low, y_high)
        mask_body = cv2.bitwise_or(
            cv2.bitwise_or(mask_red, mask_green),
            mask_yellow
        )
        kernel_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel_body, iterations=2)
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel_body, iterations=2)
        mask_body = cv2.dilate(mask_body, kernel_body, iterations=2)
        conts_body, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = None
        hull_centroid = None
        if conts_body:
            largest_cont = max(conts_body, key=lambda c: cv2.contourArea(c))
            hull = cv2.convexHull(largest_cont)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                hull_centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])

        # --- 2) 전처리: 빨강/파랑 제거 → 그레이스케일 → BilateralFilter → CLAHE ---
        img_clean = remove_red_blue(img)
        gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
        processed = cv2.bilateralFilter(gray, 9, 75, 75)
        gray_clahe = apply_clahe(processed, clipLimit=2.0, tileGridSize=(8, 8))

        # --- 3) 이진화 및 모폴로지 (Opening → Closing) ---
        thr = cv2.adaptiveThreshold(
            gray_clahe, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15, 5
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)

        # --- 4) 색상 마스크 (회색/은색) 생성 ---
        mask_gray = get_gray_mask(
            img_clean,
            sat_thresh=30, val_low=60, val_high=200,
            open_iter=1, close_iter=1
        )

        # --- 5) 스켈레톤 생성 & 가지치기 ---
        skel_full = get_skeleton(closed)
        # 0/255 → 0/1로 변환 후 prune, 다시 255 곱하기
        skel_bin = (skel_full // 255).astype(np.uint8)
        pruned_skel_bin = prune_skeleton_neighborhood(skel_bin)
        pruned_skel = (pruned_skel_bin * 255).astype(np.uint8)

        # --- 6) OR 결합: 색상마스크 | 가지치기된 스켈레톤 ---
        or_mask = cv2.bitwise_and(mask_gray, pruned_skel)

        # --- 7) 다양한 선 검출 & 끝점 후보 선택 (Hull 주변 기준) ---
        match_dist_thresh = 20
        hull_proximity = 20

        # 7-1) Standard HoughLines
        hough_img = np.zeros((h, w, 3), dtype=np.uint8)
        hough_endpoints = []
        if hull is not None:
            lines = cv2.HoughLines(or_mask, 1, np.pi/180, 100)
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    p1 = (x1, y1)
                    p2 = (x2, y2)
                    d1 = cv2.pointPolygonTest(hull, p1, True)
                    d2 = cv2.pointPolygonTest(hull, p2, True)
                    if (-hull_proximity <= d1 <= hull_proximity and d2 < -hull_proximity) or \
                       (-hull_proximity <= d2 <= hull_proximity and d1 < -hull_proximity):
                        cv2.line(hough_img, p1, p2, (0, 0, 255), 1)
                        if -hull_proximity <= d1 <= hull_proximity:
                            hough_endpoints.append(p1)
                        else:
                            hough_endpoints.append(p2)

        # 7-2) Line Segment Detector (LSD)
        lsd_img = np.zeros((h, w, 3), dtype=np.uint8)
        lsd_endpoints = []
        if hull is not None:
            lsd_lines = lsd.detect(or_mask)[0]
            if lsd_lines is not None:
                for line in lsd_lines:
                    x1, y1, x2, y2 = map(int, line[0])
                    p1 = (x1, y1)
                    p2 = (x2, y2)
                    d1 = cv2.pointPolygonTest(hull, p1, True)
                    d2 = cv2.pointPolygonTest(hull, p2, True)
                    if (-hull_proximity <= d1 <= hull_proximity and d2 < -hull_proximity) or \
                       (-hull_proximity <= d2 <= hull_proximity and d1 < -hull_proximity):
                        cv2.line(lsd_img, p1, p2, (0, 255, 0), 1)
                        if -hull_proximity <= d1 <= hull_proximity:
                            lsd_endpoints.append(p1)
                        else:
                            lsd_endpoints.append(p2)

        # 7-3) Ridge Detection (Frangi & Sato) → 스켈레톤화 → HoughLinesP
        ridge_img = np.zeros((h, w, 3), dtype=np.uint8)
        rd_endpoints = []
        if hull is not None:
            norm_input = (or_mask.astype(np.float32) / 255.0)
            frangi_resp = frangi(norm_input)
            sato_resp = sato(norm_input)
            if np.max(frangi_resp) > 0:
                frangi_norm = (255 * (frangi_resp / np.max(frangi_resp))).astype(np.uint8)
            else:
                frangi_norm = np.zeros((h, w), dtype=np.uint8)
            if np.max(sato_resp) > 0:
                sato_norm = (255 * (sato_resp / np.max(sato_resp))).astype(np.uint8)
            else:
                sato_norm = np.zeros((h, w), dtype=np.uint8)

            frangi_skel = skeletonize((frangi_norm // 255).astype(np.uint8)).astype(np.uint8) * 255
            sato_skel = skeletonize((sato_norm // 255).astype(np.uint8)).astype(np.uint8) * 255
            for skel_mask in [frangi_skel, sato_skel]:
                edges = cv2.Canny(skel_mask, 50, 150)
                houghp = cv2.HoughLinesP(
                    edges, 1, np.pi/180,
                    threshold=50, minLineLength=20, maxLineGap=5
                )
                if houghp is not None:
                    for l in houghp:
                        x1, y1, x2, y2 = map(int, l[0])
                        p1 = (x1, y1)
                        p2 = (x2, y2)
                        d1 = cv2.pointPolygonTest(hull, p1, True)
                        d2 = cv2.pointPolygonTest(hull, p2, True)
                        if (-hull_proximity <= d1 <= hull_proximity and d2 < -hull_proximity) or \
                           (-hull_proximity <= d2 <= hull_proximity and d1 < -hull_proximity):
                            cv2.line(ridge_img, p1, p2, (255, 0, 0), 1)
                            if -hull_proximity <= d1 <= hull_proximity:
                                rd_endpoints.append(p1)
                            else:
                                rd_endpoints.append(p2)

        # --- 8) 최종 끝점 취합 & 표시 ---
        final_tips = list({*hough_endpoints, *lsd_endpoints, *rd_endpoints})
        final_vis = img.copy()
        for (x, y) in final_tips:
            cv2.circle(final_vis, (x, y), 5, (0, 255, 255), -1)

        # --- 9) 결과 창 ---
        cv2.imshow("Original", img)
        cv2.imshow("LED Body Mask (Hull)", mask_body)
        if hull is not None:
            hull_draw = img.copy()
            cv2.drawContours(hull_draw, [hull], -1, (255, 255, 0), 2)
            cv2.imshow("Convex Hull", hull_draw)
        cv2.imshow("Gray Mask", mask_gray)
        cv2.imshow("Pruned Skeleton", pruned_skel)
        cv2.imshow("OR Mask (Gray|Skeleton)", or_mask)
        cv2.imshow("Standard HoughLines", hough_img)
        cv2.imshow("LSD Lines", lsd_img)
        cv2.imshow("Ridge Lines (Frangi/Sato)", ridge_img)
        cv2.imshow("Final Lead Endpoints", final_vis)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(image_files)

    cv2.destroyAllWindows()
