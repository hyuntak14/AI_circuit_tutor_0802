import cv2
import numpy as np
from hole_detector2 import HoleDetector
from skimage.morphology import skeletonize
from collections import deque

# --- 스켈레톤 엔드포인트 검출 함수 ---
def skeleton_endpoints(binary_img):
    """
    스켈레톤화 후 엔드포인트 좌표 반환
    - binary_img: 0/255 이진 이미지
    """
    bw = (binary_img // 255).astype(np.uint8)
    skel = skeletonize(bw).astype(np.uint8) * 255
    endpoints = []
    h, w = skel.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 255:
                nb = np.count_nonzero(skel[y - 1 : y + 2, x - 1 : x + 2]) - 1
                if nb == 1:
                    endpoints.append((x, y))
    return skel, endpoints

# --- 엔드포인트 간 거리 계산 함수 ---
def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

# --- 전처리 함수들 ---
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

# --- 트랙바 콜백 함수(빈 함수) ---
def nothing(x):
    pass

# --- 메인 실행부 ---
if __name__ == "__main__":
    # 1) 폴더 내 'led' 또는 'cap'이 포함된 이미지 목록 가져오기
    image_files = [
        f for f in __import__('os').listdir('.')
        if any(k in f.lower() for k in ['led', 'cap']) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
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

    # 윈도우 이름 설정
    win_name = "Preprocess & Select Endpoints"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w * 2, h * 2)

    # 트랙바: block_size(홀수 3~51), C(0~20), CLAHE_clip(1~10), hole_radius(1~20), hole_area_thresh(1~100),
    #       Canny_th1(0~200), Canny_th2(0~300), min_ep_dist(1~100)
    cv2.createTrackbar("block_size",       win_name, 15, 51, nothing)
    cv2.createTrackbar("C",                win_name, 5, 20, nothing)
    cv2.createTrackbar("CLAHE_clip",       win_name, 2, 10, nothing)
    cv2.createTrackbar("hole_radius",      win_name, 7, 20, nothing)
    cv2.createTrackbar("hole_area_thresh", win_name, 15, 100, nothing)
    cv2.createTrackbar("Canny_th1",        win_name, 50, 200, nothing)
    cv2.createTrackbar("Canny_th2",        win_name, 150, 300, nothing)
    cv2.createTrackbar("min_ep_dist",      win_name, 20, 100, nothing)

    while True:
        h, w = img.shape[:2]

        # 트랙바 값 읽기
        blk = cv2.getTrackbarPos("block_size", win_name)
        if blk < 3: blk = 3
        if blk % 2 == 0: blk += 1
        Cval = cv2.getTrackbarPos("C", win_name)
        clahe_clip = cv2.getTrackbarPos("CLAHE_clip", win_name)
        hole_r = cv2.getTrackbarPos("hole_radius", win_name)
        hole_area = cv2.getTrackbarPos("hole_area_thresh", win_name)
        canny1 = cv2.getTrackbarPos("Canny_th1", win_name)
        canny2 = cv2.getTrackbarPos("Canny_th2", win_name)
        min_ep_dist = cv2.getTrackbarPos("min_ep_dist", win_name)
        if min_ep_dist < 1: min_ep_dist = 1

        # --- 1) 그레이스케일 ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- 2) Gaussian Blur ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- 3) CLAHE ---
        gray_clahe = apply_clahe(blurred, clipLimit=float(clahe_clip), tileGridSize=(8, 8))

        # --- 4) AdaptiveThreshold → 이진 마스크 생성 ---
        thr = cv2.adaptiveThreshold(
            gray_clahe, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blk, Cval
        )
        mask_inv = cv2.bitwise_not(thr)

        # --- 5) 작은 컨투어(노이즈) 제거 ---
        conts, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_mask = np.zeros_like(mask_inv)
        for cnt in conts:
            if cv2.contourArea(cnt) < hole_area:
                cv2.drawContours(small_mask, [cnt], -1, 255, -1)
        mask_small_removed = cv2.bitwise_and(mask_inv, cv2.bitwise_not(small_mask))

        # --- 6) HoleDetector 기반 구멍 제거 ---
        mask_no_holes = remove_holes(mask_small_removed, img, hole_radius=hole_r)

        # --- 7) Morphological Closing (3x3 커널) to 연결성 강화 ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(mask_no_holes, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- 8) Canny 엣지 (확인용) ---
        edges = cv2.Canny(closed, canny1, canny2)

        # --- 9) 스켈레톤화 및 엔드포인트 추출 (closed 사용) ---
        skel, endpoints = skeleton_endpoints(closed)

        # --- 10) 가장자리 근처 엔드포인트 제외 ---
        filtered_eps = []
        for (x, y) in endpoints:
            if x <= 1 or x >= w - 2 or y <= 1 or y >= h - 2:
                continue
            filtered_eps.append((x, y))

        # --- 11) 너무 긴 스켈레톤 분절에 해당하는 엔드포인트 제외 ---
        #    엔드포인트 쌍 간의 dx >= 0.9*w 또는 dy >= 0.9*h 이면 해당 쌍의 엔드포인트 둘 다 제거
        exclude = set()
        for i in range(len(filtered_eps)):
            for j in range(i + 1, len(filtered_eps)):
                p1 = filtered_eps[i]
                p2 = filtered_eps[j]
                if abs(p1[0] - p2[0]) >= 0.9 * w or abs(p1[1] - p2[1]) >= 0.9 * h:
                    exclude.add(p1)
                    exclude.add(p2)
        filtered_eps = [pt for pt in filtered_eps if pt not in exclude]

        # --- 12) 이미지 중앙 좌표를 기준으로 최장 거리 두 점 선택 (min_ep_dist 이상 분리) ---
        body_center = (w // 2, h // 2)
        final_pair = None
        if len(filtered_eps) >= 2:
            sorted_eps = sorted(filtered_eps, key=lambda p: distance(p, body_center), reverse=True)
            for i in range(len(sorted_eps)):
                p1 = sorted_eps[i]
                for candidate in sorted_eps[i + 1:]:
                    # 두 점 간 거리가 min_ep_dist 이상이어야 함
                    if distance(p1, candidate) < min_ep_dist:
                        continue
                    final_pair = (p1, candidate)
                    break
                if final_pair:
                    break

        # 만약 위에서 찾지 못했으면, 조건 만족하는 모든 쌍 중 중심으로부터 평균 거리가 최대인 쌍 선택
        if final_pair is None and len(filtered_eps) >= 2:
            best_score = -1
            best_pair = None
            for i in range(len(filtered_eps)):
                for j in range(i + 1, len(filtered_eps)):
                    p1, p2 = filtered_eps[i], filtered_eps[j]
                    if distance(p1, p2) < min_ep_dist:
                        continue
                    score = (distance(p1, body_center) + distance(p2, body_center)) / 2
                    if score > best_score:
                        best_score = score
                        best_pair = (p1, p2)
            final_pair = best_pair

        # --- 13) 시각화: 스켈레톤 + 후보 엔드포인트(청색) + 최종 선택된 두 점(빨강) + 이미지 중심(노랑) 표시 ---
        skel_bgr = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        for (x, y) in filtered_eps:
            cv2.circle(skel_bgr, (x, y), 3, (255, 0, 0), -1)

        if final_pair:
            (x1, y1), (x2, y2) = final_pair
            cv2.circle(skel_bgr, (x1, y1), 6, (0, 0, 255), -1)
            cv2.circle(skel_bgr, (x2, y2), 6, (0, 0, 255), -1)
            cv2.circle(skel_bgr, body_center, 4, (0, 255, 255), -1)
            cv2.line(skel_bgr, body_center, (x1, y1), (0, 255, 255), 1)
            cv2.line(skel_bgr, body_center, (x2, y2), (0, 255, 255), 1)

        # --- 14) 전체 시각화: 원본, 그레이, 블러, CLAHE, Thresh Inv, Small Removed, No Holes, Skeleton+Endpoints+FinalPair ---
        def to_bgr(x):
            return x if len(x.shape) == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

        resize_size = (w // 2, h // 2)
        imgs = [
            to_bgr(img),                  # Original
            to_bgr(gray),                 # Gray
            to_bgr(blurred),              # Blurred
            to_bgr(gray_clahe),           # CLAHE
            to_bgr(mask_inv),             # Thresh Inv
            to_bgr(mask_small_removed),   # Small Removed
            to_bgr(mask_no_holes),        # No Holes
            to_bgr(skel_bgr)              # Skeleton + Endpoints + Final Pair
        ]
        tiles = [cv2.resize(v, resize_size) for v in imgs]
        top_row = np.hstack(tiles[0:4])
        bot_row = np.hstack(tiles[4:8])
        canvas = np.vstack((top_row, bot_row))

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
            continue

    cv2.destroyAllWindows()
