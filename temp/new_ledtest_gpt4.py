import cv2
import numpy as np
import os
import itertools
from collections import deque
from skimage.morphology import skeletonize
from hole_detector2 import HoleDetector  # HoleDetector를 이용해 구멍 중심을 검출

class SilverGrayLineDetector:
    """
    주요 파이프라인 (LED 끝점 검출 전용):
      1) 컬러 노이즈 제거 → CLAHE → 은색/회색 마스크 → Canny → 모폴로지 → HoughLinesP
      2) AdaptiveThresh → 구멍 검출 (HoleDetector) → 사각형으로 완전 제거 → gray_mask_pre
      3) LED 몸체 ROI 검출 (HSV)
      4) gray_mask_pre → skeletonize → skel, all_endpoints
      5) skel 내에서 ‘LED 몸체 ROI 내부 픽셀’을 다중 소스로 BFS → dist_map 계산
      6) dist_map 기반 가장 먼 두 엔드포인트를 final_pair
      7) (Fallback) 필요 시 HoughLinesP 기반 far_pair
    """
    def __init__(self):
        # Canny 파라미터
        self.canny_th1 = 50
        self.canny_th2 = 150
        # HoughLinesP 파라미터
        self.hough_th = 50
        self.min_line_len = 30
        self.max_line_gap = 5
        # Morphology Closing
        self.kernel1_h = 7
        self.kernel2_h = 15
        # LAB 은색 마스크 파라미터
        self.l_offset = 20
        self.ab_thresh = 10
        # AdaptiveThreshold 회색 마스크 파라미터
        self.adapt_block_size = 15
        self.adapt_C = 5
        # 작은 컨투어 임계 (AdaptiveThresh 단계)
        self.hole_area_thresh = 15

    # -------------------------------------------------
    # 1) 컬러 노이즈 제거 (빨간/파란 라인 제거)
    # -------------------------------------------------
    def remove_color_noise(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 빨간 영역 (두 구간)
        lr1 = np.array([0, 50, 50], dtype=np.uint8)
        ur1 = np.array([10, 255, 255], dtype=np.uint8)
        lr2 = np.array([170, 50, 50], dtype=np.uint8)
        ur2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_r1 = cv2.inRange(hsv, lr1, ur1)
        mask_r2 = cv2.inRange(hsv, lr2, ur2)
        red_mask = cv2.bitwise_or(mask_r1, mask_r2)
        # 파란 영역
        lb = np.array([100, 50, 50], dtype=np.uint8)
        ub = np.array([140, 255, 255], dtype=np.uint8)
        blue_mask = cv2.inRange(hsv, lb, ub)
        # 결합 후 팽창 → inpaint
        color_mask = cv2.bitwise_or(red_mask, blue_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        color_mask = cv2.dilate(color_mask, kernel, iterations=1)
        cleaned = cv2.inpaint(img, color_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cleaned, color_mask

    # -------------------------------------------------
    # 2) CLAHE
    # -------------------------------------------------
    def apply_clahe(self, gray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # -------------------------------------------------
    # 3) 은색 마스크 (LAB) → (구멍 제거는 나중에 회색 마스크에서)
    # -------------------------------------------------
    def silver_mask_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        Lm = int(np.mean(L))
        lt = min(Lm + self.l_offset, 255)
        condL = (L > lt).astype(np.uint8) * 255
        condA = (np.abs(A.astype(np.int16) - 128) < self.ab_thresh).astype(np.uint8) * 255
        condB = (np.abs(B.astype(np.int16) - 128) < self.ab_thresh).astype(np.uint8) * 255
        mask = cv2.bitwise_and(condL, cv2.bitwise_and(condA, condB))
        return mask

    # -------------------------------------------------
    # 4) 회색 마스크 (AdaptiveThresh) → 구멍 제거 전처리
    # -------------------------------------------------
    def gray_mask_adaptive(self, gray_clahe):
        thr = cv2.adaptiveThreshold(
            gray_clahe, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            self.adapt_block_size, self.adapt_C
        )
        # AdaptiveThresh의 결과를 반전: 전경(핀/LED) 흰색, 배경/구멍 검정
        mask = cv2.bitwise_not(thr)
        # 작은 컨투어(노이즈) 제거
        hole_mask = np.zeros_like(mask)
        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in conts:
            if cv2.contourArea(cnt) < self.hole_area_thresh:
                cv2.drawContours(hole_mask, [cnt], -1, 255, -1)
        mask_clean = cv2.bitwise_and(mask, cv2.bitwise_not(hole_mask))
        return mask_clean  # gray_mask_pre

    # -------------------------------------------------
    # 5) 다중 스케일 Canny
    # -------------------------------------------------
    def multi_scale_canny(self, gray_clahe):
        multi = np.zeros_like(gray_clahe)
        for s in [1.0, 2.0, 3.0]:
            blur = cv2.GaussianBlur(gray_clahe, (0, 0), sigmaX=s)
            edges = cv2.Canny(blur, self.canny_th1, self.canny_th2)
            multi = cv2.bitwise_or(multi, edges)
        return multi

    # -------------------------------------------------
    # 6) Morphology Closing (두 단계)
    # -------------------------------------------------
    def morphological_closing(self, edges_mask):
        k1 = max(1, self.kernel1_h)
        if k1 % 2 == 0: k1 += 1
        ker1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k1))
        c1 = cv2.morphologyEx(edges_mask, cv2.MORPH_CLOSE, ker1)
        k2 = max(3, self.kernel2_h)
        if k2 % 2 == 0: k2 += 1
        ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k2))
        c2 = cv2.morphologyEx(c1, cv2.MORPH_CLOSE, ker2)
        return c2

    # -------------------------------------------------
    # 7) HoughLinesP → 수평/수직 선분 검출 (Fallback)
    # -------------------------------------------------
    def detect_lines(self, closed_edges):
        lines = cv2.HoughLinesP(
            closed_edges, rho=1, theta=np.pi/180,
            threshold=self.hough_th,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap
        )
        if lines is None:
            return []
        cand = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            L = np.hypot(x2 - x1, y2 - y1)
            if L < self.min_line_len:
                continue
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if not ((ang <= 10) or (ang >= 170)):
                continue
            cand.append((x1, y1, x2, y2))
        cand.sort(key=lambda ln: np.hypot(ln[2]-ln[0], ln[3]-ln[1]), reverse=True)
        return cand[:2]

    # -------------------------------------------------
    # 8) 전체 파이프라인 (구멍 제거는 run_with_trackbars에서 처리)
    # -------------------------------------------------
    def process_image(self, img):
        # 1) 컬러 노이즈 제거
        img_cleaned, color_mask = self.remove_color_noise(img)

        # 2) 그레이 + CLAHE
        gray = cv2.cvtColor(img_cleaned, cv2.COLOR_BGR2GRAY)
        gray_clahe = self.apply_clahe(gray)

        # 3) 은색 마스크 (LAB) → (구멍 제거는 회색 마스크에서)
        silver_mask = self.silver_mask_lab(img_cleaned)

        # 4) 회색 마스크 (AdaptiveThresh) → 구멍 제거 전처리
        gray_mask_pre = self.gray_mask_adaptive(gray_clahe)

        # 5) Multi-scale Canny
        multi_edges = self.multi_scale_canny(gray_clahe)

        # 6) 은색/회색 엣지 추출
        edges_silver = cv2.bitwise_and(multi_edges, multi_edges, mask=silver_mask)
        edges_gray   = cv2.bitwise_and(multi_edges, multi_edges, mask=gray_mask_pre)

        # 7) Morphology Closing
        closed_silver = self.morphological_closing(edges_silver)
        closed_gray   = self.morphological_closing(edges_gray)

        # 8) HoughLinesP (선 분리용)
        silver_lines = self.detect_lines(closed_silver)
        gray_lines   = self.detect_lines(closed_gray)

        # 9) annotated_image (은색=파랑, 회색=초록)
        vis = img_cleaned.copy()
        for (x1, y1, x2, y2) in silver_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(vis, (x1, y1), 4, (255, 0, 0), -1)
            cv2.circle(vis, (x2, y2), 4, (255, 0, 0), -1)
        for (x1, y1, x2, y2) in gray_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (x1, y1), 4, (0, 255, 0), -1)
            cv2.circle(vis, (x2, y2), 4, (0, 255, 0), -1)

        return {
            'color_mask': color_mask,
            'img_cleaned': img_cleaned,
            'preproc': {
                'gray_clean': gray,
                'gray_clahe': gray_clahe,
                'silver_mask': silver_mask,
                'gray_mask_pre': gray_mask_pre,   # 수정된 키
                'multi_edges': multi_edges,
                'edges_silver': edges_silver,
                'edges_gray': edges_gray,
                'closed_silver': closed_silver,
                'closed_gray': closed_gray
            },
            'silver_lines': silver_lines,
            'gray_lines': gray_lines,
            'annotated_image': vis
        }


# ============================================================================
# 추가 함수들 (스켈레톤, BFS 거리, LED 몸체 검출 등)
# ============================================================================

def skeleton_endpoints(binary_img):
    """
    스켈레톤화 후 엔드포인트 좌표 반환
    """
    bw = (binary_img // 255).astype(np.uint8)
    skel = skeletonize(bw).astype(np.uint8) * 255
    endpoints = []
    h, w = skel.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y, x] == 255:
                nb = np.count_nonzero(skel[y-1:y+2, x-1:x+2]) - 1
                if nb == 1:
                    endpoints.append((x, y))
    return skel, endpoints

def multi_source_bfs_distance(skel, led_roi):
    """
    skel: 0/255 스켈레톤 바이너리
    led_roi: (bx, by, bw, bh) - LED 몸체 ROI (확장 포함)
    반환: dist_map: LED ROI 내부 픽셀을 거리 0으로 하여
              연결된 스켈레톤 픽셀 전체에 대해 BFS로 거리 계산 (ROI 외 픽셀은 -1)
    """
    h, w = skel.shape
    dist_map = -np.ones((h, w), dtype=np.int32)
    bx, by, bw_, bh_ = led_roi

    q = deque()
    for yy in range(by, by + bh_):
        for xx in range(bx, bx + bw_):
            if 0 <= xx < w and 0 <= yy < h and skel[yy, xx] == 255:
                dist_map[yy, xx] = 0
                q.append((xx, yy))

    dirs = [(-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)]
    while q:
        x, y = q.popleft()
        d0 = dist_map[y, x]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if skel[ny, nx] == 255 and dist_map[ny, nx] < 0:
                    dist_map[ny, nx] = d0 + 1
                    q.append((nx, ny))

    return dist_map

def detect_led_bodies(img_cleaned):
    """
    노랑/초록/빨강 LED 몸체를 HSV 기반으로 분리하여 contour → (x,y,w,h) 반환
    """
    hsv = cv2.cvtColor(img_cleaned, cv2.COLOR_BGR2HSV)
    # 빨강 LED
    lr1 = np.array([0, 100, 100], dtype=np.uint8)
    ur1 = np.array([10, 255, 255], dtype=np.uint8)
    lr2 = np.array([160, 100, 100], dtype=np.uint8)
    ur2 = np.array([180, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lr1, ur1)
    m2 = cv2.inRange(hsv, lr2, ur2)
    mask_red = cv2.bitwise_or(m1, m2)
    # 초록 LED
    lg = np.array([40, 80, 80], dtype=np.uint8)
    ug = np.array([80, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lg, ug)
    # 노랑 LED
    ly = np.array([20, 80, 80], dtype=np.uint8)
    uy = np.array([35, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, ly, uy)
    mask_led = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, mask_yellow))

    conts, _ = cv2.findContours(mask_led, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    led_bodies = []
    for cnt in conts:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0: continue
        asp = w / float(h)
        if 0.8 < asp < 1.2 and w*h > 100:
            led_bodies.append((x, y, w, h))
    return led_bodies

def hough_endpoints_from_mask(mask_binary, canny_th1, canny_th2,
                              hough_th, min_len, max_gap):
    """
    mask_binary: 0/255 이진 이미지 (AdaptiveThresh → 구멍 제거 후)
    1) GaussianBlur → Canny → Closing
    2) HoughLinesP로 선 검출
    3) 길이 상위 2개 선분을 뽑고, 끝점 네 개 중 가장 먼 두 점을 반환
    """
    blur = cv2.GaussianBlur(mask_binary, (5, 5), sigmaX=0)
    edges = cv2.Canny(blur, canny_th1, canny_th2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    raw_lines = cv2.HoughLinesP(
        closed, rho=1, theta=np.pi/180,
        threshold=hough_th,
        minLineLength=min_len,
        maxLineGap=max_gap
    )

    candidates = []
    if raw_lines is not None:
        for ln in raw_lines:
            x1, y1, x2, y2 = ln[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length < min_len:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if not (angle <= 10 or angle >= 170):
                continue
            candidates.append((length, (x1, y1, x2, y2)))

    candidates.sort(key=lambda x: x[0], reverse=True)
    lines2 = [ln for _, ln in candidates[:2]]

    far_pair = None
    if len(lines2) == 2:
        pts = []
        for (x1, y1, x2, y2) in lines2:
            pts.append((x1, y1))
            pts.append((x2, y2))
        pairs = []
        for (p1, p2) in itertools.combinations(pts, 2):
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dist2 = dx*dx + dy*dy
            pairs.append((dist2, p1, p2))
        pairs.sort(key=lambda x: x[0], reverse=True)
        far_pair = (pairs[0][1], pairs[0][2])

    return lines2, far_pair, closed

# ------------------------------------------
# 빈 콜백 (트랙바용)
# ------------------------------------------
def nothing(x):
    pass

def run_with_trackbars():
    image_files = [
        f for f in os.listdir('.')
        if 'capacitor' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not image_files:
        print("현재 폴더에 'led'가 포함된 이미지 파일이 없습니다.")
        return

    detector = SilverGrayLineDetector()
    win_name = "LED Endpoint Detection (n/p: switch | q: quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # --- 트랙바 생성 ---
    # 1) Canny
    cv2.createTrackbar('Canny Th1', win_name, detector.canny_th1, 500, nothing)
    cv2.createTrackbar('Canny Th2', win_name, detector.canny_th2, 500, nothing)
    # 2) HoughLinesP
    cv2.createTrackbar('Hough Th', win_name, detector.hough_th, 200, nothing)
    cv2.createTrackbar('Min Line Len', win_name, detector.min_line_len, 200, nothing)
    cv2.createTrackbar('Max Line Gap', win_name, detector.max_line_gap, 50, nothing)
    # 3) Morphology
    cv2.createTrackbar('Kernel1 h', win_name, detector.kernel1_h, 31, nothing)
    cv2.createTrackbar('Kernel2 h', win_name, detector.kernel2_h, 61, nothing)
    # 4) LAB 은색 마스크
    cv2.createTrackbar('l_offset', win_name, detector.l_offset, 100, nothing)
    cv2.createTrackbar('ab_thresh', win_name, detector.ab_thresh, 50, nothing)
    # 5) Adaptive Threshold (회색 마스크)
    cv2.createTrackbar('adapt_bsz', win_name, detector.adapt_block_size, 51, nothing)
    cv2.createTrackbar('adapt_C', win_name, detector.adapt_C, 20, nothing)
    # 6) 작은 컨투어 제거 임계
    cv2.createTrackbar('hole_area_thresh', win_name, detector.hole_area_thresh, 200, nothing)

    idx = 0
    hd = HoleDetector()  # HoleDetector 인스턴스 (구멍 검출용)

    while True:
        fname = image_files[idx]
        img = cv2.imread(fname)
        if img is None:
            print(f"{fname}를 읽을 수 없습니다. 넘어갑니다.")
            idx = (idx + 1) % len(image_files)
            continue

        # --- 트랙바 값 읽기 & 반영 ---
        detector.canny_th1 = cv2.getTrackbarPos('Canny Th1', win_name)
        detector.canny_th2 = cv2.getTrackbarPos('Canny Th2', win_name)
        if detector.canny_th2 < detector.canny_th1:
            detector.canny_th2 = detector.canny_th1 + 1

        detector.hough_th = cv2.getTrackbarPos('Hough Th', win_name)
        detector.min_line_len = max(1, cv2.getTrackbarPos('Min Line Len', win_name))
        detector.max_line_gap = cv2.getTrackbarPos('Max Line Gap', win_name)

        k1 = cv2.getTrackbarPos('Kernel1 h', win_name)
        if k1 < 1: k1 = 1
        if k1 % 2 == 0: k1 += 1
        detector.kernel1_h = k1
        k2 = cv2.getTrackbarPos('Kernel2 h', win_name)
        if k2 < 3: k2 = 3
        if k2 % 2 == 0: k2 += 1
        detector.kernel2_h = k2

        detector.l_offset = cv2.getTrackbarPos('l_offset', win_name)
        if detector.l_offset < 0: detector.l_offset = 0
        detector.ab_thresh = cv2.getTrackbarPos('ab_thresh', win_name)
        if detector.ab_thresh < 0: detector.ab_thresh = 0

        absz = cv2.getTrackbarPos('adapt_bsz', win_name)
        if absz < 3: absz = 3
        if absz % 2 == 0: absz += 1
        detector.adapt_block_size = absz
        detector.adapt_C = cv2.getTrackbarPos('adapt_C', win_name)

        detector.hole_area_thresh = cv2.getTrackbarPos('hole_area_thresh', win_name)
        if detector.hole_area_thresh < 0:
            detector.hole_area_thresh = 0

        # --- 기존 파이프라인 실행 ---
        result = detector.process_image(img)
        vis_final = result['annotated_image']
        pre = result['preproc']
        silver_lines = result['silver_lines']
        gray_lines = result['gray_lines']
        color_mask = result['color_mask']
        img_cleaned = result['img_cleaned']

        # --- (A) gray_mask_pre: AdaptiveThresh 단계에서 구멍 제거 전 이진 마스크 ---
        gray_mask_pre = pre['gray_mask_pre']

        # =====================================================
        # === (B) 구멍 완전 제거: HoleDetector → 사각형으로 덮어 버림 ===
        # =====================================================
        gray_mask = gray_mask_pre.copy()
        centers = hd.detect_holes_raw(img_cleaned)
        h_img, w_img = gray_mask.shape[:2]
        half_size = 14
        for cx, cy in centers:
            x_int = int(round(cx))
            y_int = int(round(cy))
            x0 = max(x_int - half_size, 0)
            y0 = max(y_int - half_size, 0)
            x1 = min(x_int + half_size, w_img - 1)
            y1 = min(y_int + half_size, h_img - 1)
            cv2.rectangle(gray_mask, (x0, y0), (x1, y1), 0, -1)

        # =====================================================
        # === 3) LED 몸체 ROI 검출 ===
        # =====================================================
        led_bodies = detect_led_bodies(img_cleaned)

        # =====================================================
        # === 4) 스켈레톤 + 엔드포인트 ===
        # =====================================================
        skel, all_endpoints = skeleton_endpoints(gray_mask)

        # =====================================================
        # === 5) 스켈레톤 BFS 거리 맵 생성 (LED ROI 다중 소스) ===
        # =====================================================
        final_pair = None
        connected_eps = []
        closed_mask_edges = None

        if led_bodies:
            bx, by, bw_, bh_ = led_bodies[0]
            margin = 10
            bx0 = max(bx - margin, 0)
            by0 = max(by - margin, 0)
            bx1 = min(bx + bw_ + margin, skel.shape[1] - 1)
            by1 = min(by + bh_ + margin, skel.shape[0] - 1)
            led_roi = (bx0, by0, bx1 - bx0, by1 - by0)

            dist_map = multi_source_bfs_distance(skel, led_roi)
            connected_eps = [(x, y) for (x, y) in all_endpoints if dist_map[y, x] >= 0]

            if connected_eps:
                dist_eps = [(dist_map[y, x], (x, y)) for (x, y) in connected_eps]
                dist_eps.sort(key=lambda e: e[0], reverse=True)
                if len(dist_eps) >= 2:
                    final_pair = (dist_eps[0][1], dist_eps[1][1])
                elif len(dist_eps) == 1:
                    final_pair = (dist_eps[0][1], dist_eps[0][1])

            _, _, closed_mask_edges = hough_endpoints_from_mask(
                gray_mask,
                detector.canny_th1, detector.canny_th2,
                detector.hough_th, detector.min_line_len, detector.max_line_gap
            )

        # =====================================================
        # === 6) Fallback: HoughLinesP 기반 far_pair ===
        # =====================================================
        if final_pair is None:
            lines2, far_pair, closed_mask_edges = hough_endpoints_from_mask(
                gray_mask,
                detector.canny_th1, detector.canny_th2,
                detector.hough_th, detector.min_line_len, detector.max_line_gap
            )
            final_pair = far_pair

        # =====================================================
        # === 7) 시각화: 최종 엔드포인트 & 각 단계 결과 ===
        # =====================================================
        vis_colored = vis_final.copy()

        # (A) LED 몸체 ROI
        for (bx, by, bw_, bh_) in led_bodies:
            cv2.rectangle(vis_colored, (bx, by), (bx + bw_, by + bh_), (0, 255, 255), 2)

        # (B) 스켈레톤 엔드포인트 (파란 점)
        for (ex, ey) in all_endpoints:
            cv2.circle(vis_colored, (ex, ey), 3, (255, 0, 0), -1)

        # (C) 거리 맵 기반 엔드포인트 (초록 점)
        for (x, y) in connected_eps:
            cv2.circle(vis_colored, (x, y), 4, (0, 255, 0), -1)

        # (D) HoughLinesP 선분 (흰 선, 디버깅용)
        lines2, _, _ = hough_endpoints_from_mask(
            gray_mask,
            detector.canny_th1, detector.canny_th2,
            detector.hough_th, detector.min_line_len, detector.max_line_gap
        )
        for (x1, y1, x2, y2) in lines2:
            cv2.line(vis_colored, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # (E) 최종 엔드포인트 (빨간 점)
        if final_pair is not None:
            (px1, py1), (px2, py2) = final_pair
            cv2.circle(vis_colored, (px1, py1), 7, (0, 0, 255), -1)
            cv2.circle(vis_colored, (px2, py2), 7, (0, 0, 255), -1)

        # =====================================================
        # === 8) 결과 윈도우 띄우기 ===
        # =====================================================
        cv2.imshow("Gray Mask (After Hole Removal)", gray_mask)
        cv2.imshow("Skeleton EP", skel)
        cv2.imshow("Hough Mask Edges", closed_mask_edges)
        cv2.imshow("Final with Endpoints", vis_colored)

        # =====================================================
        # === 9) 기존 12패널 전처리 뷰 + annotated_image ===
        # =====================================================
        h_img, w_img = img.shape[:2]
        half_w = w_img // 2
        third_h = h_img // 3

        cm = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        cm_resized = cv2.resize(cm, (half_w, third_h))

        cleaned_resized = cv2.resize(img_cleaned, (half_w, third_h))

        gray_clean = pre['gray_clean']
        gc_resized = cv2.cvtColor(cv2.resize(gray_clean, (half_w, third_h)), cv2.COLOR_GRAY2BGR)

        gray_clahe = pre['gray_clahe']
        gch_resized = cv2.cvtColor(cv2.resize(gray_clahe, (half_w, third_h)), cv2.COLOR_GRAY2BGR)

        smask = cv2.cvtColor(pre['silver_mask'], cv2.COLOR_GRAY2BGR)
        smask_resized = cv2.resize(smask, (half_w, third_h))

        # 수정: 'gray_mask_pre' 키 사용
        gmask_pre = cv2.cvtColor(pre['gray_mask_pre'], cv2.COLOR_GRAY2BGR)
        gmask_pre_resized = cv2.resize(gmask_pre, (half_w, third_h))

        me = cv2.cvtColor(pre['multi_edges'], cv2.COLOR_GRAY2BGR)
        me_resized = cv2.resize(me, (half_w, third_h))

        es = cv2.cvtColor(pre['edges_silver'], cv2.COLOR_GRAY2BGR)
        es_resized = cv2.resize(es, (half_w, third_h))

        eg = cv2.cvtColor(pre['edges_gray'], cv2.COLOR_GRAY2BGR)
        eg_resized = cv2.resize(eg, (half_w, third_h))

        cs = cv2.cvtColor(pre['closed_silver'], cv2.COLOR_GRAY2BGR)
        cs_resized = cv2.resize(cs, (half_w, third_h))

        cg = cv2.cvtColor(pre['closed_gray'], cv2.COLOR_GRAY2BGR)
        cg_resized = cv2.resize(cg, (half_w, third_h))

        blank = np.zeros((third_h, half_w, 3), dtype=np.uint8)

        row1 = np.hstack((cm_resized, cleaned_resized))
        row2 = np.hstack((gc_resized, gch_resized))
        row3 = np.hstack((smask_resized, gmask_pre_resized))
        row4 = np.hstack((me_resized, es_resized))
        row5 = np.hstack((eg_resized, cs_resized))
        row6 = np.hstack((cg_resized, blank))

        preproc_view = np.vstack((row1, row2, row3, row4, row5, row6))

        half_h = h_img // 2
        vis_small = cv2.resize(vis_final, (2 * half_w, half_h))
        final_display = np.vstack((vis_small, preproc_view))

        overlay = final_display.copy()
        info_text = f"{fname} | Silver={len(silver_lines)} | Gray={len(gray_lines)}"
        cv2.putText(overlay, info_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win_name, overlay)

        # 키 입력 처리
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(image_files)
        elif key == ord('p'):
            idx = (idx - 1 + len(image_files)) % len(image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_with_trackbars()
