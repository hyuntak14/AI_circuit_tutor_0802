# detector/led_detector.py

import cv2
import numpy as np
from skimage.morphology import skeletonize

class LedEndpointDetector:
    def __init__(self,
                 clahe_clip=2.0, clahe_grid=(8,8),
                 bg_kernel=(51,51), gamma=1.2,
                 adapt_block=51, adapt_C=5,
                 morph_kernel_size=3, morph_iter=1,
                 hough_threshold=50, min_skel_area=20,
                 max_hole_dist=20,
                 hole_mask_radius=5,            # ← 이 줄 추가
                 visualize=False):
        # 전처리 파라미터
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.bg_kernel = bg_kernel
        self.gamma = gamma
        self.adapt_block = adapt_block
        self.adapt_C = adapt_C

        # Morphology 파라미터
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        self.morph_iter = morph_iter

        # Hough & 스켈레톤 필터
        self.hough_th = hough_threshold
        self.min_skel_area = min_skel_area

        # 홀 마스킹 & 매핑 반경
        self.max_hole_dist = max_hole_dist
        self.hole_mask_radius = hole_mask_radius    # ← 할당

        # 시각화 옵션
        self.visualize = visualize

    def _remove_color_lines(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0,50,50), (10,255,255))
        m2 = cv2.inRange(hsv, (170,50,50), (180,255,255))
        m3 = cv2.inRange(hsv, (100,50,50), (140,255,255))
        mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        return cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)

    def _preprocess(self, roi):
        clean = self._remove_color_lines(roi)
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(self.clahe_clip, self.clahe_grid)
        enhanced = clahe.apply(gray)

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.bg_kernel)
        bg = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kern)
        corr = cv2.subtract(enhanced, bg)

        inv = 1.0 / self.gamma
        lut = np.array([((i/255.)**inv)*255 for i in range(256)], np.uint8)
        gamma = cv2.LUT(corr, lut)

        bw = cv2.adaptiveThreshold(
            gamma, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adapt_block, self.adapt_C
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_iter)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iter)

        return gray, gamma, bw

    def extract(self, image, bbox, holes):
        
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        gray, gamma, bw = self._preprocess(roi)

        # ─── Hole masking ───────────────────────────────────────
        # detect_holes()로 얻은 (hx,hy) 리스트를 돌며,
        # ROI 좌표계로 변환한 후 원형 마스크 적용
        for hx, hy in holes:
            if x1 <= hx < x2 and y1 <= hy < y2:
                mx, my = hx - x1, hy - y1
                cv2.circle(bw, (mx, my), self.hole_mask_radius, 0, -1)
                cv2.circle(gray, (mx, my), self.hole_mask_radius, 0, -1)
        # ────────────────────────────────────────────────────────

        # skeletonize + 작은 조각 제거
        skel = skeletonize(bw//255).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
        clean = np.zeros_like(skel)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_skel_area:
                clean[labels == i] = 255
        skel = clean

        # 엔드포인트 검출
        eps = []
        h, w = skel.shape
        for yy in range(1, h-1):
            for xx in range(1, w-1):
                if skel[yy, xx]:
                    neigh = np.sum(skel[yy-1:yy+2, xx-1:xx+2] > 0) - 1
                    if neigh == 1:
                        eps.append((xx, yy))
        if len(eps) < 2:
            return None

        # 최장거리 두 점 선택
        pts = np.array(eps)
        maxd, p1, p2 = -1, None, None
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.sum((pts[i] - pts[j])**2)
                if d > maxd:
                    maxd, p1, p2 = d, tuple(pts[i]), tuple(pts[j])
        endpoints = [(p1[0]+x1, p1[1]+y1), (p2[0]+x1, p2[1]+y1)]

        # 홀 매핑
        mapped = []
        for ex, ey in endpoints:
            dists = [((ex-hx)**2 + (ey-hy)**2, idx) for idx, (hx, hy) in enumerate(holes)]
            d2, idx = min(dists)
            if np.sqrt(d2) <= self.max_hole_dist:
                mapped.append(idx)
            else:
                mapped.append(None)

        return {'endpoints': endpoints, 'holes': mapped}

    def draw(self, image, result, holes,
             pin_color=(0,0,255), hole_color=(0,255,0), r=5):
        if result is None:
            return
        for ex, ey in result['endpoints']:
            cv2.circle(image, (ex, ey), r, pin_color, -1)
        for idx in result['holes']:
            if idx is None:
                continue
            hx, hy = holes[idx]
            cv2.circle(image, (hx, hy), r, hole_color, 2)
        if self.visualize:
            cv2.imshow('LED Process', image)
            cv2.waitKey(1)
