import cv2
import numpy as np
from skimage.morphology import skeletonize

class ResistorEndpointDetector:
    """
    CLAHE, 배경 조명 보정, 감마 보정, Adaptive Threshold, 스켈레톤 + Hough + 최장 거리 점 방식으로
    다양한 조도 환경에서도 저항 리드(endpoints) 검출 및 중간 과정을 시각화
    """
    def __init__(self,
                 clahe_clip=2.0,
                 clahe_grid=(8, 8),
                 bg_kernel=(51, 51),
                 gamma=1.2,
                 adapt_block=51,
                 adapt_C=5,
                 hough_threshold=50,
                 min_line_length_ratio=0.5,
                 max_line_gap=10,
                 morph_kernel_size=3,
                 morph_iterations=1,
                 min_skel_area=20,
                 visualize=False):
        # 대비 향상 및 보정 파라미터
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.bg_kernel = bg_kernel
        self.gamma = gamma
        self.adapt_block = adapt_block
        self.adapt_C = adapt_C
        # Hough 파라미터
        self.hough_threshold = hough_threshold
        self.min_line_length_ratio = min_line_length_ratio
        self.max_line_gap = max_line_gap
        # Morphology 파라미터
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (morph_kernel_size, morph_kernel_size))
        self.morph_iterations = morph_iterations
        self.min_skel_area = min_skel_area
        self.visualize = visualize

    def _preprocess(self, roi: np.ndarray):
        # 1) 그레이스케일 변환 + CLAHE
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                tileGridSize=self.clahe_grid)
        enhanced = clahe.apply(gray)
        # 2) 배경 조명 보정 (모폴로지 오프닝)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.bg_kernel)
        background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_bg)
        corrected = cv2.subtract(enhanced, background)
        # 3) 감마 보정
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(256)], dtype="uint8")
        gamma_corr = cv2.LUT(corrected, table)
        # 4) Adaptive Threshold (Gaussian)
        bw = cv2.adaptiveThreshold(
            gamma_corr, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adapt_block,
            self.adapt_C
        )
        # 5) 노이즈 제거 (Open/Close)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                              self.morph_kernel,
                              iterations=self.morph_iterations)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              self.morph_kernel,
                              iterations=self.morph_iterations)
        return gray, gamma_corr, bw

    def extract(self, image: np.ndarray, bbox: tuple):
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        gray, gamma_corr, bw = self._preprocess(roi)

        # 6) 스켈레톤화 및 작은 조각 제거
        skel = skeletonize(bw // 255).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        clean = np.zeros_like(skel)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_skel_area:
                clean[labels == i] = 255
        skel = clean

        # 7) 엔드포인트 검색
        endpoints = []
        h, w = skel.shape
        for yy in range(1, h - 1):
            for xx in range(1, w - 1):
                if skel[yy, xx] > 0:
                    neigh = int(np.sum(skel[yy-1:yy+2, xx-1:xx+2] > 0)) - 1
                    if neigh == 1:
                        endpoints.append((xx, yy))

        # 8) 동적 Canny 엣지 검출
        v = np.median(gamma_corr)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gamma_corr, lower, upper)

        # 9) HoughLinesP를 이용한 선 검출
        min_len = int(w * self.min_line_length_ratio)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            self.hough_threshold,
            minLineLength=min_len,
            maxLineGap=self.max_line_gap
        )
        pts = []
        hough_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for x3, y3, x4, y4 in lines.reshape(-1, 4):
                cv2.line(hough_img, (x3, y3), (x4, y4), (255, 0, 0), 1)
                pts.extend([(x3, y3), (x4, y4)])

        # 10) 최장 거리 점 선택 (엔드포인트 우선)
        data_pts = (np.array(endpoints) if len(endpoints) >= 2
                    else (np.array(pts) if len(pts) >= 2 else None))
        centers = []
        if data_pts is not None:
            max_dist = -1
            p1 = p2 = None
            for i in range(len(data_pts)):
                for j in range(i + 1, len(data_pts)):
                    d = (data_pts[i][0] - data_pts[j][0])**2 + \
                        (data_pts[i][1] - data_pts[j][1])**2
                    if d > max_dist:
                        max_dist = d
                        p1, p2 = tuple(data_pts[i]), tuple(data_pts[j])
            if p1 and p2:
                centers = [p1, p2]

        # 11) 시각화 (6 패널 그리드)
        def to_bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        viz = [
            cv2.resize(roi, (w, h)),
            to_bgr(bw),
            to_bgr(skel),
            to_bgr(gray),
            cv2.resize(hough_img, (w, h)),
            to_bgr(gray)
        ]
        ep_img, cen_img = viz[3].copy(), viz[5].copy()
        for ex, ey in endpoints:
            cv2.circle(ep_img, (ex, ey), 3, (0, 0, 255), -1)
        for cx, cy in centers:
            cv2.circle(cen_img, (cx, cy), 5, (0, 255, 0), -1)
        viz[3], viz[5] = ep_img, cen_img
        grid = np.vstack([np.hstack(viz[:3]), np.hstack(viz[3:])])
        if self.visualize:
            cv2.imshow('Process Overview', grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 12) 최종 엔드포인트 반환 (원본 좌표계)
        if centers:
            return (
                (x1 + centers[0][0], y1 + centers[0][1]),
                (x1 + centers[1][0], y1 + centers[1][1])
            )
        return None

    def draw(self, image: np.ndarray, endpoints: tuple,
             color=(0, 255, 0), radius=5, thickness=-1):
        if endpoints:
            cv2.circle(image, endpoints[0], radius, color, thickness)
            cv2.circle(image, endpoints[1], radius, color, thickness)
