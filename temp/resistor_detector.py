import cv2
import numpy as np
from skimage.morphology import skeletonize


class ResistorEndpointDetector:
    """
    스켈레톤 + HoughLinesP + 최장 거리 점 선택 방식으로
    저항 리드(endpoints) 검출 및 전체 중간 과정을 한 창에 시각화

    Args:
        hough_threshold (int): HoughLinesP 임계값
        min_line_length_ratio (float): ROI 너비 대비 최소 선 길이 비율
        max_line_gap (int): HoughLinesP 최대 선 갭
        morph_kernel_size (int): 열림/닫힘 커널 크기
        morph_iterations (int): 모폴로지 반복 횟수
        min_skel_area (int): 유지할 스켈레톤 객체 최소 픽셀 수
        visualize (bool): 중간 단계 시각화 여부
    """
    def __init__(self,
                 hough_threshold=50,
                 min_line_length_ratio=0.5,
                 max_line_gap=10,
                 morph_kernel_size=3,
                 morph_iterations=1,
                 min_skel_area=20,
                 visualize=False):
        self.hough_threshold = hough_threshold
        self.min_line_length_ratio = min_line_length_ratio
        self.max_line_gap = max_line_gap
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        self.morph_iterations = morph_iterations
        self.min_skel_area = min_skel_area
        self.visualize = visualize

    def extract(self, image: np.ndarray, bbox: tuple):
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1) 이진화 + 노이즈 제거
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_iterations)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iterations)

        # 2) 스켈레톤화 및 작은 조각 제거
        skel = skeletonize(bw // 255).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        clean = np.zeros_like(skel)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_skel_area:
                clean[labels == i] = 255
        skel = clean

        # 3) 엔드포인트 검색
        endpoints = []
        h, w = skel.shape
        for yy in range(1, h-1):
            for xx in range(1, w-1):
                if skel[yy, xx] > 0:
                    neigh = int(np.sum(skel[yy-1:yy+2, xx-1:xx+2] > 0)) - 1
                    if neigh == 1:
                        endpoints.append((xx, yy))

        # 4) HoughLinesP 엣지 및 선 검출
        edges = cv2.Canny(bw, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                minLineLength=int(w * self.min_line_length_ratio),
                                maxLineGap=self.max_line_gap)
        pts = []
        hough_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for x3, y3, x4, y4 in lines.reshape(-1, 4):
                cv2.line(hough_img, (x3, y3), (x4, y4), (255, 0, 0), 1)
                pts.extend([(x3, y3), (x4, y4)])

        # 5) 최장 거리 점 선택
        data_pts = np.array(endpoints) if len(endpoints) >= 2 else (np.array(pts) if len(pts) >= 2 else None)
        centers = []
        if data_pts is not None:
            max_dist = -1
            p1 = p2 = None
            for i in range(len(data_pts)):
                for j in range(i + 1, len(data_pts)):
                    d = (data_pts[i][0] - data_pts[j][0])**2 + (data_pts[i][1] - data_pts[j][1])**2
                    if d > max_dist:
                        max_dist = d
                        p1, p2 = tuple(data_pts[i]), tuple(data_pts[j])
            if p1 is not None and p2 is not None:
                centers = [p1, p2]

        # 6) 중간 과정 그리드 구성
        def to_bgr(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        viz = [
            cv2.resize(roi, (w, h)),
            to_bgr(bw),
            to_bgr(skel),
            to_bgr(gray),
            cv2.resize(hough_img, (w, h)),
            to_bgr(gray)
        ]
        ep_img = viz[3].copy()
        cen_img = viz[5].copy()
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

        # 7) 최종 엔드포인트 반환
        if centers:
            return ((x1 + centers[0][0], y1 + centers[0][1]),
                    (x1 + centers[1][0], y1 + centers[1][1]))
        return None

    def draw(self, image: np.ndarray, endpoints: tuple,
             color=(0, 255, 0), radius=5, thickness=-1):
        if endpoints:
            cv2.circle(image, endpoints[0], radius, color, thickness)
            cv2.circle(image, endpoints[1], radius, color, thickness)
