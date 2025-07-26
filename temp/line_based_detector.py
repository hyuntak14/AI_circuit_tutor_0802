import os
import cv2
import numpy as np
import math

class LineBasedResistorDetector:
    """
    요청된 로직 기반의 저항 끝점 검출기 (개선판)
    1. 저항 몸체(Contour) 검출 (Adaptive Threshold)
    2. PCA 기반 주요 축 추출으로 엔드포인트 계산
    3. HoughLinesP 기반 엔드포인트 계산 (Fallback)
    4. 디버그 정보 시각화 개선
    """
    def __init__(self, **kwargs):
        # 파라미터 기본값 설정
        self.params = {
            'adaptive_block_size': 11,
            'adaptive_C': 2,
            'min_area': 200,
            'max_area': 20000,
            'min_aspect_ratio': 1.5,
            'hough_threshold': 20,
            'min_line_length': 10,
            'max_line_gap': 10,
            'line_center_dist': 15,
            'line_angle_tolerance': 10
        }
        self.params.update(kwargs)
        self.debug_info = {}

    def _preprocess_and_find_body(self, gray_img):
        bw = cv2.adaptiveThreshold(
            gray_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.params['adaptive_block_size'],
            self.params['adaptive_C']
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.params['min_area'] <= area <= self.params['max_area']):
                continue
            if len(cnt) < 5:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            ar = max(w, h) / max(min(w, h), 1e-5)
            if ar < self.params['min_aspect_ratio']:
                continue
            valid.append(cnt)
        if not valid:
            return None, None, None
        body = max(valid, key=cv2.contourArea)
        M = cv2.moments(body)
        if M['m00'] == 0:
            return None, None, None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return body, (cx, cy), bw

    def _pca_endpoints(self, body_contour):
        pts = body_contour.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(pts, mean=np.empty((0)))
        direction = eigenvectors[0]
        diffs = pts - mean
        projections = diffs.dot(direction)
        min_i = np.argmin(projections)
        max_i = np.argmax(projections)
        p1 = tuple(pts[min_i].astype(int))
        p2 = tuple(pts[max_i].astype(int))
        self.debug_info['pca_endpoints'] = (p1, p2)
        return p1, p2

    def _find_best_line_pair(self, lines, center):
        if lines is None or len(lines) < 2:
            return None
        cx, cy = center
        valid = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            mid = ((x1 + x2) / 2, (y1 + y2) / 2)
            if math.hypot(mid[0] - cx, mid[1] - cy) < self.params['line_center_dist']:
                valid.append(ln[0])
        if len(valid) < 2:
            return None
        best, max_len = None, -1
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                l1, l2 = valid[i], valid[j]
                a1 = math.degrees(math.atan2(l1[3]-l1[1], l1[2]-l1[0]))
                a2 = math.degrees(math.atan2(l2[3]-l2[1], l2[2]-l2[0]))
                if abs(abs(a1) - abs(a2) - 180) > self.params['line_angle_tolerance']:
                    continue
                pts = np.array([[l1[0], l1[1]], [l1[2], l1[3]], [l2[0], l2[1]], [l2[2], l2[3]]])
                dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
                curr = dmat.max()
                if curr > max_len:
                    max_len = curr
                    idx = np.unravel_index(dmat.argmax(), dmat.shape)
                    best = (tuple(pts[idx[0]]), tuple(pts[idx[1]]))
        return best

    def extract(self, image: np.ndarray, bbox: tuple):
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        body, center, bw = self._preprocess_and_find_body(gray)
        if body is None:
            return None
        self.debug_info['body_contour'] = body
        self.debug_info['center'] = center
        edges = cv2.Canny(bw, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=self.params['hough_threshold'],
            minLineLength=self.params['min_line_length'],
            maxLineGap=self.params['max_line_gap']
        )
        endpoints = self._find_best_line_pair(lines, center)
        if endpoints is None:
            endpoints = self._pca_endpoints(body)
        if endpoints:
            (ex1, ey1), (ex2, ey2) = endpoints
            return ((x1 + ex1, y1 + ey1), (x1 + ex2, y1 + ey2))
        return None

    def draw(self, image: np.ndarray, endpoints: tuple, color=(0, 255, 0), radius=7, thickness=-1):
        if endpoints:
            cv2.circle(image, endpoints[0], radius, color, thickness)
            cv2.circle(image, endpoints[1], radius, color, thickness)
            cv2.line(image, endpoints[0], endpoints[1], (255, 0, 0), 2)
        if 'pca_endpoints' in self.debug_info:
            p1, p2 = self.debug_info['pca_endpoints']
            cv2.circle(image, p1, radius, (0, 255, 255), thickness)
            cv2.circle(image, p2, radius, (0, 255, 255), thickness)
        if 'body_contour' in self.debug_info:
            cv2.drawContours(image, [self.debug_info['body_contour']], -1, (0, 0, 255), 2)
        if 'center' in self.debug_info:
            cv2.circle(image, self.debug_info['center'], 5, (0, 0, 255), -1)


def setup_controls(window_name, detector):
    """
    OpenCV 트랙바와 on_trackbar_change를 통해 detector.params를 실시간 조정
    """
    def on_trackbar_change(name):
        def callback(v):
            if name == 'Adaptive Block Size':
                detector.params['adaptive_block_size'] = v | 1 if v >= 3 else 3
            elif name == 'Adaptive C':
                detector.params['adaptive_C'] = v - 10
            elif name == 'Min Area':
                detector.params['min_area'] = max(1, v)
            elif name == 'Max Area':
                detector.params['max_area'] = max(detector.params['min_area'], v)
            elif name == 'Min Aspect Ratio x10':
                detector.params['min_aspect_ratio'] = v / 10.0 if v > 0 else 0.1
            elif name == 'Hough Threshold':
                detector.params['hough_threshold'] = max(1, v)
            elif name == 'Min Line Length':
                detector.params['min_line_length'] = max(1, v)
            elif name == 'Max Line Gap':
                detector.params['max_line_gap'] = v
            elif name == 'Center Dist':
                detector.params['line_center_dist'] = v
            elif name == 'Angle Tol':
                detector.params['line_angle_tolerance'] = v
        return callback

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for name, maxv, init in [
        ('Adaptive Block Size', 51, detector.params['adaptive_block_size']),
        ('Adaptive C', 20, detector.params['adaptive_C'] + 10),
        ('Min Area', 50000, detector.params['min_area']),
        ('Max Area', 100000, detector.params['max_area']),
        ('Min Aspect Ratio x10', 100, int(detector.params['min_aspect_ratio']*10)),
        ('Hough Threshold', 100, detector.params['hough_threshold']),
        ('Min Line Length', 200, detector.params['min_line_length']),
        ('Max Line Gap', 100, detector.params['max_line_gap']),
        ('Center Dist', 100, detector.params['line_center_dist']),
        ('Angle Tol', 180, detector.params['line_angle_tolerance'])
    ]:
        cv2.createTrackbar(name, window_name, init, maxv, on_trackbar_change(name))


def main():
    """
    디렉토리 내 'resistor' 포함 이미지 파일들을 순회하며 저항 끝점 검출
    전체 이미지에서 바로 검출, ROI 선택 없음
    'n' 키로 다음 이미지, 'ESC'로 종료
    """
    detector = LineBasedResistorDetector()
    window_name = 'ResistorDetector'
    setup_controls(window_name, detector)

    files = [f for f in os.listdir('.') if 'resistor' in f.lower() and f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        print("처리할 'resistor' 이미지 파일이 없습니다.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for idx, fname in enumerate(files):
        img = cv2.imread(fname)
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = (0, 0, w, h)
        endpoints = detector.extract(img, bbox)
        detector.draw(img, endpoints)
        cv2.putText(img, f"{idx+1}/{len(files)}: {fname}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            continue
        elif key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
