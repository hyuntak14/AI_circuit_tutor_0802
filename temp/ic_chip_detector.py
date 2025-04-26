import cv2
import numpy as np

class ICChipPinDetector:
    """
    IC 칩(검은색 직사각형) 검출 및 핀(8개) 위치 추출 클래스
    """
    def __init__(self,
                 gray_thresh: int = 60,
                 min_area: int = 1000,
                 visualize: bool = False):
        # 그레이스케일 임계값 및 최소 면적 설정
        self.gray_thresh = gray_thresh
        self.min_area = min_area
        self.visualize = visualize

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        전처리: 그레이스케일 -> 이진화(검은 영역 강조) -> Morphology
        필요시 resistor_detector의 CLAHE, 감마보정, Adaptive Threshold 적용 가능
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 검은색 영역 강조 (임계값 이하 픽셀을 흰색으로)
        _, bw = cv2.threshold(gray, self.gray_thresh, 255, cv2.THRESH_BINARY_INV)
        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        return bw

    def detect(self, image: np.ndarray):
        """
        전체 이미지에서 IC 칩 검출 및 핀 위치 8개 반환
        Returns:
          List of dict: {'box': np.ndarray(4,2), 'pin_points': List[(x,y)*8]}
        """
        bw = self._preprocess(image)
        # 외곽선 검출
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            # 회전 최소 외접 사각형
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(int)
            # 각 변의 길이 계산
            edges = []
            for i in range(4):
                p0 = box[i]
                p1 = box[(i+1) % 4]
                length = np.linalg.norm(p1 - p0)
                edges.append((i, length))
            # 가장 긴 두 변 선택
            edges_sorted = sorted(edges, key=lambda x: x[1], reverse=True)
            long_idxs = [edges_sorted[0][0], edges_sorted[1][0]]
            pin_points = []
            # 각 긴 변에서 4등분 지점(1/5,2/5,3/5,4/5)에 핀 위치 샘플링
            for idx in long_idxs:
                p0 = box[idx]
                p1 = box[(idx + 1) % 4]
                for i in range(1, 5):
                    pt = p0 + (p1 - p0) * (i / 5.0)
                    pin_points.append(tuple(pt.astype(int)))
            detections.append({'box': box, 'pin_points': pin_points})

            if self.visualize:
                # 디버그용 시각화
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                for x, y in pin_points:
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        return detections

    def draw(self, image: np.ndarray, detections):
        """
        검출 결과 시각화
        """
        for det in detections:
            box = det['box']
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            for pt in det['pin_points']:
                cv2.circle(image, pt, 6, (0, 255, 0), -1)
