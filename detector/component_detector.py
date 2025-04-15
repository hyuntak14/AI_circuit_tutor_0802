
import cv2
import numpy as np
# =============================================================================
# ComponentPinDetector 클래스: 전기 소자(및 전선) 핀 위치 검출
# (IC인 경우 8점, 그 외는 2점 반환)
# =============================================================================
class ComponentPinDetector:
    def __init__(self):
        pass

    def detect_component_pins(self, image, component_type, component_roi):
        """
        소자 타입과 ROI(바운딩 박스)를 입력받아 핀 위치를 검출하고 전체 이미지 좌표로 변환하여 반환합니다.
        """
        # ROI 추출: ROI 좌표계는 (0,0)부터 시작
        roi = image[component_roi[1]:component_roi[3], component_roi[0]:component_roi[2]]
        if component_type == 'Resistor':
            pins = self.extract_resistor_pins(roi)
        elif component_type == 'LED':
            pins = self.extract_led_pins(roi)
        elif component_type == 'Capacitor':
            pins = self.extract_capacitor_pins(roi)
        elif component_type == 'Diode':
            # 기존 컨투어 기반 방식 대신 저항과 동일한 방식으로 검출 (Canny, HoughLinesP)
            pins = self.extract_diode_pins(roi)
        elif component_type == 'IC':
            pins = self.extract_ic_pins_from_roi(roi)
        else:
            pins = []
        # ROI 기준 좌표를 전체 이미지 좌표로 변환
        global_pins = [(x + component_roi[0], y + component_roi[1]) for (x, y) in pins]
        return global_pins

    def extract_resistor_pins(self, roi):
        # 저항: Canny로 에지 검출 후 HoughLinesP를 통해 선분 검출, 좌우 극단점 선택 (2개)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
        if lines is None:
            return self.extract_two_pins_from_roi(roi)
        points = []
        h, w = roi.shape[:2]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 클리핑: ROI 범위 내로 좌표 보정
            x1 = min(max(x1, 0), w-1)
            y1 = min(max(y1, 0), h-1)
            x2 = min(max(x2, 0), w-1)
            y2 = min(max(y2, 0), h-1)
            points.append((x1, y1))
            points.append((x2, y2))
        points = np.array(points)
        left_idx = np.argmin(points[:, 0])
        right_idx = np.argmax(points[:, 0])
        return [tuple(points[left_idx]), tuple(points[right_idx])]

    def extract_diode_pins(self, roi):
        # Diode: 기존 방식 대신 저항과 동일한 방법으로 검출 (Canny -> HoughLinesP -> 좌우 극단점)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
        if lines is None:
            return self.extract_two_pins_from_roi(roi)
        points = []
        h, w = roi.shape[:2]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 = min(max(x1, 0), w-1)
            y1 = min(max(y1, 0), h-1)
            x2 = min(max(x2, 0), w-1)
            y2 = min(max(y2, 0), h-1)
            points.append((x1, y1))
            points.append((x2, y2))
        points = np.array(points)
        left_idx = np.argmin(points[:, 0])
        right_idx = np.argmax(points[:, 0])
        return [tuple(points[left_idx]), tuple(points[right_idx])]

    def extract_led_pins(self, roi):
        # LED: 기존 방식 (컨투어 중심 등)을 사용
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        if len(centroids) < 2:
            return self.extract_two_pins_from_roi(roi)
        centroids = np.array(centroids)
        left_idx = np.argmin(centroids[:, 0])
        right_idx = np.argmax(centroids[:, 0])
        return [tuple(centroids[left_idx]), tuple(centroids[right_idx])]

    def extract_capacitor_pins(self, roi):
        # Capacitor: 양쪽 중앙 (2개)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.extract_two_pins_from_roi(roi)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return [(x, y + h // 2), (x + w, y + h // 2)]

    def extract_ic_pins_from_roi(self, roi):
        """
        IC의 경우 DIP-8 형태를 가정.
        ROI의 상단과 하단 가장자리에서 각각 4개의 점(좌측 끝, 1/3, 2/3, 우측 끝)을 계산하여 총 8개 점 반환.
        """
        h, w = roi.shape[:2]
        top_y = 0
        bottom_y = h - 1
        xs_top = np.linspace(0, w, 5, endpoint=True)[1:-1]  # 4개의 점
        xs_bottom = np.linspace(0, w, 5, endpoint=True)[1:-1]
        top_points = [(int(x), top_y) for x in xs_top]
        bottom_points = [(int(x), bottom_y) for x in xs_bottom]
        return top_points + bottom_points

    def extract_two_pins_from_roi(self, roi_or_bbox):
        """
        roi_or_bbox가 ROI 이미지이면 ROI의 크기를 기준으로,
        또는 bounding box(tuple: (x,y,w,h))일 경우 해당 값을 이용하여 좌우 중앙 2점 생성.
        """
        if isinstance(roi_or_bbox, tuple) and len(roi_or_bbox) == 4:
            x, y, w, h = roi_or_bbox
        else:
            h, w = roi_or_bbox.shape[:2]
            x, y = 0, 0
        return [(x, y + h // 2), (x + w, y + h // 2)]
