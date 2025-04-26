import cv2
import numpy as np

class LEDPinDetector:
    def __init__(self,
                 blur_ksize=(5,5),
                 thresh_type=cv2.THRESH_BINARY_INV,
                 thresh_val=200,
                 morph_kernel_size=3,
                 morph_iterations=1,
                 aspect_ratio_thresh=2.0,
                 min_area=50,
                 max_hole_distance=20):
        """
        blur_ksize: 가우시안 블러 커널 크기
        thresh_val: 이진화 임계값 (밝은 금속 리드가 흰색으로 나오도록 튜닝)
        morph_*: 형태학적 처리 파라미터
        aspect_ratio_thresh: (높이/너비) 최소 비율
        min_area: 컨투어 최소 면적
        max_hole_distance: 매핑 시 최대 허용 픽셀 거리
        """
        self.blur_ksize = blur_ksize
        self.thresh_type = thresh_type
        self.thresh_val = thresh_val
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        self.morph_iter = morph_iterations
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_area = min_area
        self.max_hole_distance = max_hole_distance

    def extract_endpoints(self, image: np.ndarray, bbox: tuple):
        """
        image: 전체 워프된 브레드보드 이미지
        bbox: (x1,y1,x2,y2) 형태의 LED bounding box
        returns: [(x_e1,y_e1), (x_e2,y_e2)] 혹은 빈 리스트
        """
        x1,y1,x2,y2 = bbox
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        _, th = cv2.threshold(blur, self.thresh_val, 255,
                              self.thresh_type)
        # 노이즈 제거
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              self.morph_kernel,
                              iterations=self.morph_iter)
        # 컨투어 검출
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        endpoints = []
        h_roi, w_roi = th.shape
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if h/w < self.aspect_ratio_thresh or cv2.contourArea(cnt) < self.min_area:
                continue
            # 다리의 아래쪽 극단점(꼽힌 부분) 취득
            pts = cnt.reshape(-1,2)
            bottom_pt = tuple(pts[pts[:,1].argmax()])
            endpoints.append((bottom_pt[0] + x1, bottom_pt[1] + y1))

        # 보통 2개의 리드가 검출되어야 함
        if len(endpoints) != 2:
            # 검출 개수 불일치 시 빈 리스트 반환
            return []
        return endpoints

    def map_to_holes(self,
                     endpoints: list,
                     holes: list):
        """
        endpoints: [(x,y),…] (LED 다리 endpoint)
        holes:    [(hx,hy),…] (브레드보드 구멍 중심 리스트)
        returns:  [hole_idx1, hole_idx2] 또는 빈 리스트
        """
        mapped = []
        for ex,ey in endpoints:
            # 각 홀까지 거리 계산
            dists = [((ex-hx)**2 + (ey-hy)**2, idx)
                     for idx, (hx,hy) in enumerate(holes)]
            d2, idx = min(dists)
            if np.sqrt(d2) <= self.max_hole_distance:
                mapped.append(idx)
            else:
                # 너무 멀면 매핑 실패
                mapped.append(None)
        return mapped

    def draw(self,
             image: np.ndarray,
             endpoints: list,
             holes: list=None,
             hole_color=(0,255,0),
             pin_color=(0,0,255),
             radius=5):
        """
        image 위에 검출 결과를 그림
        holes: 매핑된 hole 인덱스 리스트 (optional)
        """
        for ex,ey in endpoints:
            cv2.circle(image, (ex,ey), radius, pin_color, -1)
        if holes is not None:
            for idx in holes:
                if idx is None: continue
                hx,hy = holes[idx]
                cv2.circle(image, (hx,hy), radius, hole_color, 2)
