import cv2
import numpy as np
from skimage import morphology
from scipy import ndimage
from typing import Tuple, List, Optional
from skimage import morphology, measure
class ImprovedLEDEndpointDetector:
    """
    RGB 이미지(단일 LED 크롭)에 대해
    1) 스켈레톤 기반 끝점 검출을 우선 시도하고,
    2) 실패 시 HoughLinesP 기반 검출로 폴백하는 클래스
    """
    def __init__(self,
                 min_contour_area: int = 50,
                 blur_kernel_size: int = 3,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 min_object_size: int = 30,
                 dynamic_percentile: Optional[int] = None,
                 hough_min_length: int = 20,
                 debug_mode: bool = False,
                  binarization_method: str = "otsu"):
        """
        Args:
            min_contour_area: 최소 윤곽선 면적
            blur_kernel_size: 가우시안 블러 커널 크기 (홀수)
            canny_low, canny_high: Canny 엣지 임계값
            min_object_size: 스켈레톤 전 작은 오브젝트 제거 최소 픽셀 수
            dynamic_percentile: 퍼센타일 기반 이진화(p값), None이면 사용 안 함
            hough_min_length: HoughLinesP 에서 최소 선분 길이
            debug_mode: 디버그 메시지 출력 여부
        """
        self.min_contour_area = min_contour_area
        self.blur_kernel_size = blur_kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_object_size = min_object_size
        self.dynamic_percentile = dynamic_percentile
        self.hough_min_length = hough_min_length
        self.debug_mode = debug_mode
        self.binarization_method = binarization_method

    def detect_metal_pins(self, image: np.ndarray) -> np.ndarray:
        # image는 RGB
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        metal_hsv = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 60, 255]))
        metal_lab = cv2.inRange(lab, np.array([100,120,120]), np.array([255,136,136]))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(metal_hsv, metal_lab)
        mask = cv2.bitwise_or(mask, bright)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        return mask


    def preprocess_image_advanced(self, image: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            gray: 그레이스케일
            bin_raw: 이진화 직후
            bin_cleaned: 구멍 제거 후
            edges: Canny 엣지
        """
        # 1) Gray
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim==3 else image.copy()

        # 2) CLAHE + Blur
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (self.blur_kernel_size,)*2, 0)

        # 3) Otsu 이진화
        _, bin_otsu = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_raw = bin_otsu.copy()

        # (optional) percentile 이진화
        if self.binarization_method == "otsu":
            _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.binarization_method == "adaptive":
            bin_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        elif self.binarization_method == "sauvola":
            from skimage.filters import threshold_sauvola
            window_size = 25
            thr_sauvola = threshold_sauvola(blurred, window_size=window_size)
            bin_img = (blurred > thr_sauvola).astype(np.uint8) * 255
        else:
            # 기본은 Otsu
            _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4) 금속 마스크 결합 (필요시)
        metal = self.detect_metal_pins(image) if image.ndim==3 else None
        if metal is not None:
            bin_raw = cv2.bitwise_or(bin_raw, metal)

        # 5) 모폴로지 연결
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        bin_raw = cv2.morphologyEx(bin_raw, cv2.MORPH_CLOSE, k, iterations=2)

        # --- shape-based hole removal ---
        # 검은 구멍 검출: 반전해서 connected components
        inv = (bin_raw == 0).astype(np.uint8)
        lbl = measure.label(inv, connectivity=2)
        props = measure.regionprops(lbl)
        bin_clean = bin_raw.copy()

        for reg in props:
            area = reg.area
            # 면적 너무 크거나 너무 작으면 패스
            if area < 50 or area > 2000:
                continue

            # 원형도(circularity) 계산: 4πA / P²
            p = reg.perimeter if reg.perimeter>0 else 1
            circ = 4 * np.pi * area / (p*p)
            # 직사각형 정도(rectangularity): A / (bbox_area)
            minr, minc, maxr, maxc = reg.bbox
            bbox_area = (maxr-minr)*(maxc-minc)
            rect = area / bbox_area if bbox_area>0 else 0

            # 원형 또는 직사각형 형태라고 판단되면 hole fill
            if circ > 0.5 or rect > 0.5:
                # 해당 레이블 픽셀을 흰색으로 채움
                bin_clean[lbl == reg.label] = 255

        # 6) 최종 closing
        bin_clean = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, k, iterations=1)

        # 7) 엣지
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        return gray, bin_raw, bin_clean, edges



    def get_endpoints_using_skeleton(self, binary: np.ndarray
                                   ) -> Optional[Tuple[Tuple[int,int],Tuple[int,int]]]:
        bw = binary > 0
        cleaned = morphology.remove_small_objects(bw, min_size=self.min_object_size)
        skeleton = morphology.skeletonize(cleaned)
        # 커널 필터로 끝점 후보
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]],dtype=np.uint8)
        filt = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        pts = np.column_stack(np.where(filt == 11))
        if len(pts) < 2:
            return None
        # 가장 먼 두 점 찾기
        maxd = 0; pair = None
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i]-pts[j])
                if d>maxd:
                    maxd, pair = d, (pts[i],pts[j])
        if pair:
            (r1,c1),(r2,c2) = pair
            return (c1,r1),(c2,r2)
        return None

    def find_led_pins_using_lines(self, edges: np.ndarray, shape: Tuple[int,int]
                                ) -> Optional[List[Tuple[int,int]]]:
        # HoughLinesP with minLineLength=self.hough_min_length
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=30,
                                minLineLength=self.hough_min_length,
                                maxLineGap=10)
        if lines is None:
            return None
        pts = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            # 길이 필터링은 Hough 함수에서 이미 수행됨
            pts += [(x1,y1),(x2,y2)]
        if len(pts)<2:
            return None
        # 가장 먼 두 점
        maxd=0; best=None
        for i in range(len(pts)):
            for j in range(i+1,len(pts)):
                d = np.linalg.norm(np.array(pts[i])-np.array(pts[j]))
                if d>maxd:
                    maxd, best = d, (pts[i], pts[j])
        return list(best) if best else None

    def validate_endpoints(self, endpoints, shape: Tuple[int,int]) -> bool:
        (x1,y1),(x2,y2) = endpoints
        h,w = shape[:2]
        if not all(0<=x<w and 0<=y<h for x,y in endpoints):
            return False
        if np.linalg.norm(np.array(endpoints[0]) - np.array(endpoints[1])) < 10:
            return False
        return True

    def detect_endpoints(self, image: np.ndarray) -> Optional[Tuple[Tuple[int,int],Tuple[int,int]]]:
        """
        최종적으로:
        1) 스켈레톤 기반 검출을 먼저 시도하고,
        2) 실패 시 HoughLinesP 기반 검출 결과를 반환합니다.
        """
        gray, bin_raw, bin_clean, edges = self.preprocess_image_advanced(image)

        # 1) Skeleton: bin_clean을 사용!
        se = self.get_endpoints_using_skeleton(bin_clean)
        if se and self.validate_endpoints(se, image.shape):
            if self.debug_mode: print("Skeleton 검출 성공")
            return se

        # 2) Hough: edges 그대로 사용
        he = self.find_led_pins_using_lines(edges, image.shape)
        if he and self.validate_endpoints(tuple(he), image.shape):
            if self.debug_mode: print("Hough 검출 성공")
            return tuple(he)

        return None
