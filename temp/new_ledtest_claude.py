import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from skimage.morphology import skeletonize
from typing import List, Tuple, Optional, Dict

class LEDEndpointDetector:
    """
    LED 끝점 검출 클래스 (구멍 내부 채우기 기능 포함)
    """
    
    def __init__(self,
                 clahe_clip: float = 2.0,
                 clahe_grid: Tuple[int, int] = (8, 8),
                 bg_kernel_size: Tuple[int, int] = (51, 51),
                 gamma: float = 1.2,
                 adapt_block_size: int = 51,
                 adapt_C: int = 5,
                 morph_kernel_size: int = 3,
                 morph_iterations: int = 1,
                 min_skel_area: int = 20,
                 hough_threshold: int = 50,
                 min_line_length_ratio: float = 0.5,
                 max_line_gap: int = 10,
                 hole_mask_radius: int = 5,
                 
                 # --- 내부 홀 채우기 관련 파라미터 ---
                 min_hole_area: int = 50   # 내부 홀로 간주할 최소 면적 (픽셀^2)
                 ):
        
        # 전처리 파라미터
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.bg_kernel_size = bg_kernel_size
        self.gamma = gamma
        self.adapt_block_size = adapt_block_size
        self.adapt_C = adapt_C
        
        # 모폴로지 파라미터
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self.morph_iterations = morph_iterations
        self.min_skel_area = min_skel_area
        
        # Hough Lines 파라미터
        self.hough_threshold = hough_threshold
        self.min_line_length_ratio = min_line_length_ratio
        self.max_line_gap = max_line_gap
        
        # 원형 구멍 마스킹
        self.hole_mask_radius = hole_mask_radius
        
        # 내부 홀 채우기용 면적 기준
        self.min_hole_area = min_hole_area  # 확실하지 않음: 실제 홀 크기에 맞춰 조정 필요
        
        # 디버그 이미지 저장소
        self.debug_images = {}
    
    def _remove_color_lines(self, roi: np.ndarray) -> np.ndarray:
        """색상선 제거 (인페인팅)"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        m3 = cv2.inRange(hsv, (100, 50, 50), (140, 255, 255))
        mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        result = cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)
        self.debug_images['color_mask'] = mask
        self.debug_images['color_removed'] = result
        return result
    
    def _preprocess_image(self, roi: np.ndarray, holes: List[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """이미지 전처리 + 내부 홀 채우기"""
        
        # 1. 색상선 제거
        clean_roi = self._remove_color_lines(roi)
        
        # 2. 그레이스케일 변환
        gray = cv2.cvtColor(clean_roi, cv2.COLOR_BGR2GRAY)
        self.debug_images['gray'] = gray
        
        # 3. CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        enhanced = clahe.apply(gray)
        self.debug_images['clahe'] = enhanced
        
        # 4. 배경 보정
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.bg_kernel_size)
        background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, bg_kernel)
        corrected = cv2.subtract(enhanced, background)
        self.debug_images['background'] = background
        self.debug_images['bg_corrected'] = corrected
        
        # 5. 감마 보정
        inv_gamma = 1.0 / self.gamma
        gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype="uint8")
        gamma_corrected = cv2.LUT(corrected, gamma_table)
        self.debug_images['gamma_corrected'] = gamma_corrected
        
        # 6. Adaptive Threshold
        binary = cv2.adaptiveThreshold(
            gamma_corrected, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adapt_block_size,
            self.adapt_C
        )
        self.debug_images['adaptive_thresh'] = binary
        
        # 7. 모폴로지 연산 (Open + Close)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_iterations)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iterations)
        self.debug_images['morphed_initial'] = morphed
        
        # 8. 원형 구멍 마스킹(기존)
        if holes:
            hole_masked = morphed.copy()
            for hx, hy in holes:
                cv2.circle(hole_masked, (hx, hy), self.hole_mask_radius, 0, -1)
            self.debug_images['hole_masked'] = hole_masked
            morphed = hole_masked
        
        # 9. 내부 홀(빈 영역) 찾아서 채우기
        # 9-1) 이진화된 이미지를 반전: 내부 배경(구멍) 픽셀을 흰색으로 만듦
        inv = cv2.bitwise_not(morphed)
        # 9-2) 연결 요소 분석
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        filled = morphed.copy()
        h, w = morphed.shape
        # 9-3) 각 라벨 검사
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            lbl_w = stats[lbl, cv2.CC_STAT_WIDTH]
            lbl_h = stats[lbl, cv2.CC_STAT_HEIGHT]
            # 1) 면적이 기준 이하인지 확인
            if area < self.min_hole_area:
                # 2) 외곽에 닿아 있지 않은 영역인지 확인 (테두리에 닿으면 LED 선일 수 있음)
                if not (x == 0 or y == 0 or x + lbl_w == w or y + lbl_h == h):
                    # 내부 홀로 간주 → 내부를 채워서 제거
                    filled[labels == lbl] = 255   # 흰색(=LED 영역)으로 채움
        self.debug_images['hole_filled'] = filled
        
        return gray, gamma_corrected, filled
    
    def _extract_skeleton_endpoints(self, binary_img: np.ndarray) -> List[Tuple[int, int]]:
        """스켈레톤 기반 끝점 추출 (기존 코드와 동일)"""
        skeleton = skeletonize(binary_img // 255).astype(np.uint8) * 255
        self.debug_images['skeleton_raw'] = skeleton
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        clean_skeleton = np.zeros_like(skeleton)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_skel_area:
                clean_skeleton[labels == i] = 255
        self.debug_images['skeleton_clean'] = clean_skeleton
        
        endpoints = []
        h, w = clean_skeleton.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if clean_skeleton[y, x] > 0:
                    neighbors = np.sum(clean_skeleton[y-1:y+2, x-1:x+2] > 0) - 1
                    if neighbors == 1:
                        endpoints.append((x, y))
        endpoint_vis = cv2.cvtColor(clean_skeleton, cv2.COLOR_GRAY2BGR)
        for ex, ey in endpoints:
            cv2.circle(endpoint_vis, (ex, ey), 3, (0, 0, 255), -1)
        self.debug_images['skeleton_endpoints'] = endpoint_vis
        
        return endpoints
    
    def _extract_hough_endpoints(self, gray_img: np.ndarray) -> List[Tuple[int, int]]:
        """Hough 기반 끝점 추출 (기존 코드와 동일)"""
        v = np.median(gray_img)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray_img, lower, upper)
        self.debug_images['canny_edges'] = edges
        
        h, w = gray_img.shape
        min_line_length = int(w * self.min_line_length_ratio)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                 minLineLength=min_line_length,
                                 maxLineGap=self.max_line_gap)
        
        hough_vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        endpoints = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                endpoints.extend([(x1, y1), (x2, y2)])
        self.debug_images['hough_lines'] = hough_vis
        
        return endpoints
    
    def _select_best_endpoints(self, skeleton_endpoints: List[Tuple[int, int]], 
                              hough_endpoints: List[Tuple[int, int]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """끝점 쌍 선택 (기존 코드와 동일)"""
        candidates = skeleton_endpoints if len(skeleton_endpoints) >= 2 else hough_endpoints
        if len(candidates) < 2:
            return None
        max_dist2 = -1
        best_pair = None
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                p1, p2 = candidates[i], candidates[j]
                d2 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                if d2 > max_dist2:
                    max_dist2 = d2
                    best_pair = (p1, p2)
        return best_pair
    
    def detect_endpoints(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                         holes: List[Tuple[int, int]] = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """메인 함수 (기존 코드와 동일 흐름)"""
        self.debug_images.clear()
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        self.debug_images['original_roi'] = roi
        
        relative_holes = None
        if holes:
            relative_holes = [(hx - x1, hy - y1) for hx, hy in holes 
                              if x1 <= hx < x2 and y1 <= hy < y2]
        
        gray, gamma_corr, binary = self._preprocess_image(roi, relative_holes)
        
        sk_end = self._extract_skeleton_endpoints(binary)
        hough_end = self._extract_hough_endpoints(gamma_corr)
        
        best = self._select_best_endpoints(sk_end, hough_end)
        if best is None:
            return None
        
        p1, p2 = best
        abs_p1 = (p1[0] + x1, p1[1] + y1)
        abs_p2 = (p2[0] + x1, p2[1] + y1)
        
        final_vis = roi.copy()
        cv2.circle(final_vis, p1, 8, (0, 255, 0), -1)
        cv2.circle(final_vis, p2, 8, (0, 255, 0), -1)
        cv2.line(final_vis, p1, p2, (255, 0, 0), 2)
        self.debug_images['final_result'] = final_vis
        
        return abs_p1, abs_p2
    
    def get_debug_images(self) -> Dict[str, np.ndarray]:
        """디버그 이미지 반환 (기존 코드)"""
        return self.debug_images.copy()



class LEDEndpointTester:
    """
    LED 끝점 검출 테스트 및 시각화 클래스
    """
    
    def __init__(self, detector: LEDEndpointDetector):
        self.detector = detector
    
    def test_detection(self, image_path: str, bbox: Tuple[int, int, int, int], 
                      holes: List[Tuple[int, int]] = None, 
                      save_path: str = None, show_plots: bool = True):
        """
        LED 끝점 검출 테스트 및 시각화
        
        Args:
            image_path: 테스트 이미지 경로
            bbox: LED 영역 (x1, y1, x2, y2)
            holes: 구멍 위치 리스트
            save_path: 결과 저장 경로
            show_plots: 플롯 표시 여부
        """
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 끝점 검출 실행
        print("LED 끝점 검출 시작...")
        endpoints = self.detector.detect_endpoints(image, bbox, holes)
        
        # 디버그 이미지 가져오기
        debug_images = self.detector.get_debug_images()
        
        # 결과 출력
        if endpoints:
            print(f"✅ 끝점 검출 성공: {endpoints[0]} - {endpoints[1]}")
            distance = np.sqrt((endpoints[0][0] - endpoints[1][0])**2 + 
                             (endpoints[0][1] - endpoints[1][1])**2)
            print(f"   끝점 간 거리: {distance:.2f} pixels")
        else:
            print("❌ 끝점 검출 실패")
        
        # 시각화
        if show_plots:
            self._visualize_process(image_rgb, bbox, debug_images, endpoints, holes)
        
        # 결과 저장
        '''if save_path:
            self._save_results(image_rgb, bbox, debug_images, endpoints, save_path)'''
        
        return endpoints
    
    def _visualize_process(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                          debug_images: Dict[str, np.ndarray], 
                          endpoints: Optional[Tuple], holes: List[Tuple[int, int]] = None):
        """전 과정 시각화"""
        
        # 서브플롯 설정
        fig = plt.figure(figsize=(20, 24))
        
        # 1. 원본 이미지와 ROI
        plt.subplot(6, 4, 1)
        plt.imshow(image)
        if bbox:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
        if holes:
            hx, hy = zip(*holes)
            plt.scatter(hx, hy, c='blue', s=10, marker='o')
        plt.title('1. Original Image + ROI + Holes')
        plt.axis('off')
        
        # 2. ROI
        if 'original_roi' in debug_images:
            plt.subplot(6, 4, 2)
            roi_rgb = cv2.cvtColor(debug_images['original_roi'], cv2.COLOR_BGR2RGB)
            plt.imshow(roi_rgb)
            plt.title('2. Original ROI')
            plt.axis('off')
        
        # 3. 색상선 마스크
        if 'color_mask' in debug_images:
            plt.subplot(6, 4, 3)
            plt.imshow(debug_images['color_mask'], cmap='gray')
            plt.title('3. Color Line Mask')
            plt.axis('off')
        
        # 4. 색상선 제거 결과
        if 'color_removed' in debug_images:
            plt.subplot(6, 4, 4)
            color_removed_rgb = cv2.cvtColor(debug_images['color_removed'], cv2.COLOR_BGR2RGB)
            plt.imshow(color_removed_rgb)
            plt.title('4. Color Lines Removed')
            plt.axis('off')
        
        # 5. 그레이스케일
        if 'gray' in debug_images:
            plt.subplot(6, 4, 5)
            plt.imshow(debug_images['gray'], cmap='gray')
            plt.title('5. Grayscale')
            plt.axis('off')
        
        # 6. CLAHE
        if 'clahe' in debug_images:
            plt.subplot(6, 4, 6)
            plt.imshow(debug_images['clahe'], cmap='gray')
            plt.title('6. CLAHE Enhanced')
            plt.axis('off')
        
        # 7. 배경
        if 'background' in debug_images:
            plt.subplot(6, 4, 7)
            plt.imshow(debug_images['background'], cmap='gray')
            plt.title('7. Background')
            plt.axis('off')
        
        # 8. 배경 보정
        if 'bg_corrected' in debug_images:
            plt.subplot(6, 4, 8)
            plt.imshow(debug_images['bg_corrected'], cmap='gray')
            plt.title('8. Background Corrected')
            plt.axis('off')
        
        # 9. 감마 보정
        if 'gamma_corrected' in debug_images:
            plt.subplot(6, 4, 9)
            plt.imshow(debug_images['gamma_corrected'], cmap='gray')
            plt.title('9. Gamma Corrected')
            plt.axis('off')
        
        # 10. Adaptive Threshold
        if 'adaptive_thresh' in debug_images:
            plt.subplot(6, 4, 10)
            plt.imshow(debug_images['adaptive_thresh'], cmap='gray')
            plt.title('10. Adaptive Threshold')
            plt.axis('off')
        
        # 11. 모폴로지
        if 'morphed' in debug_images:
            plt.subplot(6, 4, 11)
            plt.imshow(debug_images['morphed'], cmap='gray')
            plt.title('11. Morphology')
            plt.axis('off')
        
        # 12. 구멍 마스킹
        if 'hole_masked' in debug_images:
            plt.subplot(6, 4, 12)
            plt.imshow(debug_images['hole_masked'], cmap='gray')
            plt.title('12. Hole Masked')
            plt.axis('off')
        
        # 13. 원본 스켈레톤
        if 'skeleton_raw' in debug_images:
            plt.subplot(6, 4, 13)
            plt.imshow(debug_images['skeleton_raw'], cmap='gray')
            plt.title('13. Raw Skeleton')
            plt.axis('off')
        
        # 14. 정제된 스켈레톤
        if 'skeleton_clean' in debug_images:
            plt.subplot(6, 4, 14)
            plt.imshow(debug_images['skeleton_clean'], cmap='gray')
            plt.title('14. Clean Skeleton')
            plt.axis('off')
        
        # 15. 스켈레톤 끝점
        if 'skeleton_endpoints' in debug_images:
            plt.subplot(6, 4, 15)
            endpoint_img_rgb = cv2.cvtColor(debug_images['skeleton_endpoints'], cv2.COLOR_BGR2RGB)
            plt.imshow(endpoint_img_rgb)
            plt.title('15. Skeleton Endpoints')
            plt.axis('off')
        
        # 16. Canny 엣지
        if 'canny_edges' in debug_images:
            plt.subplot(6, 4, 16)
            plt.imshow(debug_images['canny_edges'], cmap='gray')
            plt.title('16. Canny Edges')
            plt.axis('off')
        
        # 17. Hough Lines
        if 'hough_lines' in debug_images:
            plt.subplot(6, 4, 17)
            hough_img_rgb = cv2.cvtColor(debug_images['hough_lines'], cv2.COLOR_BGR2RGB)
            plt.imshow(hough_img_rgb)
            plt.title('17. Hough Lines')
            plt.axis('off')
        
        # 18. 최종 결과
        if 'final_result' in debug_images:
            plt.subplot(6, 4, 18)
            final_img_rgb = cv2.cvtColor(debug_images['final_result'], cv2.COLOR_BGR2RGB)
            plt.imshow(final_img_rgb)
            plt.title('18. Final Result')
            plt.axis('off')
        
        # 19. 전체 이미지에 결과 표시
        plt.subplot(6, 4, 19)
        result_img = image.copy()
        if endpoints:
            cv2.circle(result_img, endpoints[0], 10, (0, 255, 0), -1)
            cv2.circle(result_img, endpoints[1], 10, (0, 255, 0), -1)
            cv2.line(result_img, endpoints[0], endpoints[1], (255, 0, 0), 3)
        plt.imshow(result_img)
        plt.title('19. Final Detection on Full Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _save_results(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                     debug_images: Dict[str, np.ndarray], 
                     endpoints: Optional[Tuple], save_path: str):
        """결과 저장"""
        # 전체 이미지에 결과 표시
        result_img = image.copy()
        if endpoints:
            cv2.circle(result_img, endpoints[0], 10, (0, 255, 0), -1)
            cv2.circle(result_img, endpoints[1], 10, (0, 255, 0), -1)
            cv2.line(result_img, endpoints[0], endpoints[1], (255, 0, 0), 3)
        
        # BGR로 변환하여 저장
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
        print(f"결과 저장됨: {save_path}")

import os
import cv2


if __name__ == "__main__":
    # 1) 검출기 생성: min_hole_area는 실제 이미지의 구멍 크기에 맞춰 조정하세요.
    detector = LEDEndpointDetector(
        clahe_clip=2.0,
        gamma=1.2,
        adapt_block_size=51,
        adapt_C=5,
        min_skel_area=20,
        hough_threshold=50,
        min_hole_area=1  # 내부 홀로 간주할 최소 면적 (픽셀^2). 필요 시 조정
    )
    
    # 2) 테스터 생성
    tester = LEDEndpointTester(detector)

    # 3) 현재 폴더 내 'led'가 포함된 이미지 파일 목록 가져오기
    current_folder = "."
    files = os.listdir(current_folder)
    led_files = [
        f for f in files
        if "led" in f.lower() and f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # 4) 이미지별로 끝점 검출 수행
    for image_file in led_files:
        image_path = os.path.join(current_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지 로드 실패: {image_file}")
            continue

        height, width = image.shape[:2]
        bbox = (0, 0, width, height)

        # holes 파라미터를 사용하지 않는 경우 None으로 전달
        endpoints = tester.test_detection(
            image_path=image_path,
            bbox=bbox,
            holes=None,       # 원형 구멍 위치가 있을 때는 리스트로 전달할 수 있습니다.
            save_path=None,   # 결과 저장이 필요하면 경로를 지정하세요.
            show_plots=True   # 디버그 이미지 표시 여부
        )

        if endpoints:
            print(f"{image_file} → 끝점: {endpoints[0]} , {endpoints[1]}")
        else:
            print(f"{image_file} → 끝점 검출 실패")

    print("LED 끝점 검출 완료!")
