import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Tuple, List, Optional

class LEDPreprocessingAnalyzer:
    """
    LED 이미지 전처리 과정을 분석하는 클래스
    """
    
    def __init__(self):
        """초기화"""
        pass
    
    def analyze_color_channels(self, image: np.ndarray) -> dict:
        """
        색상 채널별 분석
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            각 채널별 이미지 딕셔너리
        """
        # BGR 채널 분리
        b, g, r = cv2.split(image)
        
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # LAB 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b_lab = cv2.split(lab)
        
        return {
            'BGR_B': b,
            'BGR_G': g,
            'BGR_R': r,
            'HSV_H': h,
            'HSV_S': s,
            'HSV_V': v,
            'LAB_L': l,
            'LAB_A': a,
            'LAB_B': b_lab
        }
    
    def apply_threshold_methods(self, gray: np.ndarray) -> dict:
        """
        다양한 임계값 방법 적용
        
        Args:
            gray: 그레이스케일 이미지
            
        Returns:
            각 방법별 이진화 결과
        """
        results = {}
        
        # 1. Global Thresholding (다양한 값)
        for thresh_val in [50, 100, 150, 200]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            results[f'Global_{thresh_val}'] = binary
        
        # 2. Otsu's method
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results['Otsu'] = otsu
        
        # 3. Adaptive Thresholding
        adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        results['Adaptive_Mean'] = adaptive_mean
        
        adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
        results['Adaptive_Gaussian'] = adaptive_gaussian
        
        # 4. Inverse thresholding (핀이 어두운 경우)
        _, inv_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        results['Inverse_150'] = inv_thresh
        
        return results
    
    def apply_canny_variations(self, gray: np.ndarray) -> dict:
        """
        다양한 Canny 엣지 검출 매개변수 적용
        
        Args:
            gray: 그레이스케일 이미지
            
        Returns:
            각 매개변수별 엣지 검출 결과
        """
        results = {}
        
        # 블러 처리 (노이즈 감소)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 다양한 Canny 임계값 조합
        canny_params = [
            (30, 100),   # 낮은 임계값
            (50, 150),   # 중간 임계값
            (100, 200),  # 높은 임계값
            (20, 60),    # 매우 낮은 임계값
            (80, 180),   # 중간-높은 임계값
        ]
        
        for low, high in canny_params:
            edges = cv2.Canny(gray, low, high)
            results[f'Canny_{low}_{high}'] = edges
            
            # 블러 처리된 이미지에도 적용
            edges_blurred = cv2.Canny(blurred, low, high)
            results[f'Canny_Blur_{low}_{high}'] = edges_blurred
        
        return results
    
    def detect_green_led(self, image: np.ndarray) -> np.ndarray:
        """
        녹색 LED 본체 검출
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            녹색 LED 마스크
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 녹색 범위 정의 (HSV)
        # 녹색 LED는 보통 H: 40-80, S: 50-255, V: 50-255
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # 녹색 마스크 생성
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        return green_mask
    
    def enhance_metal_pins(self, image: np.ndarray, led_mask: np.ndarray) -> np.ndarray:
        """
        LED 마스크를 이용해 금속 핀 영역 강조
        
        Args:
            image: 입력 이미지
            led_mask: LED 본체 마스크
            
        Returns:
            핀 영역이 강조된 이미지
        """
        # LED 마스크 확장 (핀 영역 포함)
        kernel = np.ones((15, 15), np.uint8)
        expanded_mask = cv2.dilate(led_mask, kernel, iterations=2)
        
        # LED 본체 제외한 주변 영역 (핀이 있을 가능성이 높은 영역)
        pin_region_mask = cv2.subtract(expanded_mask, led_mask)
        
        # 원본 이미지에서 핀 영역만 추출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pin_region = cv2.bitwise_and(gray, gray, mask=pin_region_mask)
        
        return pin_region, pin_region_mask
    
    def visualize_preprocessing(self, image_path: str) -> None:
        """
        전처리 과정 시각화
        
        Args:
            image_path: 이미지 경로
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return
        
        print(f"\n분석 중: {os.path.basename(image_path)}")
        print(f"이미지 크기: {image.shape}")
        
        # RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 색상 채널 분석
        channels = self.analyze_color_channels(image)
        
        # 2. 임계값 방법들
        threshold_results = self.apply_threshold_methods(gray)
        
        # 3. Canny 엣지 검출
        canny_results = self.apply_canny_variations(gray)
        
        # 4. 녹색 LED 검출
        green_mask = self.detect_green_led(image)
        
        # 5. 핀 영역 강조
        pin_region, pin_mask = self.enhance_metal_pins(image, green_mask)
        
        # 시각화 1: 색상 채널
        fig1, axes1 = plt.subplots(3, 3, figsize=(15, 15))
        fig1.suptitle('색상 채널 분석', fontsize=16)
        
        for idx, (name, img) in enumerate(channels.items()):
            row = idx // 3
            col = idx % 3
            axes1[row, col].imshow(img, cmap='gray')
            axes1[row, col].set_title(name)
            axes1[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 시각화 2: 임계값 방법
        fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
        fig2.suptitle('임계값 방법 비교', fontsize=16)
        
        # 원본 이미지 표시
        axes2[0, 0].imshow(image_rgb)
        axes2[0, 0].set_title('원본')
        axes2[0, 0].axis('off')
        
        # 그레이스케일 표시
        axes2[0, 1].imshow(gray, cmap='gray')
        axes2[0, 1].set_title('그레이스케일')
        axes2[0, 1].axis('off')
        
        # 임계값 결과들
        for idx, (name, img) in enumerate(list(threshold_results.items())[:7]):
            row = (idx + 2) // 3
            col = (idx + 2) % 3
            axes2[row, col].imshow(img, cmap='gray')
            axes2[row, col].set_title(name)
            axes2[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 시각화 3: Canny 엣지 검출
        fig3, axes3 = plt.subplots(3, 4, figsize=(20, 15))
        fig3.suptitle('Canny Edge Detection 비교', fontsize=16)
        
        for idx, (name, img) in enumerate(list(canny_results.items())[:12]):
            row = idx // 4
            col = idx % 4
            axes3[row, col].imshow(img, cmap='gray')
            axes3[row, col].set_title(name)
            axes3[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 시각화 4: LED 및 핀 검출
        fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
        fig4.suptitle('LED 및 핀 영역 검출', fontsize=16)
        
        axes4[0, 0].imshow(image_rgb)
        axes4[0, 0].set_title('원본')
        axes4[0, 0].axis('off')
        
        axes4[0, 1].imshow(green_mask, cmap='gray')
        axes4[0, 1].set_title('녹색 LED 마스크')
        axes4[0, 1].axis('off')
        
        axes4[0, 2].imshow(pin_mask, cmap='gray')
        axes4[0, 2].set_title('핀 영역 마스크')
        axes4[0, 2].axis('off')
        
        axes4[1, 0].imshow(pin_region, cmap='gray')
        axes4[1, 0].set_title('핀 영역 추출')
        axes4[1, 0].axis('off')
        
        # 핀 영역에 대한 엣지 검출
        pin_edges = cv2.Canny(pin_region, 30, 100)
        axes4[1, 1].imshow(pin_edges, cmap='gray')
        axes4[1, 1].set_title('핀 영역 엣지')
        axes4[1, 1].axis('off')
        
        # 핀 영역에 대한 임계값
        _, pin_thresh = cv2.threshold(pin_region, 50, 255, cv2.THRESH_BINARY)
        axes4[1, 2].imshow(pin_thresh, cmap='gray')
        axes4[1, 2].set_title('핀 영역 임계값')
        axes4[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 결과 저장
        output_dir = os.path.splitext(image_path)[0] + '_preprocessing'
        os.makedirs(output_dir, exist_ok=True)
        
        # 가장 유망한 결과들 저장
        cv2.imwrite(os.path.join(output_dir, 'green_mask.png'), green_mask)
        cv2.imwrite(os.path.join(output_dir, 'pin_region.png'), pin_region)
        cv2.imwrite(os.path.join(output_dir, 'pin_edges.png'), pin_edges)
        cv2.imwrite(os.path.join(output_dir, 'otsu.png'), threshold_results['Otsu'])
        cv2.imwrite(os.path.join(output_dir, 'canny_30_100.png'), canny_results['Canny_30_100'])
        
        print(f"\n결과가 {output_dir}에 저장되었습니다.")
    
    def test_all_led_images(self, folder_path: str = ".") -> None:
        """
        폴더의 모든 LED 이미지에 대해 전처리 분석
        
        Args:
            folder_path: 검색할 폴더 경로
        """
        # LED 이미지 찾기
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        led_images = []
        
        for ext in extensions:
            pattern = os.path.join(folder_path, ext)
            files = glob.glob(pattern)
            led_files = [f for f in files if 'led' in os.path.basename(f).lower()]
            led_images.extend(led_files)
        
        if not led_images:
            print("LED 이미지를 찾을 수 없습니다.")
            return
        
        print(f"총 {len(led_images)}개의 LED 이미지를 찾았습니다.")
        
        for img_path in led_images:
            self.visualize_preprocessing(img_path)
            print("\n" + "="*60 + "\n")


# 사용 예제
if __name__ == "__main__":
    analyzer = LEDPreprocessingAnalyzer()
    
    # 현재 폴더의 모든 LED 이미지 분석
    print("🔍 LED 이미지 전처리 분석을 시작합니다...")
    analyzer.test_all_led_images(".")
    
    # 특정 이미지만 분석하려면:
    # analyzer.visualize_preprocessing("led16.jpg")