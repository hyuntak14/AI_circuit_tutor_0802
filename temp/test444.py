import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class LEDLeadDetector:
    def __init__(self):
        self.debug = False
    
    def set_debug(self, debug: bool):
        """디버그 모드 설정"""
        self.debug = debug
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred
    
    def detect_led_body(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """LED 몸체 검출 (색상 기반)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # LED 색상 범위 정의 (빨간색, 녹색, 파란색 등)
        color_ranges = [
            # 빨간색 LED
            [(0, 50, 50), (10, 255, 255)],
            [(170, 50, 50), (180, 255, 255)],
            # 녹색 LED
            [(40, 50, 50), (80, 255, 255)],
            # 파란색 LED
            [(100, 50, 50), (130, 255, 255)],
            # 노란색 LED
            [(20, 50, 50), (40, 255, 255)]
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 윤곽선을 LED로 가정
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # LED 크기 필터링
            if w > 10 and h > 10 and w < 100 and h < 100:
                return (x, y, w, h)
        
        return None
    
    def detect_metallic_leads(self, image: np.ndarray, led_region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """금속성 리드 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 관심 영역 설정 (LED 주변)
        if led_region:
            x, y, w, h = led_region
            # LED 주변 영역 확장
            margin = 50
            roi_x1 = max(0, x - margin)
            roi_y1 = max(0, y - margin)
            roi_x2 = min(image.shape[1], x + w + margin)
            roi_y2 = min(image.shape[0], y + h + margin)
            
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            offset = (roi_x1, roi_y1)
        else:
            roi = gray
            offset = (0, 0)
        
        # 금속성 리드는 보통 밝은 색상
        # 적응적 임계값 사용
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 형태학적 연산으로 선형 구조 강화
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # 수직 및 수평 선 검출
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_horizontal)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)
        
        # 결합
        lead_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # 원본 이미지 크기로 복원
        if led_region:
            full_mask = np.zeros(gray.shape, dtype=np.uint8)
            full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = lead_mask
            return full_mask
        
        return lead_mask
    
    def detect_lead_endpoints_hough(self, image: np.ndarray, lead_mask: np.ndarray) -> List[Tuple[int, int]]:
        """허프 변환을 사용한 리드 끝점 검출"""
        # 에지 검출
        edges = cv2.Canny(lead_mask, 50, 150)
        
        # 허프 변환으로 직선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        endpoints = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 직선의 길이 계산
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # 충분히 긴 직선만 고려
                if length > 15:
                    endpoints.extend([(x1, y1), (x2, y2)])
        
        return endpoints
    
    def detect_lead_endpoints_contour(self, image: np.ndarray, lead_mask: np.ndarray) -> List[Tuple[int, int]]:
        """윤곽선 기반 리드 끝점 검출"""
        contours, _ = cv2.findContours(lead_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        endpoints = []
        for contour in contours:
            # 윤곽선이 충분히 길어야 함
            if cv2.contourArea(contour) > 20:
                # 윤곽선의 극단점 찾기
                leftmost = tuple(contour[contour[:,:,0].argmin()][0])
                rightmost = tuple(contour[contour[:,:,0].argmax()][0])
                topmost = tuple(contour[contour[:,:,1].argmin()][0])
                bottommost = tuple(contour[contour[:,:,1].argmax()][0])
                
                endpoints.extend([leftmost, rightmost, topmost, bottommost])
        
        return endpoints
    
    def filter_endpoints(self, endpoints: List[Tuple[int, int]], led_region: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int]]:
        """끝점 필터링 및 정제"""
        if not endpoints:
            return []
        
        # 중복 제거 (유사한 위치의 점들 병합)
        filtered_endpoints = []
        for point in endpoints:
            is_duplicate = False
            for existing_point in filtered_endpoints:
                distance = np.sqrt((point[0] - existing_point[0])**2 + (point[1] - existing_point[1])**2)
                if distance < 10:  # 10픽셀 이내면 중복으로 간주
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_endpoints.append(point)
        
        # LED 영역과의 거리 기반 필터링
        if led_region and len(filtered_endpoints) > 2:
            x, y, w, h = led_region
            led_center = (x + w//2, y + h//2)
            
            # LED 중심과의 거리 계산
            distances = []
            for point in filtered_endpoints:
                dist = np.sqrt((point[0] - led_center[0])**2 + (point[1] - led_center[1])**2)
                distances.append((dist, point))
            
            # 거리 기준 정렬
            distances.sort(key=lambda x: x[0])
            
            # 가장 가까운 2개 점 선택
            if len(distances) >= 2:
                filtered_endpoints = [distances[0][1], distances[1][1]]
        
        return filtered_endpoints[:2]  # 최대 2개 점만 반환
    
    def detect_lead_endpoints(self, image_path: str) -> List[Tuple[int, int]]:
        """메인 검출 함수"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        # 전처리
        processed_image = self.preprocess_image(image)
        
        # LED 몸체 검출
        led_region = self.detect_led_body(processed_image)
        
        if self.debug:
            print(f"LED 영역: {led_region}")
        
        # 금속성 리드 검출
        lead_mask = self.detect_metallic_leads(processed_image, led_region)
        
        # 끝점 검출 (두 가지 방법 시도)
        endpoints_hough = self.detect_lead_endpoints_hough(processed_image, lead_mask)
        endpoints_contour = self.detect_lead_endpoints_contour(processed_image, lead_mask)
        
        # 결과 합치기
        all_endpoints = endpoints_hough + endpoints_contour
        
        # 필터링
        final_endpoints = self.filter_endpoints(all_endpoints, led_region)
        
        if self.debug:
            self._visualize_results(image, led_region, lead_mask, final_endpoints)
        
        return final_endpoints
    
    def _visualize_results(self, image: np.ndarray, led_region: Optional[Tuple[int, int, int, int]], 
                          lead_mask: np.ndarray, endpoints: List[Tuple[int, int]]):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('원본 이미지')
        axes[0, 0].axis('off')
        
        # LED 영역 표시
        image_with_led = image.copy()
        if led_region:
            x, y, w, h = led_region
            cv2.rectangle(image_with_led, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        axes[0, 1].imshow(cv2.cvtColor(image_with_led, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('LED 영역 검출')
        axes[0, 1].axis('off')
        
        # 리드 마스크
        axes[1, 0].imshow(lead_mask, cmap='gray')
        axes[1, 0].set_title('리드 마스크')
        axes[1, 0].axis('off')
        
        # 최종 결과
        result_image = image.copy()
        for i, (x, y) in enumerate(endpoints):
            cv2.circle(result_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(result_image, f'Lead {i+1}', (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if led_region:
            x, y, w, h = led_region
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        axes[1, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'최종 결과 (검출된 끝점: {len(endpoints)}개)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

# 고급 검출 방법들
class AdvancedLEDDetector(LEDLeadDetector):
    def __init__(self):
        super().__init__()
    
    def detect_by_template_matching(self, image: np.ndarray, template_path: str) -> List[Tuple[int, int]]:
        """템플릿 매칭을 사용한 리드 검출"""
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.8)
        
        endpoints = []
        for pt in zip(*locations[::-1]):
            endpoints.append(pt)
        
        return endpoints
    
    def detect_by_machine_learning(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """머신러닝 기반 검출 (YOLO 등 사용 가능)"""
        # 여기서는 개념적 구현
        # 실제로는 사전 훈련된 모델을 사용
        pass
    
    def detect_by_edge_analysis(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """에지 분석 기반 정밀 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 소벨 에지 검출
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 에지 강도 및 방향 계산
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # 직선적 에지만 필터링 (리드의 특성)
        linear_edges = np.where(
            (magnitude > 50) & 
            ((np.abs(direction) < 0.1) | (np.abs(direction - np.pi/2) < 0.1))
        )
        
        endpoints = []
        if len(linear_edges[0]) > 0:
            # 에지 점들을 클러스터링하여 끝점 찾기
            points = np.column_stack((linear_edges[1], linear_edges[0]))
            
            # 간단한 클러스터링으로 끝점 그룹 찾기
            from sklearn.cluster import DBSCAN
            if len(points) > 10:
                clustering = DBSCAN(eps=10, min_samples=5).fit(points)
                
                for cluster_id in set(clustering.labels_):
                    if cluster_id != -1:
                        cluster_points = points[clustering.labels_ == cluster_id]
                        # 클러스터의 중심점을 끝점으로 사용
                        center = np.mean(cluster_points, axis=0)
                        endpoints.append((int(center[0]), int(center[1])))
        
        return endpoints

# 사용 예제 및 성능 테스트
def main():
    # 기본 검출기
    detector = LEDLeadDetector()
    detector.set_debug(True)
    
    # 고급 검출기
    advanced_detector = AdvancedLEDDetector()
    
    try:
        # 이미지 경로 지정
        image_path = 'led4.jpg'  # 실제 이미지 경로로 변경
        
        # 기본 검출
        print("=== 기본 검출 방법 ===")
        endpoints = detector.detect_lead_endpoints(image_path)
        
        print(f"검출된 리드 끝점: {len(endpoints)}개")
        for i, (x, y) in enumerate(endpoints):
            print(f"끝점 {i+1}: ({x}, {y})")
        
        # 고급 검출 (에지 분석)
        print("\n=== 에지 분석 방법 ===")
        image = cv2.imread(image_path)
        if image is not None:
            edge_endpoints = advanced_detector.detect_by_edge_analysis(image)
            print(f"에지 분석으로 검출된 끝점: {len(edge_endpoints)}개")
            for i, (x, y) in enumerate(edge_endpoints):
                print(f"끝점 {i+1}: ({x}, {y})")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        print("샘플 데이터로 테스트하려면 실제 이미지 파일을 준비해주세요.")

# 정확도 평가 함수
def evaluate_detection_accuracy(detector, test_images, ground_truth):
    """검출 정확도 평가"""
    total_precision = 0
    total_recall = 0
    
    for img_path, true_points in zip(test_images, ground_truth):
        detected_points = detector.detect_lead_endpoints(img_path)
        
        # 정밀도와 재현율 계산
        precision, recall = calculate_precision_recall(detected_points, true_points)
        total_precision += precision
        total_recall += recall
    
    avg_precision = total_precision / len(test_images)
    avg_recall = total_recall / len(test_images)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    
    return avg_precision, avg_recall, f1_score

def calculate_precision_recall(detected, ground_truth, threshold=10):
    """정밀도와 재현율 계산"""
    if len(detected) == 0:
        return 0, 0
    
    true_positives = 0
    for det_point in detected:
        for true_point in ground_truth:
            distance = np.sqrt((det_point[0] - true_point[0])**2 + 
                             (det_point[1] - true_point[1])**2)
            if distance <= threshold:
                true_positives += 1
                break
    
    precision = true_positives / len(detected)
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    
    return precision, recall

if __name__ == "__main__":
    main()