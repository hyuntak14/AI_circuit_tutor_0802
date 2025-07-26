import cv2
import numpy as np
from skimage.morphology import skeletonize


class ResistorEndpointDetector:
    """
    개선된 저항 엔드포인트 검출기
    - 객체 형태 필터링으로 저항 형태에 맞는 스켈레톤만 유지
    - PCA 기반 주축 분석으로 저항의 방향성 고려한 끝점 검출
    """
    def __init__(self,
                 hough_threshold=50,
                 min_line_length_ratio=0.5,
                 max_line_gap=10,
                 morph_kernel_size=1,
                 morph_iterations=1,
                 min_skel_area=10,
                 # 새로운 파라미터들
                 min_aspect_ratio=1.5,     # 저항 최소 가로세로 비율
                 max_aspect_ratio=30.0,    # 저항 최대 가로세로 비율
                 min_resistor_area=100,    # 저항 최소 면적
                 max_resistor_area=5000,   # 저항 최대 면적
                 visualize=False,
                 use_opening=True,         # 열림 연산 사용 여부
                 use_closing=True,         # 닫힘 연산 사용 여부
                 threshold_method='otsu',
                 fixed_threshold=120):    # 'otsu', 'fixed', 'adaptive'
        self.hough_threshold = hough_threshold
        self.min_line_length_ratio = min_line_length_ratio
        self.max_line_gap = max_line_gap
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        self.morph_iterations = morph_iterations
        self.min_skel_area = min_skel_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_resistor_area = min_resistor_area
        self.max_resistor_area = max_resistor_area
        self.visualize = visualize
        self.use_opening = use_opening
        self.use_closing = use_closing
        self.threshold_method = threshold_method
        self.fixed_threshold = fixed_threshold

    def filter_resistor_skeleton(self, skel):
        """
        저항 형태에 맞는 스켈레톤 객체만 필터링
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel, connectivity=8)
        clean = np.zeros_like(skel)
        
        valid_components = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 기본 면적 필터
            if area < self.min_skel_area:
                continue
                
            # 저항 형태 필터링
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            
            # 저항 같은 직사각형 형태인지 확인
            if (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and 
                self.min_resistor_area <= area <= self.max_resistor_area):
                clean[labels == i] = 255
                valid_components.append({
                    'label': i,
                    'area': area,
                    'centroid': centroids[i],
                    'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                })
        
        return clean, valid_components

    def find_principal_axis_endpoints(self, binary_img, component_info=None):
        """
        PCA를 사용하여 주축 방향을 찾고 양 끝점 검출
        """
        # 흰색 픽셀들의 좌표 추출
        points = np.column_stack(np.where(binary_img > 0))
        if len(points) < 10:  # 점이 너무 적으면 실패
            return []
            
        # PCA 수행
        points = points.astype(np.float32)
        mean = np.mean(points, axis=0)
        
        # 공분산 행렬 계산
        cov_matrix = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 가장 큰 고유값에 해당하는 고유벡터 (주축 방향)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        
        # 주축 방향으로 점들을 투영
        centered_points = points - mean
        projections = np.dot(centered_points, principal_axis)
        
        # 투영된 값의 최대/최소 인덱스 찾기
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        # 실제 좌표로 변환 (y, x -> x, y)
        endpoint1 = (int(points[min_idx][1]), int(points[min_idx][0]))
        endpoint2 = (int(points[max_idx][1]), int(points[max_idx][0]))
        
        return [endpoint1, endpoint2]

    def extract(self, image: np.ndarray, bbox: tuple):
            x1, y1, x2, y2 = bbox
            roi = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 1) 이진화 + 선택적 노이즈 제거
            if self.threshold_method == 'otsu':
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif self.threshold_method == 'fixed':
                _, bw = cv2.threshold(gray, self.fixed_threshold, 255, cv2.THRESH_BINARY_INV)
            elif self.threshold_method == 'adaptive':
                bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
            else:
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 선택적 모폴로지 연산
            if self.use_opening:
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_iterations)
            if self.use_closing:
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iterations)

            # 2) 스켈레톤화 및 저항 형태 필터링
            skel = skeletonize(bw // 255).astype(np.uint8) * 255
            filtered_skel, valid_components = self.filter_resistor_skeleton(skel)
            
            # 3) 주축 기반 엔드포인트 검출
            endpoints = []
            if len(valid_components) > 0:
                # 가장 큰 유효한 컴포넌트 선택
                largest_component = max(valid_components, key=lambda x: x['area'])
                
                # 해당 컴포넌트만 추출
                component_mask = np.zeros_like(filtered_skel)
                num_labels, labels = cv2.connectedComponents(filtered_skel, connectivity=8)
                for i in range(1, num_labels):
                    if np.array_equal(np.mean(np.column_stack(np.where(labels == i)), axis=0), 
                                    [largest_component['centroid'][1], largest_component['centroid'][0]]):
                        component_mask[labels == i] = 255
                        break
                
                # PCA 기반 끝점 찾기
                #pca_endpoints = self.find_principal_axis_endpoints(component_mask)
                pca_endpoints = component_mask
                if len(pca_endpoints) == 2:
                    endpoints = pca_endpoints
            
            # 4) PCA 방법이 실패하면 기존 방법 사용 (백업)
            if not endpoints:
                # 기존 엔드포인트 검출 방법
                h, w = filtered_skel.shape
                for yy in range(1, h-1):
                    for xx in range(1, w-1):
                        if filtered_skel[yy, xx] > 0:
                            neigh = int(np.sum(filtered_skel[yy-1:yy+2, xx-1:xx+2] > 0)) - 1
                            if neigh == 1:
                                endpoints.append((xx, yy))
                
                # 최장 거리 선택
                if len(endpoints) >= 2:
                    max_dist = -1
                    best_pair = None
                    for i in range(len(endpoints)):
                        for j in range(i + 1, len(endpoints)):
                            d = (endpoints[i][0] - endpoints[j][0])**2 + (endpoints[i][1] - endpoints[j][1])**2
                            if d > max_dist:
                                max_dist = d
                                best_pair = (endpoints[i], endpoints[j])
                    if best_pair:
                        endpoints = list(best_pair)

            # 5) HoughLinesP 백업 (필요시)
            if not endpoints:
                edges = cv2.Canny(bw, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                        minLineLength=int(w * self.min_line_length_ratio),
                                        maxLineGap=self.max_line_gap)
                if lines is not None and len(lines) > 0:
                    # 가장 긴 선분의 끝점 사용
                    longest_line = max(lines.reshape(-1, 4), 
                                    key=lambda line: (line[0]-line[2])**2 + (line[1]-line[3])**2)
                    endpoints = [(longest_line[0], longest_line[1]), (longest_line[2], longest_line[3])]

            # 6) 시각화
            if self.visualize:
                def to_bgr(img): 
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
                
                h, w = roi.shape[:2]
                viz = [
                    cv2.resize(roi, (w, h)),
                    to_bgr(bw),
                    to_bgr(filtered_skel),
                    to_bgr(gray),
                    to_bgr(gray),
                    to_bgr(gray)
                ]
                
                # 엔드포인트 표시
                if endpoints and len(endpoints) >= 2:
                    result_img = viz[5].copy()
                    cv2.circle(result_img, endpoints[0], 5, (0, 255, 0), -1)
                    cv2.circle(result_img, endpoints[1], 5, (0, 255, 0), -1)
                    cv2.line(result_img, endpoints[0], endpoints[1], (255, 0, 0), 2)
                    viz[5] = result_img
                
                grid = np.vstack([np.hstack(viz[:3]), np.hstack(viz[3:])])
                cv2.imshow('Improved Process Overview', grid)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # 7) 글로벌 좌표로 변환하여 반환
            if endpoints and len(endpoints) >= 2:
                return ((x1 + endpoints[0][0], y1 + endpoints[0][1]),
                        (x1 + endpoints[1][0], y1 + endpoints[1][1]))
            return None

    def draw(self, image: np.ndarray, endpoints: tuple,
             color=(0, 255, 0), radius=5, thickness=-1):
        if endpoints:
            cv2.circle(image, endpoints[0], radius, color, thickness)
            cv2.circle(image, endpoints[1], radius, color, thickness)


