import cv2
import numpy as np
from skimage.morphology import skeletonize

class ImprovedResistorEndpointDetector:
    """
    개선된 저항 엔드포인트 검출기
    - Contour 극점 찾기, PCA, 이웃 픽셀 검사 등 다양한 감지 방법 선택 가능
    - 실시간 파라미터 조정을 위한 속성 업데이트 지원
    """
    def __init__(self, **kwargs):
        # 기본 파라미터 설정
        self.params = {
            'threshold_method': 'otsu', # 'otsu', 'fixed', 'adaptive'
            'fixed_threshold': 120,
            'use_opening': True,
            'use_closing': True,
            'morph_kernel_size': 1,
            'morph_iterations': 1,
            'min_resistor_area': 100,
            'max_resistor_area': 10000,
            'min_aspect_ratio': 1.5,
            'max_aspect_ratio': 30.0,
            'endpoint_method': 'contour', # 'contour', 'pca', 'neighbor'
            'visualize': False
        }
        # 생성자에서 받은 파라미터로 업데이트
        self.params.update(kwargs)

    def _find_contour_extremes(self, binary_img):
        """
        Contour와 minAreaRect를 이용해 저항의 양쪽 극점을 찾는 방법.
        회전된 객체에 매우 강건함.
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # 가장 큰 contour를 저항으로 간주
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 윤곽선 포인트가 너무 적으면 실패 처리
        if len(largest_contour) < 5:
            return []

        # 최소 면적 사각형 계산
        rect = cv2.minAreaRect(largest_contour)
        
        # 주축(장축) 벡터 계산
        box = cv2.boxPoints(rect)
        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]
        
        major_axis = edge1 if np.linalg.norm(edge1) > np.linalg.norm(edge2) else edge2
        major_axis_norm = major_axis / np.linalg.norm(major_axis)
        
        # 모든 contour 포인트를 주축에 투영
        contour_points = largest_contour.squeeze(axis=1)
        projections = contour_points.dot(major_axis_norm)
        
        # 투영 값이 최소/최대인 점이 양 끝점
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        endpoint1 = tuple(contour_points[min_idx])
        endpoint2 = tuple(contour_points[max_idx])
        
        return [endpoint1, endpoint2]

    def _find_pca_endpoints(self, binary_img):
        """ PCA를 사용하여 주축 방향을 찾고 양 끝점 검출 """
        points = np.column_stack(np.where(binary_img > 0))
        if len(points) < 10:
            return []
        
        points = points.astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points, mean=None)
        principal_axis = eigenvectors[0]
        
        projections = np.dot(points - mean, principal_axis)
        
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        endpoint1 = (int(points[min_idx][1]), int(points[min_idx][0]))
        endpoint2 = (int(points[max_idx][1]), int(points[max_idx][0]))
        
        return [endpoint1, endpoint2]

    def _find_neighbor_endpoints(self, skel_img):
        """ 스켈레톤에서 이웃이 1개인 점을 끝점으로 검출 """
        endpoints = []
        h, w = skel_img.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skel_img[y, x] > 0:
                    # 3x3 이웃 픽셀의 합계를 계산하여 이웃 수 확인
                    neighbors = np.sum(skel_img[y-1:y+2, x-1:x+2] > 0) - 1
                    if neighbors == 1:
                        endpoints.append((x, y))
        
        if len(endpoints) < 2:
            return endpoints

        # 가장 거리가 먼 두 점을 최종 끝점으로 선택
        max_dist = -1
        best_pair = None
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                dist_sq = (endpoints[i][0] - endpoints[j][0])**2 + (endpoints[i][1] - endpoints[j][1])**2
                if dist_sq > max_dist:
                    max_dist = dist_sq
                    best_pair = (endpoints[i], endpoints[j])
        
        return list(best_pair) if best_pair else []


    def extract(self, image: np.ndarray, bbox: tuple):
        p = self.params
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        if p['threshold_method'] == 'otsu':
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif p['threshold_method'] == 'fixed':
            _, bw = cv2.threshold(gray, p['fixed_threshold'], 255, cv2.THRESH_BINARY_INV)
        else: # adaptive
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p['morph_kernel_size'], p['morph_kernel_size']))
        if p['use_opening']:
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=p['morph_iterations'])
        if p['use_closing']:
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=p['morph_iterations'])

        # Contour 기반 필터링
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(bw)
        
        valid_contours = []
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (p['min_resistor_area'] <= area <= p['max_resistor_area']):
                    continue
                
                if len(cnt) < 5: continue
                rect = cv2.minAreaRect(cnt)
                (w_rect, h_rect) = rect[1]
                aspect_ratio = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1)

                if p['min_aspect_ratio'] <= aspect_ratio <= p['max_aspect_ratio']:
                    valid_contours.append(cnt)

        if not valid_contours:
            return None

        # 가장 큰 유효 contour를 사용
        largest_valid_contour = max(valid_contours, key=cv2.contourArea)
        cv2.drawContours(filtered_mask, [largest_valid_contour], -1, 255, -1)
        
        endpoints = []
        if p['endpoint_method'] == 'contour':
            endpoints = self._find_contour_extremes(filtered_mask)
        elif p['endpoint_method'] == 'pca':
            endpoints = self._find_pca_endpoints(filtered_mask)
        elif p['endpoint_method'] == 'neighbor':
            skel = skeletonize(filtered_mask // 255).astype(np.uint8) * 255
            endpoints = self._find_neighbor_endpoints(skel)

        if self.params['visualize']:
            self.show_visualization(roi, bw, filtered_mask, endpoints)

        if endpoints and len(endpoints) == 2:
            return ((x1 + endpoints[0][0], y1 + endpoints[0][1]),
                    (x1 + endpoints[1][0], y1 + endpoints[1][1]))
        return None

    def draw(self, image: np.ndarray, endpoints: tuple, color=(0, 255, 0), radius=5, thickness=-1):
        if endpoints:
            cv2.circle(image, endpoints[0], radius, color, thickness)
            cv2.circle(image, endpoints[1], radius, color, thickness)
            cv2.line(image, endpoints[0], endpoints[1], (255, 0, 0), 2)
    
    def show_visualization(self, roi, bw, filtered_mask, endpoints):
        result_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        self.draw(result_img, endpoints)

        viz_grid = np.hstack([
            cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR),
            result_img
        ])
        cv2.imshow("Debug Visualization", viz_grid)