import cv2
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from skimage.feature import local_binary_pattern
from skimage.morphology import skeletonize

class WireDetector:
    def __init__(self, kernel_size=3, endpoint_eps=20,
                 # 검은색 와이어 필터링 파라미터
                 black_min_thickness=6, black_max_thickness=40,
                 black_min_length=40, black_min_area=80,
                 # 빨간색 와이어 필터링 파라미터  
                 red_min_thickness=10, red_max_thickness=35,
                 red_min_length=30, red_min_area=60,
                 red_aspect_ratio=1.5):
        # 기존 파라미터 유지
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 80]),
            'red1': ([0, 80, 80], [20, 255, 255]),
            'red2': ([150, 80, 80], [180, 255, 255])
        }
        self.endpoint_eps = endpoint_eps
        
        # 검은색 와이어 파라미터
        self.black_min_thickness = black_min_thickness
        self.black_max_thickness = black_max_thickness
        self.black_min_length = black_min_length
        self.black_min_area = black_min_area
        
        # 빨간색 와이어 파라미터
        self.red_min_thickness = red_min_thickness
        self.red_max_thickness = red_max_thickness
        self.red_min_length = red_min_length
        self.red_min_area = red_min_area
        self.red_aspect_ratio = red_aspect_ratio
        
        # 기존 속성들
        self.white_block = None
        self.white_c = None
        self.full_white_mask = None

    def configure_white_thresholds(self, full_image):
        """기존 인터페이스 유지하면서 개선된 흰색 와이어 마스크 생성"""
        # 기존 방식 유지 (호환성)
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray_eq = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        
        # 기본값 설정
        self.white_block = 15
        self.white_c = -5
        
        # 개선된 흰색 마스크 생성
        hsv = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        
        thresh = cv2.adaptiveThreshold(
            gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, self.white_block, self.white_c
        )
        
        comb = cv2.bitwise_or(hsv_mask, thresh)
        
        # 개선된 필터링 적용
        comb = self.filter_by_thickness_and_shape(comb, 
                                                  self.black_min_thickness, 
                                                  self.black_max_thickness,
                                                  self.black_min_length,
                                                  self.black_min_area)
        comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, self.kernel)
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, self.kernel)
        
        self.full_white_mask = comb

    def remove_breadboard_holes(self, mask, image):
        """브레드보드 구멍 패턴 제거"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, 
                                  minRadius=2, maxRadius=6)
        
        hole_mask = np.zeros_like(mask)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(hole_mask, (x, y), r + 2, 255, -1)
        
        # 작은 점들도 제거
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 15:  # 매우 작은 점들
                cv2.fillPoly(hole_mask, [contour], 255)
        
        return cv2.bitwise_and(mask, cv2.bitwise_not(hole_mask))

    def filter_by_thickness_and_shape(self, mask, min_thickness, max_thickness, min_length, min_area, min_aspect_ratio=1.5):
        """색상별 맞춤 굵기와 형태 필터링"""
        # 거리 변환으로 굵기 측정
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        thickness_mask = np.zeros_like(mask)
        thickness_mask[dist_transform >= min_thickness/2] = 255
        thickness_mask[dist_transform > max_thickness/2] = 0
        filtered_mask = cv2.bitwise_and(mask, thickness_mask)
        
        # 형태 필터링
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter < min_length:
                continue
                
            # 형태 분석
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if min(width, height) > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio < min_aspect_ratio:  # 너무 정사각형이면 제외
                    continue
            
            cv2.fillPoly(final_mask, [contour], 255)
        
        return final_mask

    def extract_white_wire_mask(self, image, bbox):
        """기존 흰색 와이어 검출 로직 그대로 복원"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        # 1) LAB 색 공간 기반 마스크 (기존)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        _, l_mask = cv2.threshold(l_channel, 160, 255, cv2.THRESH_BINARY)
        ab_mask = cv2.inRange(cv2.merge([a_channel, b_channel]),
                            np.array([120, 120]), np.array([140, 150]))
        ab_mask = cv2.bitwise_not(ab_mask)
        combined = cv2.bitwise_and(l_mask, ab_mask)
        
        # 2) Canny 엣지 추가 (기존)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 120)
        combined = cv2.bitwise_or(combined, edges)
        
        # 3) 모폴로지로 노이즈 제거 및 연결 (기존)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel)
        
        # 4) 핀 위치(구멍) 제거 (기존)
        return self.remove_holes(combined, roi)

    def remove_holes(self, mask, image):
        """기존 메서드 개선"""
        try:
            from detector.pin_detector import PinDetector
            _, pins = PinDetector().detect_pins(image)
            hole_mask = np.zeros_like(mask)
            for x, y in pins:
                cv2.circle(hole_mask, (x, y), 8, 255, -1)
            return cv2.bitwise_and(mask, cv2.bitwise_not(hole_mask))
        except:
            # PinDetector를 사용할 수 없으면 기본 구멍 제거만 수행
            return self.remove_breadboard_holes(mask, image)

    def skeletonize(self, img):
        """개선된 스켈레톤화"""
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # scikit-image 스켈레톤화 사용 (더 robust)
        try:
            binary = (img_bin > 127).astype(np.uint8)
            skeleton = skeletonize(binary)
            skel = (skeleton * 255).astype(np.uint8)
        except:
            # 실패하면 기존 방식 사용
            skel = np.zeros(img_bin.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                open_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, element)
                temp = cv2.subtract(img_bin, open_img)
                skel = cv2.bitwise_or(skel, temp)
                img_bin = cv2.erode(img_bin, element)
                if cv2.countNonZero(img_bin) == 0:
                    break
        
        return skel

    def skeleton_to_graph(self, skel):
        """기존 메서드 유지"""
        G = nx.Graph()
        coords = np.argwhere(skel > 0)
        for y, x in coords:
            G.add_node((x, y))
        for y, x in coords:
            for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1] and skel[ny, nx_] > 0:
                    G.add_edge((x, y), (nx_, ny))
        return G

    def find_wire_endpoints_graph(self, skel):
        """개선된 끝점 검출"""
        G = self.skeleton_to_graph(skel)
        endpoints = [node for node, deg in G.degree() if deg == 1]
        
        # 끝점이 너무 많으면 클러스터링
        if len(endpoints) > 6:
            endpoints = self.cluster_endpoints(endpoints)
        
        return endpoints

    def cluster_endpoints(self, endpoints):
        """끝점 클러스터링"""
        if len(endpoints) < 2:
            return endpoints
            
        coords = np.array(endpoints)
        clustering = DBSCAN(eps=self.endpoint_eps, min_samples=1).fit(coords)
        labels = clustering.labels_
        
        clustered_endpoints = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_points = coords[labels == label]
            center = np.mean(cluster_points, axis=0).astype(int)
            clustered_endpoints.append(tuple(center))
        
        return clustered_endpoints

    def detect_wires(self, image):
        """기존 인터페이스 유지하면서 개선된 검출"""
        if self.full_white_mask is None:
            self.configure_white_thresholds(image)
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        segments = {}
        
        # 검은색 와이어 - 개선된 검출
        lo, hi = self.color_ranges['black']
        mask_b = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask_b = self.remove_breadboard_holes(mask_b, image)
        mask_b = self.filter_by_thickness_and_shape(mask_b, 
                                                   self.black_min_thickness,
                                                   self.black_max_thickness,
                                                   self.black_min_length,
                                                   self.black_min_area)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        skel_b = self.skeletonize(mask_b)
        ep_b = self.find_wire_endpoints_graph(skel_b)
        segments['black'] = {'mask': mask_b, 'skeleton': skel_b, 'endpoints': ep_b}
        
        # 빨간색 와이어 - 개선된 검출 (파라미터 조정 가능)
        m1 = cv2.inRange(hsv, *map(np.array, self.color_ranges['red1']))
        m2 = cv2.inRange(hsv, *map(np.array, self.color_ranges['red2']))
        mask_r = cv2.bitwise_or(m1, m2)
        mask_r = self.filter_by_thickness_and_shape(mask_r,
                                                   self.red_min_thickness,
                                                   self.red_max_thickness,
                                                   self.red_min_length,
                                                   self.red_min_area,
                                                   self.red_aspect_ratio)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        skel_r = self.skeletonize(mask_r)
        ep_r = self.find_wire_endpoints_graph(skel_r)
        segments['red'] = {'mask': mask_r, 'skeleton': skel_r, 'endpoints': ep_r}
        
        # 흰색 와이어 - 기존 로직 그대로 사용
        white_mask = self.extract_white_wire_mask(image, [0, 0, image.shape[1], image.shape[0]])
        skel_w = self.skeletonize(white_mask)
        ep_w = self.find_wire_endpoints_graph(skel_w)
        segments['white'] = {'mask': white_mask, 'skeleton': skel_w, 'endpoints': ep_w}
        
        return segments

    def is_good_endpoints(self, endpoints):
        """기존 메서드 유지"""
        if len(endpoints) < 2:
            return False
        coords = np.array(endpoints)
        labels = DBSCAN(eps=self.endpoint_eps, min_samples=1).fit_predict(coords)
        if len(set(labels) - {-1}) != 2:
            return False
        try:
            score = silhouette_score(coords, labels)
            return score > 0.3  # 기준 완화
        except Exception:
            return False

    def endpoint_distance_ratio(self, endpoints):
        """기존 메서드 유지"""
        dists = [(p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                 for i,p1 in enumerate(endpoints) for j,p2 in enumerate(endpoints) if j>i]
        if not dists:
            return 1.0
        dists.sort()
        return dists[-2] / dists[-1] if len(dists) > 1 else 0.0

    def select_best_endpoints(self, segments):
        """개선된 최적 끝점 선택"""
        # 각 색상별 품질 평가
        color_scores = {}
        
        for color in ['white', 'black', 'red']:
            eps = segments.get(color, {}).get('endpoints', [])
            mask = segments.get(color, {}).get('mask', np.zeros((1,1), dtype=np.uint8))
            
            if not eps:
                color_scores[color] = 0
                continue
                
            # 품질 점수 계산
            endpoint_score = 1.0 if len(eps) == 2 else max(0, 1.0 - abs(len(eps) - 2) * 0.15)
            
            # 거리 점수
            if len(eps) >= 2:
                distances = []
                for i in range(len(eps)):
                    for j in range(i+1, len(eps)):
                        d = np.sqrt((eps[i][0] - eps[j][0])**2 + (eps[i][1] - eps[j][1])**2)
                        distances.append(d)
                max_distance = max(distances) if distances else 0
                distance_score = min(1.0, max_distance / 150)
            else:
                distance_score = 0
            
            # 마스크 품질
            mask_quality = cv2.countNonZero(mask) / max(mask.shape[0] * mask.shape[1], 1)
            mask_score = min(1.0, mask_quality * 8)
            
            total_score = endpoint_score * 0.4 + distance_score * 0.4 + mask_score * 0.2
            color_scores[color] = total_score
        
        # 최고 점수 색상 선택
        best_color = max(color_scores.keys(), key=lambda c: color_scores[c]) if color_scores else 'black'
        best_endpoints = segments.get(best_color, {}).get('endpoints', [])
        
        # 가장 먼 두 점만 선택
        if len(best_endpoints) >= 2:
            max_d = 0
            pair = (best_endpoints[0], best_endpoints[1] if len(best_endpoints) > 1 else best_endpoints[0])
            for i in range(len(best_endpoints)):
                for j in range(i + 1, len(best_endpoints)):
                    dx = best_endpoints[i][0] - best_endpoints[j][0]
                    dy = best_endpoints[i][1] - best_endpoints[j][1]
                    d = dx*dx + dy*dy
                    if d > max_d:
                        max_d, pair = d, (best_endpoints[i], best_endpoints[j])
            return list(pair), best_color
        
        return best_endpoints, best_color

    def process_line_area_wires(self, warped_img, all_components, scale_factor=4):
        """기존 메서드 유지"""
        line_area_comps = [c for c in all_components if c[0].lower()=='line_area']
        if not line_area_comps:
            print("Line_area 객체가 없습니다.")
            return

        comps = []
        for cls, _, bbox in line_area_comps:
            x1,y1,x2,y2 = bbox
            roi = warped_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            h, w = roi.shape[:2]
            roi_s = cv2.resize(roi, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_LINEAR)

            segments = self.detect_wires(roi)
            best_eps, best_color = self.select_best_endpoints(segments)

            imgs = []
            imgs.append(roi_s)
            
            mask = segments.get(best_color, {}).get('mask', np.zeros_like(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
            skel = segments.get(best_color, {}).get('skeleton', np.zeros_like(mask))
            
            mask_c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            skel_c = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
            mask_c = cv2.resize(mask_c, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_NEAREST)
            skel_c = cv2.resize(skel_c, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_NEAREST)

            for pt in best_eps:
                x, y = int(pt[0]*scale_factor), int(pt[1]*scale_factor)
                cv2.circle(skel_c, (x, y), 5, (0, 0, 255), -1)

            imgs.append(np.hstack((mask_c, skel_c)))
            comp = np.vstack(imgs)
            comps.append(comp)

        if not comps:
            return

        mw = max(img.shape[1] for img in comps)
        padded = []
        for img in comps:
            h, w = img.shape[:2]
            if w < mw:
                img = cv2.copyMakeBorder(img, 0, 0, 0, mw-w, cv2.BORDER_CONSTANT, value=[0,0,0])
            padded.append(img)

        final = np.vstack(padded)
        cv2.imshow("Line_area 전선 최적 끝점 2개만 표시", final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()