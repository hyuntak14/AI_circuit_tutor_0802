import cv2
import numpy as np
import networkx as nx
from hole_detector import HoleDetector
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from skimage.feature import local_binary_pattern

class WireDetector:
    def __init__(self, kernel_size=3, endpoint_eps=20):
        # Morphology kernel for noise filtering
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # HSV color ranges for black and red wires
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 80]),
            'red1': ([0, 80, 80], [20, 255, 255]),
            'red2': ([150, 80, 80], [180, 255, 255])
        }
        # Hole detector for pin removal
        self.hole_detector = HoleDetector()
        # Parameters for adaptive thresholding of white wires
        self.white_block = None
        self.white_c = None
        self.full_white_mask = None
        # DBSCAN eps for endpoint clustering
        self.endpoint_eps = endpoint_eps

    def configure_white_thresholds(self, full_image):
        
        # Compute optimal adaptive threshold parameters for white wires
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        
        # 더 강한 명암비 강화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray_eq = clahe.apply(gray)
        
        # 추가적인 필터링
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        
        # 다양한 블록 크기 시도
        block_sizes = [7, 11, 15, 19, 23]
        c_values = [-5, 0, 5, 10, 15]
        
        self.white_block, self.white_c = self.hole_detector.find_best_threshold_params(
            blurred, block_sizes=block_sizes, c_values=c_values
        )
        if self.white_block is None:
            raise RuntimeError("흰 전선 임계치 파라미터를 찾지 못했습니다.")
        # HSV-based mask for bright (white) pixels
        hsv = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, self.kernel)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, self.kernel)
        # Adaptive threshold inversion
        thresh = cv2.adaptiveThreshold(
            gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, self.white_block, self.white_c
        )
        # Combine HSV and adaptive results
        comb = cv2.bitwise_or(hsv_mask, thresh)
        comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, self.kernel)
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, self.kernel)
        self.full_white_mask = comb

    
    '''def extract_white_wire_mask(self, image, bbox):
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        # LAB 색 공간으로 변환
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # L 채널에서 밝은 영역 추출 (흰색)
        _, l_mask = cv2.threshold(l_channel, 160, 255, cv2.THRESH_BINARY)
        
        # a,b 채널에서 특정 색상 범위 제외 (살색 제외)
        ab_mask = cv2.inRange(cv2.merge([a_channel, b_channel]), 
                            np.array([120, 120]), np.array([140, 150]))
        ab_mask = cv2.bitwise_not(ab_mask)
        
        # 두 마스크 결합
        combined = cv2.bitwise_and(l_mask, ab_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel)
        
        return self.remove_holes(combined, roi)'''
    

    def extract_white_wire_mask(self, image, bbox):
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
        
        # 2) Canny 엣지 추가
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 120)
        combined = cv2.bitwise_or(combined, edges)
        
        # 3) 모폴로지로 노이즈 제거 및 연결
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel)
        
        # 4) 핀 위치(구멍) 제거
        return self.remove_holes(combined, roi)


    def remove_holes(self, mask, image):
        # Remove pin-hole regions from mask
        from detector.pin_detector import PinDetector
        _, pins = PinDetector().detect_pins(image)
        hole_mask = np.zeros_like(mask)
        for x, y in pins:
            cv2.circle(hole_mask, (x, y), 8, 255, -1)
        return cv2.bitwise_and(mask, cv2.bitwise_not(hole_mask))

    def skeletonize(self, img):
        # Extract skeleton lines
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
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
        # Convert skeleton pixels to graph nodes and edges
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
        # Endpoints are graph nodes with degree==1
        G = self.skeleton_to_graph(skel)
        return [node for node, deg in G.degree() if deg == 1]

    def detect_wires(self, image):
        # Ensure white mask initialized
        if self.full_white_mask is None:
            self.configure_white_thresholds(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        segments = {}
        # Black wire detection
        lo, hi = self.color_ranges['black']
        mask_b = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, self.kernel)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, self.kernel)
        skel_b = self.skeletonize(mask_b)
        ep_b = self.find_wire_endpoints_graph(skel_b)
        segments['black'] = {'mask': mask_b, 'skeleton': skel_b, 'endpoints': ep_b}
        # Red wire detection
        m1 = cv2.inRange(hsv, *map(np.array, self.color_ranges['red1']))
        m2 = cv2.inRange(hsv, *map(np.array, self.color_ranges['red2']))
        mask_r = cv2.bitwise_or(m1, m2)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, self.kernel)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, self.kernel)
        skel_r = self.skeletonize(mask_r)
        ep_r = self.find_wire_endpoints_graph(skel_r)
        segments['red'] = {'mask': mask_r, 'skeleton': skel_r, 'endpoints': ep_r}
        # White wire detection using full_white_mask

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # LBP 이미지에서 특정 패턴 범위를 선택
        lbp_mask = ((lbp >= 2) & (lbp <= 6)).astype(np.uint8) * 255
        
        # 기존 HSV 마스크와 결합
        white_mask = cv2.bitwise_or(self.extract_white_wire_mask(image, [0, 0, image.shape[1], image.shape[0]]), lbp_mask)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, self.kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, self.kernel)
        
        skel_w = self.skeletonize(white_mask)
        ep_w = self.find_wire_endpoints_graph(skel_w)
        segments['white'] = {'mask': white_mask, 'skeleton': skel_w, 'endpoints': ep_w}
        return segments



        '''mask_w = self.full_white_mask.copy()
        mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, self.kernel)
        mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, self.kernel)
        skel_w = self.skeletonize(mask_w)
        ep_w = self.find_wire_endpoints_graph(skel_w)
        segments['white'] = {'mask': mask_w, 'skeleton': skel_w, 'endpoints': ep_w}
        return segments'''

    def is_good_endpoints(self, endpoints):
        # Check for exactly 2 well-separated clusters
        if len(endpoints) < 2:
            return False
        coords = np.array(endpoints)
        labels = DBSCAN(eps=self.endpoint_eps, min_samples=1).fit_predict(coords)
        if len(set(labels) - {-1}) != 2:
            return False
        try:
            score = silhouette_score(coords, labels)
            return score > 0.5
        except Exception:
            return False

    def endpoint_distance_ratio(self, endpoints):
        # Ratio of second-largest to largest squared distance
        dists = [(p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                 for i,p1 in enumerate(endpoints) for j,p2 in enumerate(endpoints) if j>i]
        if not dists:
            return 1.0
        dists.sort()
        return dists[-2] / dists[-1] if len(dists) > 1 else 0.0

    def select_best_endpoints(self, segments):
        # 1) White/black/red 순서로 클러스터링이 잘 된 채널부터
        #    항상 가장 먼 두 점(farthest-pair)만 골라 반환
        for color in ['white', 'black', 'red']:
            eps = segments.get(color, {}).get('endpoints', [])
            if eps and self.is_good_endpoints(eps):
                # farthest-pair 계산
                max_d = 0
                pair = (eps[0], eps[0])
                for i in range(len(eps)):
                    for j in range(i + 1, len(eps)):
                        dx = eps[i][0] - eps[j][0]
                        dy = eps[i][1] - eps[j][1]
                        d = dx*dx + dy*dy
                        if d > max_d:
                            max_d, pair = d, (eps[i], eps[j])
                return list(pair), color

        # 2) Fallback: 스켈레톤 픽셀 수가 가장 많은 채널 선택
        best_color, best_count = None, -1
        for color, seg in segments.items():
            cnt = cv2.countNonZero(seg.get('skeleton', np.zeros_like(seg.get('mask', []))))
            if cnt > best_count:
                best_color, best_count = color, cnt

        if best_color is None:
            return [], None

        endpoints = segments[best_color].get('endpoints', [])
        if len(endpoints) < 2:
            return endpoints, best_color

        # 3) 그 외엔 두 점 간 거리가 가장 먼 farthest-pair 선택
        max_d = 0
        pair = (endpoints[0], endpoints[1])
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                dx = endpoints[i][0] - endpoints[j][0]
                dy = endpoints[i][1] - endpoints[j][1]
                d = dx*dx + dy*dy
                if d > max_d:
                    max_d, pair = d, (endpoints[i], endpoints[j])
        return list(pair), best_color


    def process_line_area_wires(self, warped_img, all_components, scale_factor=4):
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

            # 1) 영역 확대
            h, w = roi.shape[:2]
            roi_s = cv2.resize(roi, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_LINEAR)

            # 2) 와이어 검출
            segments = self.detect_wires(roi)

            # 3) 최적 끝점 2개 선택
            best_eps, best_color = self.select_best_endpoints(segments)
            # best_eps 는 [(x1,y1), (x2,y2)] 형태, best_color 는 'white'/'black'/'red' 중 하나

            # 4) 시각화를 위한 이미지 준비
            imgs = []
            # 원본 뷰
            imgs.append(roi_s)
            # 선택된 끝점 표시
            mask = segments.get(best_color, {}).get('mask', np.zeros_like(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
            skel = segments.get(best_color, {}).get('skeleton', np.zeros_like(mask))
            # 크기 맞추기
            mask_c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            skel_c = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
            mask_c = cv2.resize(mask_c, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_NEAREST)
            skel_c = cv2.resize(skel_c, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_NEAREST)

            # 선택된 끝점만 붉은 점으로 표시
            for pt in best_eps:
                # pt 좌표는 ROI 기준이므로, scale_factor 적용
                x, y = int(pt[0]*scale_factor), int(pt[1]*scale_factor)
                cv2.circle(skel_c, (x, y), 5, (0, 0, 255), -1)

            imgs.append(np.hstack((mask_c, skel_c)))

            # 5) 결과 합치기
            comp = np.vstack(imgs)
            comps.append(comp)

        if not comps:
            return

        # 세로로 이어 붙여서 한 창에 표시
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
