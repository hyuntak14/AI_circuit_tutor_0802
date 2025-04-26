import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class HoleDetector:
    """
    Breadboard 구멍 검출 & 실제 연결 토폴로지(net) 추론 클래스
    """

    def __init__(self,
                 block_sizes=[11,15,19],
                 c_values=[0,5,10,15],
                 min_area=5,
                 max_area=150,
                 aspect_ratio_range=(0.5,1.5),
                 circ_threshold=0.1,
                 y_threshold_range=(5,30),
                 y_step=0.5,
                 target_rows=17):
        self.block_sizes        = block_sizes
        self.c_values           = c_values
        self.min_area           = min_area
        self.max_area           = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.circ_threshold     = circ_threshold
        self.y_threshold_range  = y_threshold_range
        self.y_step             = y_step
        self.target_rows        = target_rows


    def _filter_valid_holes(self, contours, draw=False, image=None, return_stats=False):
        """
        public 버전: return_stats=True 로 circularities, aspect_ratios 까지 함께 반환
        """
        hole_centers = []
        circularities = []
        aspect_ratios = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / float(h)
            if not (self.aspect_ratio_range[0] <= ar <= self.aspect_ratio_range[1]):
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ < self.circ_threshold:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            hole_centers.append((cx, cy))
            circularities.append(circ)
            aspect_ratios.append(ar)
            if draw and image is not None:
                epsilon = 0.02 * peri
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
                cv2.drawContours(image, [approx], -1, (255, 0, 0), 2)
        if return_stats:
            return hole_centers, circularities, aspect_ratios
        return hole_centers

    def detect_holes(self, image):
        """
        이미지에서 구멍(center) 좌표를 검출해 반환
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)

        # 최적 파라미터 탐색
        best_score = -np.inf
        best_params = (None, None)
        for b in self.block_sizes:
            for c in self.c_values:
                th = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV,
                    b, c
                )
                cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                centers = self._filter_valid_holes(cnts)
                if len(centers) > best_score:
                    best_score = len(centers)
                    best_params = (b, c)

        # 최적 파라미터로 확정
        b, c = best_params
        th = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            b, c
        )
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._filter_valid_holes(cnts)

    def group_rows(self, holes, y_thresh):
        """
        y_thresh 픽셀 간격으로 holes를 행별로 그룹화
        반환: [(row_idx, [(x,y),…], median_y), …]
        """
        holes_sorted = sorted(holes, key=lambda p: p[1])
        rows = []
        for pt in holes_sorted:
            if not rows:
                rows.append([pt])
            else:
                med_y = np.median([p[1] for p in rows[-1]])
                if abs(pt[1] - med_y) < y_thresh:
                    rows[-1].append(pt)
                else:
                    rows.append([pt])
        result = []
        for i, grp in enumerate(rows):
            pts = sorted(grp, key=lambda p: p[0])
            med_y = np.median([p[1] for p in pts])
            result.append((i, pts, med_y))
        return result

    def find_optimal_y(self, holes):
        """
        group_rows의 y_thresh를 자동 탐색
        """
        best = (None, None, float('inf'))
        for th in np.arange(self.y_threshold_range[0],
                            self.y_threshold_range[1] + self.y_step,
                            self.y_step):
            gr = self.group_rows(holes, th)
            row_cnt = len(gr)
            dev = sum(abs(len(g[1]) - 50) for g in gr) / row_cnt if row_cnt else np.inf
            score = abs(row_cnt - self.target_rows) + dev
            if score < best[2]:
                best = (th, gr, score)
        return best  # (best_y_thresh, grouped_rows, score)

    def cluster_columns(self, holes, x_eps=None):
        """
        DBSCAN으로 x 좌표 기준 column-cluster 생성
        반환: [[(x,y),…], …] (x 평균 오름차순)
        """
        xs = np.array([[x] for x,y in holes])
        if x_eps is None:
            diffs = np.diff(np.sort(xs.flatten()))
            x_eps = float(np.median(diffs)) * 0.5
        labels = DBSCAN(eps=x_eps, min_samples=1).fit_predict(xs)
        clusters = {}
        for pt, lbl in zip(holes, labels):
            clusters.setdefault(lbl, []).append(pt)
        # x 평균 기준 오름차순 정렬
        return [clusters[k] for k in sorted(clusters, key=lambda k: np.mean([p[0] for p in clusters[k]]))]

    def _cluster_by_pattern(self, pts, pattern):
        """
        순차적 분할: pts 정렬 뒤 pattern만큼 자르기,
        부족 시 보간(interpolation)으로 보충
        """
        pts = sorted(pts, key=lambda p: p[0])
        total = sum(pattern)
        if len(pts) != total:
            xs = [p[0] for p in pts]
            x0, x1 = min(xs), max(xs)
            y_med = int(np.median([p[1] for p in pts]))
            pts = [(float(x), y_med) for x in np.linspace(x0, x1, total)]
        clusters = []
        idx = 0
        for sz in pattern:
            clusters.append(pts[idx:idx+sz])
            idx += sz
        return clusters

    def get_board_nets(self, holes, y_thresh=None, x_eps=None):
        """
        최종 넷 리스트 생성
        반환: list of [ (x,y), … ] 각 넷에 속한 hole 좌표 리스트
        """
        # 1) 행 그룹화
        if y_thresh is None:
            y_thresh, gr, _ = self.find_optimal_y(holes)
        else:
            gr = self.group_rows(holes, y_thresh)

        # 분리: rails vs terminal
        rails = gr[:4]    # 맨 위 4행
        term  = gr[4:]    # 터미널 스트립

        nets = []
        # 2) Rails: 가로 5홀씩 → 10 nets per row
        for _, pts, _ in rails:
            nets.extend(self._cluster_by_pattern(pts, [5]*10))

        # 3) Terminal: 열 클러스터링 → vertical nets
        term_holes = [pt for _, pts, _ in term for pt in pts]
        col_clusters = self.cluster_columns(term_holes, x_eps)

        # pattern for terminal columns: 2,5,5,2,5,5,2,5,5,2,5,5,2
        pattern13 = [2,5,5,2,5,5,2,5,5,2,5,5,2]
        idx = 0
        for sz in pattern13:
            if sz == 2:
                # 세로 5홀 열(두 개 consecutive columns each)
                for j in range(idx, idx+2):
                    nets.append(col_clusters[j])
            idx += sz

        # 4) Terminal: 각 행별 가로 5홀 nets
        for _, pts, _ in term:
            clusters = self._cluster_by_pattern(pts, pattern13)
            for c in clusters:
                if len(c) == 5:
                    nets.append(c)

        return nets

    def get_row_nets(self, holes, y_thresh=None):
        if y_thresh is None:
            y_thresh, gr, _ = self.find_optimal_y(holes)
        else:
            gr = self.group_rows(holes, y_thresh)
        pattern13 = [2,5,5,2,5,5,2,5,5,2,5,5,2]
        row_nets = []
        for row_idx, pts, _ in gr:
            pattern = [5] * 10 if row_idx < 4 else pattern13
            clusters = self._cluster_by_pattern(pts, pattern)
            row_nets.append((row_idx, clusters))
        return sorted(row_nets, key=lambda x: x[0])





    def find_best_threshold_params(self, gray_img, block_sizes, c_values):
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        best_score = -np.inf
        best_params = (None, None)
        for block_size in block_sizes:
            for c_val in c_values:
                thresh = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, block_size, c_val
                )
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 수정: class 메서드를 사용해 통계 정보 추출
                hole_centers, circularities, aspect_ratios = self._filter_valid_holes(contours, return_stats=True)
                valid = len(hole_centers)
                if valid == 0:
                    continue
                mean_circ = np.mean(circularities)
                mean_ar = np.mean(aspect_ratios)
                noise_ratio = 1 - (valid / len(contours)) if contours else 1
                score = (
                    0.5 * (1 - abs(mean_circ - 0.785)) +
                    0.3 * (1 - abs(mean_ar - 1.0)) -
                    0.2 * noise_ratio
                )
                if score > best_score:
                    best_score = score
                    best_params = (block_size, c_val)
        return best_params

    