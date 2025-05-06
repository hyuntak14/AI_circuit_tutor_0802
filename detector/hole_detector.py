import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
class HoleDetector:
    PATTERN = [2, 5, 5, 2, 5, 5, 2, 5, 5, 2,5, 5, 2]
    """
    Breadboard 구멍 검출 & 실제 연결 토폴로지 추론 클래스
    - raw detection + template-based affine alignment with reprojection error filtering
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
                 target_rows=17,
                 template_csv_path=None,
                 template_image_path=None,
                 max_nn_dist=20.0):
        self.block_sizes = block_sizes
        self.c_values = c_values
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.circ_threshold = circ_threshold
        self.y_threshold_range = y_threshold_range
        self.y_step = y_step
        self.target_rows = target_rows
        self.template_csv_path = template_csv_path
        self.template_image_path = template_image_path
        self.max_nn_dist = max_nn_dist

    def _filter_valid_holes(self, contours, return_stats=False):
        holes = []
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
            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])
            holes.append((cx, cy))
            circularities.append(circ)
            aspect_ratios.append(ar)
        if return_stats:
            return holes, circularities, aspect_ratios
        else:
            return holes

    '''def get_board_nets(self, holes, base_img=None, show=False):
        """
        holes: list of (x, y) tuples
        base_img: 이미지 (show=True 시 시각화용)
        show: Boolean
        returns: list of nets (각 net: list of (x,y))
        """
        # --- Rails(상단 4행) 분리 ---
        holes_sorted = sorted(holes, key=lambda p: p[1])
        N_TOP = 4 * 50
        if len(holes_sorted) < N_TOP:
            raise ValueError(f"구멍 개수({len(holes_sorted)})가 상단 4행({N_TOP})에 부족합니다.")
        top4 = holes_sorted[:N_TOP]
        other = holes_sorted[N_TOP:]
        print(f"[DEBUG] 총 구멍: {len(holes)}, rails(상단4행): {len(top4)}, other: {len(other)}")

        nets = [top4[i*50:(i+1)*50] for i in range(4)]

        # --- 나머지 행 클러스터링 (y 기준) ---
        ys = np.array([p[1] for p in other])
        # 중복 제거한 y값으로만 차분 계산
        unique_ys = np.unique(ys)
        diffs = np.diff(np.sort(unique_ys))
        #eps_y = (np.median(diffs) * 0.5) if len(diffs) > 0 else 1.0
        eps_y = 1.5
        # DBSCAN 은 1D 데이터에 대해서도 동작
        labels_y = DBSCAN(eps=eps_y, min_samples=1).fit_predict(ys.reshape(-1,1))

        row_groups = {}
        for pt, lbl in zip(other, labels_y):
            row_groups.setdefault(lbl, []).append(pt)
        # y 평균값 순서로 정렬된 행 리스트
        rows = [row_groups[k] for k in sorted(row_groups, key=lambda k: np.mean([p[1] for p in row_groups[k]]))]

        # --- 패턴별 영역 계산 (기존 코드 유지) ---
        first_row = sorted(rows[0], key=lambda p: p[0])
        xs = [p[0] for p in first_row]
        total = sum(self.PATTERN)
        if len(xs) != total:
            xm, xM = min(xs), max(xs)
            width = (xM - xm) / len(self.PATTERN)
            bounds = [(xm + i*width, xm + (i+1)*width) for i in range(len(self.PATTERN))]
        else:
            bounds, idx = [], 0
            for sz in self.PATTERN:
                seg = xs[idx:idx+sz]
                bounds.append((min(seg), max(seg)))
                idx += sz

        # --- other_nets 생성 ---
        other_nets = []
        for (xmin, xmax), sz in zip(bounds, self.PATTERN):
            if sz == 2:
                col = [p for p in other if xmin-1 <= p[0] <= xmax+1]
                other_nets.append(sorted(col, key=lambda p: p[1]))
            else:
                for row in rows:
                    grp = [p for p in row if xmin-1 <= p[0] <= xmax+1]
                    other_nets.append(sorted(grp, key=lambda p: p[0]))
        print(f"[DEBUG] other_nets 개수: {len(other_nets)}")

        nets.extend(other_nets)

        # --- 전체 nets 시각화 ---
        if base_img is not None and show:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR) if len(base_img.shape)==2 else base_img.copy()
            rng = np.random.default_rng(123)
            colors = [tuple(int(v) for v in rng.integers(0,255,3)) for _ in nets]
            print(f"[DEBUG] show 직전 nets 개수: {len(nets)}")
            for idx, cluster in enumerate(nets):
                color = colors[idx]
                for i,(x,y) in enumerate(cluster):
                    cv2.circle(vis, (int(x),int(y)), 3, color, -1)
                    if i < len(cluster)-1:
                        x2,y2 = cluster[i+1]
                        cv2.line(vis,(int(x),int(y)),(int(x2),int(y2)),color,1)
            cv2.imshow('Cluster Visualization2', vis)
            cv2.waitKey(0)
            cv2.destroyWindow('Cluster Visualization2')

        return nets'''

    '''def get_board_nets(self, holes, base_img=None, show=False):
        """
        holes: list of (x, y) tuples
        base_img: 이미지 (show=True 시 시각화용)
        show: Boolean
        returns: list of nets (각 net: list of (x,y)))
        """
        import numpy as np
        import cv2

        # 1) 상단 4행 (Rails)
        holes_sorted = sorted(holes, key=lambda p: p[1])
        N_TOP = 4 * 50
        if len(holes_sorted) < N_TOP:
            raise ValueError(f"구멍 개수({len(holes_sorted)})가 상단 4행({N_TOP})에 부족합니다.")
        rails = holes_sorted[:N_TOP]
        other = holes_sorted[N_TOP:]
        print(f"[DEBUG] 총 구멍: {len(holes)}, rails(상단4행): {len(rails)}, other: {len(other)}")

        # Rails 4개 net
        nets = [rails[i*50:(i+1)*50] for i in range(4)]

        # 2) PATTERN 기반 구간(bounds) 계산
        #    대표 행(first_row)의 x 좌표로부터 구간 경계 산출
        first_row = sorted(other[:150], key=lambda p: p[0])
        xs = [p[0] for p in first_row]
        total_slots = sum(self.PATTERN)
        if len(xs) == total_slots:
            bounds = []
            idx = 0
            for sz in self.PATTERN:
                seg = xs[idx:idx+sz]
                bounds.append((min(seg), max(seg)))
                idx += sz
        else:
            # 만약 first_row 구멍 개수가 패턴과 다르면 균등 분할
            xm, xM = min(xs), max(xs)
            width = (xM - xm) / len(self.PATTERN)
            bounds = [(xm + i*width, xm + (i+1)*width) for i in range(len(self.PATTERN))]

        # 3) other 구멍들을 각 bounds 별로 묶기 (margin 허용)
        margin = 2  # 좌표 오차 허용범위 (픽셀)
        other_nets = []
        for xmin, xmax in bounds:
            # 이 패턴이 폭 2(세로열 1줄)인지 판단
            span = xmax - xmin
            group = [p for p in other if (xmin - margin) <= p[0] <= (xmax + margin)]
            # 세로열(폭이 작으면): y 기준 정렬, 
            # 그렇지 않으면 각 행별로 x 기준 정렬
            other_nets.append(sorted(group, key=lambda p: p[1] if span < (width*0.5) else p[0]))

        print(f"[DEBUG] PATTERN 기반 other_nets 개수: {len(other_nets)}")

        nets.extend(other_nets)

        # 4) (옵션) 시각화
        if base_img is not None and show:
            vis = base_img.copy()
            rng = np.random.default_rng(123)
            colors = [tuple(int(v) for v in rng.integers(0, 255, size=3)) for _ in nets]
            print(f"[DEBUG] 시각화 직전 nets 개수: {len(nets)}")
            for idx, cluster in enumerate(nets):
                c = colors[idx]
                for i, (x, y) in enumerate(cluster):
                    cv2.circle(vis, (int(x), int(y)), 3, c, -1)
                    if i < len(cluster) - 1:
                        x2, y2 = cluster[i+1]
                        cv2.line(vis, (int(x), int(y)), (int(x2), int(y2)), c, 1)
            cv2.imshow('Cluster Visualization2', vis)
            cv2.waitKey(0)
            cv2.destroyWindow('Cluster Visualization2')

        return nets'''



    def get_board_nets(self, holes, base_img=None, show=False):
        """
        holes: list of (x, y) tuples (affine 변환된 좌표)
        base_img: 이미지 (show=True 시 시각화용)
        show: Boolean
        returns: list of nets (각 net: list of (x, y)))
        """
        # 1) numpy 배열로 변환
        coords = np.array(holes, dtype=np.float32)

        # 2) DBSCAN 클러스터링 (eps, min_samples는 필요에 맞게 조정)
        db = DBSCAN(eps=12, min_samples=2).fit(coords)
        labels = db.labels_

        # 3) 클러스터별 포인트 묶기 (노이즈 라벨 -1 은 제외)
        unique_labels = sorted(set(labels))  # e.g. [-1, 0,1,2...]
        nets = [
            coords[labels == lbl].tolist()
            for lbl in unique_labels
            if lbl != -1
        ]

        # 4) (옵션) 시각화
        if show and base_img is not None:
            # BGR 채널 확보
            if base_img.ndim == 2:
                viz = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            else:
                viz = base_img.copy()

            rng = np.random.default_rng(123)
            colors = [
                tuple(int(v) for v in rng.integers(0, 256, size=3).tolist())
                for _ in nets
            ]

            for idx, pts in enumerate(nets):
                color = colors[idx]
                if not pts:
                    continue

                # 직선 혹은 수직 판단
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                width  = max(xs) - min(xs)
                height = max(ys) - min(ys)

                # 정렬 기준 선택
                if height > width:
                    sorted_pts = sorted(pts, key=lambda p: p[1])
                else:
                    sorted_pts = sorted(pts, key=lambda p: p[0])

                # 폴리라인 연결
                for i in range(len(sorted_pts) - 1):
                    x1, y1 = map(lambda v: int(round(v)), sorted_pts[i])
                    x2, y2 = map(lambda v: int(round(v)), sorted_pts[i + 1])
                    cv2.line(viz, (x1, y1), (x2, y2), color, 1)

                # 구멍 위치 표시
                for x, y in pts:
                    cx, cy = int(round(x)), int(round(y))
                    cv2.circle(viz, (cx, cy), 3, color, -1)

            cv2.imshow('Affine-aligned Holes Clusters', viz)
            cv2.waitKey(0)
            cv2.destroyWindow('Affine-aligned Holes Clusters')

        return nets


    def detect_holes(self, image, visualize_nets=False):
        raw = self.detect_holes_raw(image)
        holes = raw
        if len(raw) >= 3 and self.template_csv_path and os.path.exists(self.template_csv_path):
            pts_raw = np.array(raw, dtype=np.float32)
            tpl = np.loadtxt(self.template_csv_path, delimiter=',', dtype=np.float32)
            scale_x = scale_y = 1.0
            if self.template_image_path and os.path.exists(self.template_image_path):
                tmpl = cv2.imread(self.template_image_path)
                if tmpl is not None:
                    ih, iw = image.shape[:2]
                    th, tw = tmpl.shape[:2]
                    scale_x, scale_y = iw / tw, ih / th
            tpl_scaled = np.array([(x * scale_x, y * scale_y) for x, y in tpl], dtype=np.float32)
            self.template_pts = tpl_scaled.tolist()
            nbrs = NearestNeighbors(n_neighbors=1).fit(pts_raw)
            dists, inds = nbrs.kneighbors(tpl_scaled.reshape(-1, 2))
            mask = dists.flatten() < self.max_nn_dist
            if mask.sum() >= 3:
                src = tpl_scaled[mask].reshape(-1, 1, 2)
                dst = pts_raw[inds.flatten()[mask]].reshape(-1, 1, 2)
                M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    holes = [tuple(p) for p in cv2.transform(tpl_scaled.reshape(-1, 1, 2), M).reshape(-1, 2)]
        nets = self.get_board_nets(holes, base_img=image, show=visualize_nets)
        if visualize_nets:
            return holes, nets
        else:
            return holes

    def remove_spatial_outliers(self, holes, eps=5.0, min_samples=3):
        pts = np.array(holes)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
        return [tuple(p) for p,lbl in zip(pts, labels) if lbl != -1]

    def remove_far_outliers(self, holes, z_thresh=2.0):
        pts = np.array(holes)
        nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
        dists, _ = nbrs.kneighbors(pts)
        dist1 = dists[:,1]
        mean, std = dist1.mean(), dist1.std()
        return [tuple(p) for p,keep in zip(pts, dist1 < mean + z_thresh*std) if keep]

    def custom_cluster_rows_and_columns(self, holes):
        top_rows = [holes[i * 50:(i + 1) * 50] for i in range(4)]
        upper_clusters = top_rows

        pattern = [2, 5, 5, 2, 5, 5, 2, 5, 5, 2, 5, 5, 2]
        remaining = holes[200:]
        col_clusters = []
        idx = 0

        for width in pattern:
            count = width * 17
            col = remaining[idx:idx + count]
            if width == 2:
                # 2열 → 2줄로 나누기 (세로로)
                for i in range(0, len(col), 17):
                    col_clusters.append(col[i:i+17])
            else:
                # 5열 → 행 기준으로 5개씩 클러스터링
                for i in range(17):
                    offset = i * 5
                    col_clusters.append(col[offset:offset + 5])
            idx += count

        return upper_clusters + col_clusters

    def visualize_clusters(self, base_img, clusters, affine_pts=None, window_name='Cluster Visualization123123'):
        img = base_img.copy()
        if affine_pts:
            for x, y in affine_pts:
                cv2.circle(img, (int(round(x)), int(round(y))), 4, (0,255,0), -1)
        rng = np.random.default_rng(42)
        colors = [tuple(int(c) for c in rng.integers(0, 255, size=3)) for _ in range(len(clusters))]

        for cluster_id, pts in enumerate(clusters):
            color = colors[cluster_id]
            for x, y in pts:
                cv2.circle(img, (int(round(x)), int(round(y))), 3, color, -1)
            for i in range(len(pts) - 1):
                x1, y1 = map(int, pts[i])
                x2, y2 = map(int, pts[i + 1])
                cv2.line(img, (x1, y1), (x2, y2), color, 1)

        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    def detect_holes_raw(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        best_score = -np.inf
        best_params = (None, None)
        for b in self.block_sizes:
            for c in self.c_values:
                th = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, b, c)
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                centers = self._filter_valid_holes(cnts)
                if len(centers) > best_score:
                    best_score = len(centers)
                    best_params = (b, c)
        b, c = best_params
        th = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, b, c)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        return self._filter_valid_holes(cnts)

    

    # 이하 cluster_rows, cluster_columns, get_board_nets, find_optimal_y, get_row_nets 등 unchanged

    def _group_by_row(self, points, row_count=17):
        # y 기준으로 정렬 후 그룹화
        sorted_pts = sorted(points, key=lambda p: p[1])
        rows = [sorted_pts[i*50:(i+1)*50] for i in range(row_count)]
        return rows

    def group_rows(self, holes, y_thresh):
        holes_sorted = sorted(holes, key=lambda p: p[1])
        groups = []
        for p in holes_sorted:
            if not groups:
                groups.append([p])
            else:
                med_y = np.median([q[1] for q in groups[-1]])
                if abs(p[1] - med_y) < y_thresh:
                    groups[-1].append(p)
                else:
                    groups.append([p])
        out = []
        for i, g in enumerate(groups):
            pts = sorted(g, key=lambda p: p[0])
            out.append((i, pts, np.median([p[1] for p in pts])))
        return out

    # 이하 클러스터링 로직 동일 적용
    def cluster_columns(self, holes, x_eps=None):
        xs = np.array([[x] for x,y in holes])
        if x_eps is None:
            diffs = np.diff(np.sort(xs.flatten()))
            med = float(np.median(diffs)) if diffs.size>0 else 0.0
            x_eps = med*0.5 if med>0 else 1.0
        labels = DBSCAN(eps=x_eps, min_samples=1).fit_predict(xs)
        clusters = {}
        for p,lbl in zip(holes, labels):
            clusters.setdefault(lbl, []).append(p)
        return [clusters[k] for k in sorted(clusters, key=lambda k: np.mean([pt[0] for pt in clusters[k]]))]

    def _cluster_by_pattern(self, pts, pattern):
        pts_sorted = sorted(pts, key=lambda p: p[0])
        total = sum(pattern)
        if len(pts_sorted) != total:
            xs = [p[0] for p in pts_sorted]
            xm, xM = min(xs), max(xs)
            ymed = int(np.median([p[1] for p in pts_sorted]))
            pts_sorted = [(float(x), ymed) for x in np.linspace(xm, xM, total)]
        clusters = []
        idx = 0
        for sz in pattern:
            clusters.append(pts_sorted[idx:idx+sz])
            idx += sz
        return clusters

    


    def find_optimal_y(self, holes):
        best = (None, None, float('inf'))
        for th in np.arange(self.y_threshold_range[0], self.y_threshold_range[1] + self.y_step, self.y_step):
            gr = self.group_rows(holes, th)
            cnt = len(gr)
            dev = sum(abs(len(g[1]) - 50) for g in gr) / cnt if cnt else np.inf
            score = abs(cnt - self.target_rows) + dev
            if score < best[2]:
                best = (th, gr, score)
        return best

    def get_row_nets(self, holes, y_thresh=None):
        if y_thresh is None:
            y_thresh, grp, _ = self.find_optimal_y(holes)
        else:
            grp = self.group_rows(holes, y_thresh)
        pattern = [2,5,5,2,5,5,2,5,5,2,5,5,2]
        result = []
        for i, pts, _ in grp:
            pat = [5]*10 if i < 4 else pattern
            result.append((i, self._cluster_by_pattern(pts, pat)))
        return result

    def remove_spatial_outliers(holes, eps=5.0, min_samples=3):
        # holes: List[(x,y)]
        pts = np.array(holes)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
        # noise 레이블==-1 제거
        filtered = pts[labels != -1]
        return [tuple(p) for p in filtered]
    
    def remove_far_outliers(holes, z_thresh=2.0):
        pts = np.array(holes)
        nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
        dists, _ = nbrs.kneighbors(pts)
        # dists[:,1] 은 첫 번째 이웃(자기자신 제외)까지 거리
        dist1 = dists[:,1]
        mean, std = dist1.mean(), dist1.std()
        mask = dist1 < (mean + z_thresh*std)
        return [tuple(p) for p,keep in zip(pts,mask) if keep]
    
    
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

    