# hole_detector.py
import cv2
import numpy as np

class HoleDetector:
    """
    Breadboard 구멍 검출 클래스로, adaptive threshold + contour 필터링,
    행 그룹화 및 보간 기능 제공
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
                 target_rows=67):
        self.block_sizes = block_sizes
        self.c_values = c_values
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.circ_threshold = circ_threshold
        self.y_threshold_range = y_threshold_range
        self.y_step = y_step
        self.target_rows = target_rows

    def _filter_valid_holes(self, contours):
        centers=[]
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area<=self.min_area or area>self.max_area: continue
            x,y,w,h=cv2.boundingRect(cnt)
            ar=float(w)/h
            if ar<self.aspect_ratio_range[0] or ar>self.aspect_ratio_range[1]: continue
            peri=cv2.arcLength(cnt,True)
            if peri==0: continue
            circ=4*np.pi*area/(peri**2)
            if circ<self.circ_threshold: continue
            M=cv2.moments(cnt)
            if M['m00']==0: continue
            cx=int(M['m10']/M['m00']); cy=int(M['m01']/M['m00'])
            centers.append((cx,cy))
        return centers

    def detect_holes(self, image, debug=False):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        gray=clahe.apply(gray)
        # 자동 파라미터 탐색
        best_score=-np.inf; best_params=(None,None)
        blurred=cv2.GaussianBlur(gray,(5,5),0)
        for b in self.block_sizes:
            for c in self.c_values:
                th=cv2.adaptiveThreshold(blurred,255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV,
                                          b,c)
                cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                centers=self._filter_valid_holes(cnts)
                valid=len(centers)
                score=valid  # 간단: 검출 개수 최대화
                if score>best_score:
                    best_score=score; best_params=(b,c)
        # 최적 파라미터 적용
        b,c=best_params
        th=cv2.adaptiveThreshold(blurred,255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV,
                                  b,c)
        cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        holes=self._filter_valid_holes(cnts)
        return holes

    def group_rows(self, holes, y_thresh):
        sorted_h=sorted(holes,key=lambda p:p[1])
        groups=[]
        for pt in sorted_h:
            if not groups:
                groups.append([pt])
            else:
                med=np.median([p[1] for p in groups[-1]])
                if abs(pt[1]-med)<y_thresh:
                    groups[-1].append(pt)
                else:
                    groups.append([pt])
        numbered=[]
        for i,g in enumerate(groups):
            pts=sorted(g,key=lambda p:p[0])
            med_y=np.median([p[1] for p in pts])
            numbered.append((i,pts,med_y))
        return numbered

    def find_optimal_y(self, holes):
        best=(None,None,float('inf'))
        for th in np.arange(self.y_threshold_range[0],self.y_threshold_range[1]+self.y_step,self.y_step):
            gr=self.group_rows(holes,th)
            row_cnt=len(gr)
            dev=sum(abs(len(g[1])-50) for g in gr)/row_cnt if row_cnt else np.inf
            score=abs(row_cnt-self.target_rows)+dev
            if score<best[2]: best=(th,gr,score)
        return best

    def interpolate_rows(self, groups, min_req=20):
        new=[]
        for idx,pts,med in groups:
            tgt=50 if 5<=idx<=64 else (40 if idx<5 else 40)
            if len(pts)>=min_req and len(pts)<tgt:
                xs=[p[0] for p in pts]; x0,x1=min(xs),max(xs)
                interp=[(float(x),med) for x in np.linspace(x0,x1,tgt)]
                new.append((idx,interp,med,True))
            else:
                new.append((idx,pts,med,False))
        return new
    
    def cluster_by_pattern(self, row_pts, pattern):
        """
        row_pts: list of (x,y) 좌표
        pattern: [int,...] 각 넷이 이 행에서 차지하는 홀 개수의 순서 리스트 (합이 예상 홀 개수)
        반환: pattern 길이만큼의 클러스터 리스트 [[(x,y),...], …]
        """
        # 1) x 오름차순 정렬
        pts = sorted(row_pts, key=lambda p: p[0])
        expected = sum(pattern)
        # 2) 검출 홀 개수가 예상과 다르면 균등 보간
        if len(pts) != expected:
            xs = [p[0] for p in pts]
            x0, x1 = min(xs), max(xs)
            y_med = int(np.median([p[1] for p in pts]))
            pts = [(float(x), y_med) for x in np.linspace(x0, x1, expected)]
        # 3) 패턴대로 분할
        clusters = []
        idx = 0
        for size in pattern:
            clusters.append(pts[idx:idx+size])
            idx += size
        return clusters

    def get_row_nets(self, holes, y_thresh=None):
        """
        holes: [(x,y),...]
        y_thresh: 행 구분 임계치 (None이면 find_optimal_y 사용)
        반환: [(row_idx, [[(x,y),...],…]), …] 각 행별 넷 클러스터
        """
        # 1) 행 그룹화
        if y_thresh is None:
            y_thresh, gr, _ = self.find_optimal_y(holes)
        else:
            gr = self.group_rows(holes, y_thresh)
        # 2) 각 행별 패턴 적용
        pattern13 = [2,5,5,2,5,5,2,5,5,2,5,5,2]
        row_nets = []
        for row_idx, pts, _ in gr:
            # 맨 위 4행은 5개씩 → 10개 넷
            if row_idx < 4:
                pattern = [5] * 10
            else:
                pattern = pattern13
            clusters = self.cluster_by_pattern(pts, pattern)
            row_nets.append((row_idx, clusters))
        # 정렬된 순서로 반환
        return sorted(row_nets, key=lambda x: x[0])

