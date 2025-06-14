import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import math
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionParams:
    """검출 파라미터를 관리하는 클래스"""
    min_length: int = 20
    hull_proximity: int = 20
    merge_angle: float = 0.174  # 10도 in radians
    merge_distance: int = 25
    remove_holes: bool = True
    clahe_clip: float = 2.0
    adaptive_block_size: int = 15
    adaptive_c: int = 4
    gamma: float = 1.2
    use_gamma: bool = False
    use_spur: bool = False
    use_ransac: bool = False
    use_hough: bool = False
    use_intensity: bool = False
    use_ridge: bool = False
    use_score_filter: bool = False
    use_endpoint_refine: bool = False
    # 새로운 파라미터들
    gradient_threshold: float = 0.3
    contour_area_threshold: int = 500
    endpoint_confidence_threshold: float = 0.5

class LEDEndpointDetector:
    """LED 끝점 검출기 클래스"""
    
    def __init__(self, params: DetectionParams = None):
        self.params = params or DetectionParams()
        self.debug_images = {}
        
    def ransac_extrapolate_endpoints(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
        """RANSAC을 이용한 끝점 외삽"""
        if len(lines) < 2:
            return []
        
        try:
            # 모든 끝점 수집
            pts = np.array([(x1, y1) for x1, y1, x2, y2 in lines] + 
                          [(x2, y2) for x1, y1, x2, y2 in lines])
            X = pts[:, 0].reshape(-1, 1)
            y = pts[:, 1]
            
            model = RANSACRegressor(random_state=0, max_trials=100)
            model.fit(X, y)
            
            xs = np.array([[X.min()], [X.max()]])
            ys = model.predict(xs)
            
            return [(int(xs[0, 0]), int(ys[0])), (int(xs[1, 0]), int(ys[1]))]
        
        except Exception as e:
            logger.error(f"RANSAC 외삽 중 오류: {e}")
            return []

    def remove_spurs(self, skel_img: np.ndarray, length: int = 10) -> np.ndarray:
        """스켈레톤 스퍼 제거"""
        try:
            clean = remove_small_objects(skel_img.astype(bool), min_size=length)
            return (clean.astype(np.uint8) * 255)
        except Exception as e:
            logger.error(f"스퍼 제거 중 오류: {e}")
            return skel_img

    def verify_intensity_endpoint(self, pt: Tuple[int, int], gray_img: np.ndarray, window: int = 5) -> bool:
        """1D 강도 프로파일을 이용한 끝점 검증"""
        try:
            x, y = pt
            h, w = gray_img.shape
            
            vals = []
            for dx in range(-window, window + 1):
                xx = min(max(x + dx, 0), w - 1)
                vals.append(gray_img[y, xx])
            
            grad = np.abs(np.diff(vals))
            return grad.max() >= 10
        
        except Exception as e:
            logger.error(f"강도 검증 중 오류: {e}")
            return False

    def subpixel_refine(self, pt: Tuple[int, int], grad_mag: np.ndarray) -> Tuple[float, int]:
        """서브픽셀 정제"""
        try:
            x, y = pt
            
            if x <= 0 or x >= grad_mag.shape[1] - 1:
                return pt
            
            g = grad_mag[y, x-1:x+2]
            denom = 2 * (g[0] - 2 * g[1] + g[2])
            
            if denom == 0:
                return pt
            
            offset = (g[0] - g[2]) / denom
            return (x + offset, y)
        
        except Exception as e:
            logger.error(f"서브픽셀 정제 중 오류: {e}")
            return pt

    def apply_gamma_correction(self, img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """감마 보정"""
        try:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img, table)
        except Exception as e:
            logger.error(f"감마 보정 중 오류: {e}")
            return img

    def detect_hough_lines(self, gray_img: np.ndarray, min_length: int = 20) -> List[Tuple[int, int, int, int]]:
        """Canny + Hough를 이용한 선 검출"""
        try:
            edges = cv2.Canny(gray_img, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=min_length, maxLineGap=5)
            if lines is None:
                return []
            return [tuple(l[0]) for l in lines]
        except Exception as e:
            logger.error(f"Hough 선 검출 중 오류: {e}")
            return []

    def detect_lines_from_ridge(self, gray_img: np.ndarray, min_length: int = 20) -> List[Tuple[int, int, int, int]]:
        """Ridge 검출을 이용한 선 검출"""
        try:
            # Hessian 행렬 계산
            hxx, hxy, hyy = hessian_matrix(gray_img, sigma=1.5, use_gaussian_derivatives=False)
            H_elems = np.stack((hxx, hxy, hyy), axis=0)
            eigs = hessian_matrix_eigvals(H_elems)
            i1, i2 = eigs[0], eigs[1]

            # Ridge 마스크 생성
            thresh = np.mean(np.abs(i1)) + 1.5 * np.std(np.abs(i1))
            ridge_img = (np.abs(i1) > thresh).astype(np.uint8) * 255

            # 스켈레톤화
            skel = skeletonize(ridge_img // 255).astype(np.uint8) * 255

            # LSD를 이용한 선 검출
            lsd = cv2.createLineSegmentDetector(0)
            lines, _, _, _ = lsd.detect(skel)
            if lines is None:
                return []

            # 최소 길이로 필터링
            filtered = []
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                if np.hypot(x2 - x1, y2 - y1) >= min_length:
                    filtered.append((x1, y1, x2, y2))
            
            return filtered
        
        except Exception as e:
            logger.error(f"Ridge 검출 중 오류: {e}")
            return []

    def score_lead_candidate(self, line: Tuple[int, int, int, int], 
                           hull: np.ndarray, centroid: Tuple[int, int]) -> float:
        """리드 후보 점수 계산"""
        try:
            x1, y1, x2, y2 = line

            # 근접성 점수
            dist1 = cv2.pointPolygonTest(hull, (float(x1), float(y1)), True)
            dist2 = cv2.pointPolygonTest(hull, (float(x2), float(y2)), True)
            inner_dist = min(abs(dist1), abs(dist2))
            outer_dist = max(abs(dist1), abs(dist2))
            score_proximity = (1 / (inner_dist + 1)) * outer_dist
            
            # 방향성 점수
            inner_pt, outer_pt = ((x1, y1), (x2, y2)) if dist1 > dist2 else ((x2, y2), (x1, y1))
            vec_out = (outer_pt[0] - centroid[0], outer_pt[1] - centroid[1])
            vec_line = (outer_pt[0] - inner_pt[0], outer_pt[1] - inner_pt[1])
            score_direction = max(np.dot(vec_out, vec_line), 0)
            
            # 가중 합계
            return score_proximity * 0.5 + score_direction * 1.5
        
        except Exception as e:
            logger.error(f"점수 계산 중 오류: {e}")
            return 0.0

    def select_best_pair(self, scored_lines: List[Tuple[Tuple[int, int, int, int], float]], 
                        min_distance: int = 10) -> List[Tuple[int, int, int, int]]:
        """최적의 선 쌍 선택"""
        try:
            lines = [l for l, s in scored_lines]
            if len(lines) < 2:
                return lines
            
            # 충분히 떨어진 첫 번째 쌍 찾기
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    l1, l2 = lines[i], lines[j]
                    m1 = ((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2)
                    m2 = ((l2[0] + l2[2]) / 2, (l2[1] + l2[3]) / 2)
                    if np.hypot(m1[0] - m2[0], m1[1] - m2[1]) > min_distance:
                        return [l1, l2]
            
            return lines[:2]
        
        except Exception as e:
            logger.error(f"최적 쌍 선택 중 오류: {e}")
            return []

    def refine_endpoint(self, endpoint: Tuple[int, int], gray_img: np.ndarray, 
                       search_radius: int = 5) -> Tuple[int, int]:
        """끝점 정제"""
        try:
            x, y = endpoint
            grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            
            max_val = -1
            best = endpoint
            h, w = gray_img.shape
            
            for yy in range(max(0, y - search_radius), min(h, y + search_radius)):
                for xx in range(max(0, x - search_radius), min(w, x + search_radius)):
                    if grad_mag[yy, xx] > max_val:
                        max_val = grad_mag[yy, xx]
                        best = (xx, yy)
            
            return best
        
        except Exception as e:
            logger.error(f"끝점 정제 중 오류: {e}")
            return endpoint

    def remove_breadboard_holes(self, img: np.ndarray, kernel_size: int = 5, 
                               iterations: int = 2) -> np.ndarray:
        """브레드보드 구멍 제거"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                     param1=50, param2=30, minRadius=3, maxRadius=15)
            
            mask = np.ones_like(gray, dtype=np.uint8) * 255
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(mask, (i[0], i[1]), i[2] + 5, 0, -1)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            return cv2.bitwise_and(img, img, mask=mask)
        
        except Exception as e:
            logger.error(f"브레드보드 구멍 제거 중 오류: {e}")
            return img

    def apply_clahe(self, img_gray: np.ndarray, clip_limit: float = 2.0, 
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """CLAHE 적용"""
        try:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(img_gray)
        except Exception as e:
            logger.error(f"CLAHE 적용 중 오류: {e}")
            return img_gray

    def remove_color_regions(self, img: np.ndarray, remove_red: bool = True, 
                           remove_blue: bool = True) -> np.ndarray:
        """색상 영역 제거"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            
            if remove_red:
                mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_or(mask1, mask2)))
            
            if remove_blue:
                blue_mask = cv2.inRange(hsv, (100, 150, 50), (140, 255, 255))
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(blue_mask))
            
            return cv2.bitwise_and(img, img, mask=mask)
        
        except Exception as e:
            logger.error(f"색상 영역 제거 중 오류: {e}")
            return img

    def detect_led_body_advanced(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """향상된 LED 본체 검출"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 다양한 LED 색상 마스크
            red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
            blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
            
            # 모든 마스크 결합
            body_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask1, red_mask2), 
                                      cv2.bitwise_or(cv2.bitwise_or(green_mask, yellow_mask), blue_mask))
            
            # 모폴로지 연산
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # 윤곽선 검출
            contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
            
            # 면적 기준 필터링
            valid_contours = [c for c in contours if cv2.contourArea(c) > self.params.contour_area_threshold]
            
            if not valid_contours:
                # 임계값을 낮춰 재시도
                valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
                if not valid_contours:
                    return None, None
            
            # 가장 큰 윤곽선 선택
            largest_contour = max(valid_contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            
            # 중심점 계산
            M = cv2.moments(hull)
            if M["m00"] != 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                centroid = None
            
            self.debug_images['led_body_mask'] = body_mask
            return hull, centroid
        
        except Exception as e:
            logger.error(f"LED 본체 검출 중 오류: {e}")
            return None, None

    def detect_lines_lsd(self, img_gray: np.ndarray, min_length: int = 30) -> List[Tuple[int, int, int, int]]:
        """LSD를 이용한 선 검출"""
        try:
            lsd = cv2.createLineSegmentDetector(0)
            lines, _, _, _ = lsd.detect(img_gray)
            if lines is None:
                return []
            
            result = []
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                if np.hypot(x2 - x1, y2 - y1) >= min_length:
                    result.append((x1, y1, x2, y2))
            
            return result
        
        except Exception as e:
            logger.error(f"LSD 선 검출 중 오류: {e}")
            return []

    def filter_common_lines(self, lines1: List[Tuple[int, int, int, int]], 
                           lines2: List[Tuple[int, int, int, int]], 
                           angle_thresh: float, dist_thresh: float) -> List[Tuple[int, int, int, int]]:
        """공통 선 필터링"""
        try:
            common = []
            for x1, y1, x2, y2 in lines1:
                theta1 = math.atan2(y2 - y1, x2 - x1)
                mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                for a1, b1, a2, b2 in lines2:
                    theta2 = math.atan2(b2 - b1, a2 - a1)
                    mid2 = ((a1 + a2) / 2, (b1 + b2) / 2)
                    
                    if (abs(theta1 - theta2) < angle_thresh and 
                        math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1]) < dist_thresh):
                        common.append((int((x1+a1)/2), int((y1+b1)/2), 
                                     int((x2+a2)/2), int((y2+b2)/2)))
                        break
            
            return common
        
        except Exception as e:
            logger.error(f"공통 선 필터링 중 오류: {e}")
            return []

    def merge_similar_lines(self, lines: List[Tuple[int, int, int, int]], 
                           angle_thresh: float, dist_thresh: float) -> List[Tuple[int, int, int, int]]:
        """유사한 선들 병합"""
        try:
            merged = []
            for line in lines:
                matched = False
                for i, merged_line in enumerate(merged):
                    angle1 = math.atan2(line[3] - line[1], line[2] - line[0])
                    angle2 = math.atan2(merged_line[3] - merged_line[1], merged_line[2] - merged_line[0])
                    mid1 = ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2)
                    mid2 = ((merged_line[0] + merged_line[2]) / 2, (merged_line[1] + merged_line[3]) / 2)
                    
                    if (abs(angle1 - angle2) < angle_thresh and 
                        math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1]) < dist_thresh):
                        
                        points = [line[:2], line[2:], merged_line[:2], merged_line[2:]]
                        farthest_pair = max([(p1, p2) for p1 in points for p2 in points], 
                                          key=lambda pair: math.hypot(pair[0][0] - pair[1][0], 
                                                                     pair[0][1] - pair[1][1]))
                        merged[i] = (farthest_pair[0][0], farthest_pair[0][1], 
                                   farthest_pair[1][0], farthest_pair[1][1])
                        matched = True
                        break
                
                if not matched:
                    merged.append(line)
            
            return merged
        
        except Exception as e:
            logger.error(f"선 병합 중 오류: {e}")
            return lines

    def select_lead_lines(self, lines: List[Tuple[int, int, int, int]], 
                         hull: np.ndarray, centroid: Tuple[int, int], 
                         proximity_thresh: float) -> List[Tuple[int, int, int, int]]:
        """리드 선 선택 - 최종적으로 2개의 LED 리드 선분 검출"""
        try:
            if hull is None or centroid is None or not lines:
                return []
            
            # 조건을 만족하는 후보 리드 선들 찾기
            candidate_leads = []
            for x1, y1, x2, y2 in lines:
                pt1, pt2 = (float(x1), float(y1)), (float(x2), float(y2))
                dist1 = cv2.pointPolygonTest(hull, pt1, True)
                dist2 = cv2.pointPolygonTest(hull, pt2, True)
                
                # hull로부터의 절댓값 거리 계산
                abs_dist1 = abs(dist1)
                abs_dist2 = abs(dist2)
                
                # 한 점은 hull proximity 범위 안에, 한 점은 범위 밖에
                cond1 = (abs_dist1 <= proximity_thresh and abs_dist2 > proximity_thresh)
                cond2 = (abs_dist2 <= proximity_thresh and abs_dist1 > proximity_thresh)
                
                if cond1 or cond2:
                    # 더 가까운 점을 inner_point, 더 먼 점을 outer_point로 설정
                    inner_point, outer_point = (pt1, pt2) if abs_dist1 < abs_dist2 else (pt2, pt1)
                    
                    # 방향성 확인 - outer_point가 centroid에서 바깥쪽으로 향하는지
                    vec_out = (outer_point[0] - centroid[0], outer_point[1] - centroid[1])
                    vec_line = (outer_point[0] - inner_point[0], outer_point[1] - inner_point[1])
                    
                    if np.dot(vec_out, vec_line) > 0:
                        # 점수 계산 (품질 평가)
                        score = self.score_lead_candidate((x1, y1, x2, y2), hull, centroid)
                        candidate_leads.append(((x1, y1, x2, y2), score))
            
            if not candidate_leads:
                logger.warning("조건을 만족하는 리드 선이 없습니다.")
                return []
            
            # 점수 순으로 정렬 (높은 점수부터)
            candidate_leads.sort(key=lambda x: x[1], reverse=True)
            
            # 최대 2개의 리드 선 선택 (서로 충분히 떨어져 있는 것들)
            selected_leads = []
            min_distance = 30  # 최소 거리 임계값
            
            for candidate_line, score in candidate_leads:
                if len(selected_leads) >= 2:
                    break
                
                # 이미 선택된 선들과 충분히 떨어져 있는지 확인
                is_far_enough = True
                for selected_line in selected_leads:
                    # 두 선의 중점 간 거리 계산
                    mid1 = ((candidate_line[0] + candidate_line[2]) / 2, 
                           (candidate_line[1] + candidate_line[3]) / 2)
                    mid2 = ((selected_line[0] + selected_line[2]) / 2, 
                           (selected_line[1] + selected_line[3]) / 2)
                    
                    distance = np.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1])
                    if distance < min_distance:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    selected_leads.append(candidate_line)
            
            logger.info(f"후보 리드 선: {len(candidate_leads)}개, 최종 선택: {len(selected_leads)}개")
            return selected_leads
        
        except Exception as e:
            logger.error(f"리드 선 선택 중 오류: {e}")
            return []

    def get_final_endpoints(self, leads: List[Tuple[int, int, int, int]], 
                           hull: np.ndarray, proximity_thresh: float) -> List[Tuple[int, int]]:
        """최종 끝점 추출 - 각 리드 선에서 hull proximity 바깥에 있는 점 1개씩 선택"""
        try:
            if not leads or hull is None:
                return []
            
            final_endpoints = []
            
            for i, (x1, y1, x2, y2) in enumerate(leads):
                pt1, pt2 = (x1, y1), (x2, y2)
                
                # 각 점의 hull로부터의 거리 계산
                dist1 = cv2.pointPolygonTest(hull, (float(x1), float(y1)), True)
                dist2 = cv2.pointPolygonTest(hull, (float(x2), float(y2)), True)
                abs_dist1 = abs(dist1)
                abs_dist2 = abs(dist2)
                
                # hull proximity 바깥에 있는 점 찾기
                outer_point = None
                
                if abs_dist1 > proximity_thresh and abs_dist2 <= proximity_thresh:
                    # pt1이 바깥에 있음
                    outer_point = pt1
                elif abs_dist2 > proximity_thresh and abs_dist1 <= proximity_thresh:
                    # pt2가 바깥에 있음
                    outer_point = pt2
                elif abs_dist1 > proximity_thresh and abs_dist2 > proximity_thresh:
                    # 둘 다 바깥에 있는 경우 더 먼 점 선택
                    outer_point = pt1 if abs_dist1 > abs_dist2 else pt2
                else:
                    # 예외 상황: 둘 다 안에 있는 경우 (이론적으로 발생하지 않아야 함)
                    logger.warning(f"리드 선 {i}: 두 점 모두 proximity 범위 안에 있음")
                    outer_point = pt1 if abs_dist1 > abs_dist2 else pt2
                
                if outer_point:
                    final_endpoints.append(outer_point)
                    logger.debug(f"리드 선 {i}: 끝점 {outer_point} 선택 (거리: {abs(cv2.pointPolygonTest(hull, (float(outer_point[0]), float(outer_point[1])), True)):.1f})")
            
            logger.info(f"최종 끝점 {len(final_endpoints)}개 추출됨")
            return final_endpoints
        
        except Exception as e:
            logger.error(f"끝점 추출 중 오류: {e}")
            return []

    def cluster_endpoints(self, endpoints: List[Tuple[int, int]], 
                         eps: float = 15.0, min_samples: int = 1) -> List[Tuple[int, int]]:
        """DBSCAN을 이용한 끝점 클러스터링"""
        if len(endpoints) < 2:
            return endpoints
        
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(endpoints)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            clustered_endpoints = []
            
            for label in unique_labels:
                if label == -1:  # 노이즈 포인트
                    continue
                
                cluster_points = [endpoints[i] for i in range(len(endpoints)) if labels[i] == label]
                
                # 클러스터 중심점 계산
                center_x = int(np.mean([p[0] for p in cluster_points]))
                center_y = int(np.mean([p[1] for p in cluster_points]))
                clustered_endpoints.append((center_x, center_y))
            
            return clustered_endpoints if clustered_endpoints else endpoints
        
        except Exception as e:
            logger.error(f"클러스터링 중 오류: {e}")
            return endpoints

    def calculate_endpoint_confidence(self, endpoint: Tuple[int, int], 
                                    gray_img: np.ndarray, 
                                    hull: np.ndarray, proximity_thresh: float = 20) -> float:
        """끝점 신뢰도 계산"""
        try:
            x, y = endpoint
            h, w = gray_img.shape
            
            if x < 5 or y < 5 or x >= w-5 or y >= h-5:
                return 0.0
            
            confidence_scores = []
            
            # 1. 그래디언트 강도
            grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            grad_score = grad_mag[y, x] / 255.0
            confidence_scores.append(grad_score)
            
            # 2. 로컬 대비
            local_patch = gray_img[y-5:y+6, x-5:x+6]
            if local_patch.size > 0:
                contrast_score = np.std(local_patch) / 255.0
                confidence_scores.append(contrast_score)
            
            # 3. Hull proximity 바깥에 있는 정도 (바깥에 있을수록 높은 점수)
            if hull is not None:
                distance = cv2.pointPolygonTest(hull, (float(x), float(y)), True)
                abs_distance = abs(distance)
                
                # proximity_thresh 바깥에 있으면 높은 점수, 안에 있으면 낮은 점수
                if abs_distance > proximity_thresh:
                    # 바깥에 있는 경우: 거리에 비례하여 점수 증가
                    dist_score = min(1.0, (abs_distance - proximity_thresh) / 50.0)
                else:
                    # proximity 범위 안에 있는 경우: 낮은 점수
                    dist_score = 0.1
                
                confidence_scores.append(dist_score)
            
            # 4. 엣지 연결성
            edges = cv2.Canny(gray_img, 50, 150)
            neighborhood = edges[max(0, y-3):y+4, max(0, x-3):x+4]
            edge_count = np.sum(neighborhood > 0)
            edge_score = edge_count / (7 * 7)
            confidence_scores.append(edge_score)
            
            # 가중 평균 (proximity 점수에 더 높은 가중치)
            weights = [0.25, 0.15, 0.4, 0.2]  # proximity에 더 높은 가중치
            final_confidence = sum(w * s for w, s in zip(weights, confidence_scores))
            
            return min(1.0, final_confidence)
        
        except Exception as e:
            logger.error(f"신뢰도 계산 중 오류: {e}")
            return 0.0

    def visualize_lines(self, img: np.ndarray, lines: List[Tuple[int, int, int, int]], 
                       window_name: str, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """선 시각화"""
        vis = img.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(vis, f"{window_name}: {len(lines)} lines", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    def visualize_endpoint_detection(self, img: np.ndarray, 
                                   all_lines: List[Tuple[int, int, int, int]], 
                                   lead_lines: List[Tuple[int, int, int, int]], 
                                   endpoints: List[Tuple[int, int]], 
                                   hull: np.ndarray, centroid: Tuple[int, int],
                                   confidences: List[float] = None) -> np.ndarray:
        """끝점 검출 결과 시각화"""
        vis = img.copy()
        
        # Hull과 중심점 그리기
        if hull is not None:
            cv2.drawContours(vis, [hull], -1, (0, 255, 255), 2)
        if centroid is not None:
            cv2.circle(vis, centroid, 5, (0, 255, 255), -1)
        
        # 모든 선 그리기 (회색)
        for x1, y1, x2, y2 in all_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # 리드 선 그리기 (녹색)
        for x1, y1, x2, y2 in lead_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 끝점 그리기
        for i, (x, y) in enumerate(endpoints):
            # 신뢰도에 따른 색상 결정
            if confidences and i < len(confidences):
                confidence = confidences[i]
                if confidence > 0.7:
                    color = (0, 255, 0)  # 녹색 (높은 신뢰도)
                elif confidence > 0.5:
                    color = (0, 165, 255)  # 주황색 (중간 신뢰도)
                else:
                    color = (0, 0, 255)  # 빨간색 (낮은 신뢰도)
                
                cv2.circle(vis, (x, y), 8, color, -1)
                cv2.putText(vis, f"T{i+1}({confidence:.2f})", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(vis, f"T{i+1}", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis

    def process_image(self, img: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """이미지 처리 메인 함수"""
        # 기본 결과 구조 초기화
        orig = img.copy()
        gray_fallback = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if len(orig.shape) == 3 else orig
        
        results = {
            'endpoints': [],
            'confidences': [],
            'hull': None,
            'centroid': None,
            'debug_images': {
                'original': orig,
                'holes_removed': orig,
                'gamma_corrected': orig,
                'color_removed': orig,
                'gray': gray_fallback,
                'clahe': gray_fallback,
                'adaptive': np.zeros_like(gray_fallback),
                'skeleton': np.zeros_like(gray_fallback)
            },
            'all_lines': [],
            'lead_lines': []
        }
        
        try:
            # 전처리 단계별 수행
            # 구멍 제거
            holes_removed = self.remove_breadboard_holes(orig) if params.get('remove_holes', False) else orig
            
            # 감마 보정
            gamma_corrected = self.apply_gamma_correction(holes_removed, 1.2) if params.get('use_gamma', False) else holes_removed
            
            # 색상 영역 제거
            color_removed = self.remove_color_regions(gamma_corrected)
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(color_removed, cv2.COLOR_BGR2GRAY)
            
            # CLAHE 적용
            clahe_img = self.apply_clahe(gray, clip_limit=params.get('clahe_clip', 2.0))
            
            # 적응형 임계값
            bs = params.get('adaptive_bs', 15)
            if bs % 2 == 0:
                bs += 1
            adaptive = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, bs, params.get('adaptive_c', 4))
            
            # 스켈레톤화
            if params.get('use_spur', False):
                skel = self.remove_spurs(skeletonize(adaptive // 255).astype(np.uint8) * 255)
            else:
                skel = skeletonize(adaptive // 255).astype(np.uint8) * 255
            
            # 디버그 이미지 업데이트
            results['debug_images'].update({
                'original': orig,
                'holes_removed': holes_removed,
                'gamma_corrected': gamma_corrected,
                'color_removed': color_removed,
                'gray': gray,
                'clahe': clahe_img,
                'adaptive': adaptive,
                'skeleton': skel
            })
            
            # LED 본체 검출
            hull, centroid = self.detect_led_body_advanced(orig)
            results['hull'] = hull
            results['centroid'] = centroid
            
            if hull is None:
                logger.warning("LED 본체를 검출할 수 없습니다.")
                return results
            
            # 선 검출 - adaptive threshold 이미지와 원본 이미지에서 각각 LSD 검출
            # 1. adaptive threshold된 이미지에서 LSD 검출
            lines_adaptive = self.detect_lines_lsd(adaptive, params.get('min_length', 20))
            
            # 2. 원본 이미지 (그레이스케일)에서 LSD 검출
            lines_original = self.detect_lines_lsd(gray, params.get('min_length', 20))
            
            # 3. 두 결과를 filter_common_lines로 병합
            merged = self.filter_common_lines(lines_adaptive, lines_original, 
                                            np.deg2rad(params.get('merge_angle', 10)), 
                                            params.get('merge_distance', 25))
            
            # 4. 추가적인 선 병합 (유사한 선들 통합)
            merged = self.merge_similar_lines(merged, 
                                            np.deg2rad(params.get('merge_angle', 10)), 
                                            params.get('merge_distance', 25))
            
            logger.info(f"Adaptive 이미지 선: {len(lines_adaptive)}개, "
                       f"원본 이미지 선: {len(lines_original)}개, "
                       f"병합된 선: {len(merged)}개")
            
            results['all_lines'] = merged
            
            # 리드 선 선택 - 최종적으로 2개의 리드 선 선택
            if params.get('use_score_filter', False):
                # 점수 기반 선택 (이미 select_lead_lines에서 점수 기반으로 선택됨)
                leads = self.select_lead_lines(merged, hull, centroid, params.get('hull_proximity', 20))
            else:
                # 기존 방식 (조건 기반 선택)
                leads = self.select_lead_lines(merged, hull, centroid, params.get('hull_proximity', 20))
            
            # 최대 2개로 제한 (안전장치)
            if len(leads) > 2:
                logger.warning(f"리드 선이 {len(leads)}개 검출됨. 상위 2개만 선택합니다.")
                leads = leads[:2]
            
            results['lead_lines'] = leads
            
            # 끝점 검출
            if params.get('use_ransac', False):
                endpoints = self.ransac_extrapolate_endpoints(leads)
            else:
                endpoints = self.get_final_endpoints(leads, hull, params.get('hull_proximity', 20))
            
            # 정확히 2개의 끝점이 나와야 함
            if len(endpoints) != 2:
                logger.warning(f"예상과 다른 끝점 개수: {len(endpoints)}개 (예상: 2개)")
            
            # 끝점 클러스터링 (2개 끝점일 때만 적용, 너무 가까우면 클러스터링)
            if len(endpoints) == 2:
                dist_between = np.hypot(endpoints[0][0] - endpoints[1][0], 
                                      endpoints[0][1] - endpoints[1][1])
                if dist_between < 15:  # 15픽셀보다 가까우면 클러스터링 적용
                    endpoints = self.cluster_endpoints(endpoints, eps=10.0)
                    logger.info(f"가까운 끝점들을 클러스터링했습니다. 결과: {len(endpoints)}개")
            elif len(endpoints) > 2:
                # 2개보다 많으면 클러스터링으로 줄이기
                endpoints = self.cluster_endpoints(endpoints, eps=15.0)
                if len(endpoints) > 2:
                    # 그래도 많으면 상위 2개만 선택 (hull에서 더 먼 순서로)
                    endpoint_distances = []
                    for ep in endpoints:
                        dist = abs(cv2.pointPolygonTest(hull, (float(ep[0]), float(ep[1])), True))
                        endpoint_distances.append((ep, dist))
                    endpoint_distances.sort(key=lambda x: x[1], reverse=True)
                    endpoints = [ep for ep, _ in endpoint_distances[:2]]
                    logger.info(f"끝점을 2개로 제한했습니다.")
            
            # 강도 검증
            if params.get('use_intensity', False):
                endpoints = [pt for pt in endpoints if self.verify_intensity_endpoint(pt, gray)]
            
            # 끝점 정제
            if params.get('use_endpoint_refine', False):
                endpoints = [self.refine_endpoint(pt, gray) for pt in endpoints]
            
            # 신뢰도 계산
            confidences = [self.calculate_endpoint_confidence(pt, gray, hull, params.get('hull_proximity', 20)) 
                          for pt in endpoints]
            
            # 신뢰도 기준 필터링
            validated_endpoints = []
            validated_confidences = []
            for pt, conf in zip(endpoints, confidences):
                if conf > self.params.endpoint_confidence_threshold:
                    validated_endpoints.append(pt)
                    validated_confidences.append(conf)
            
            results['endpoints'] = validated_endpoints
            results['confidences'] = validated_confidences
            
            return results
        
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {e}")
            return results


def main():
    """메인 실행 함수"""
    # 이미지 파일 찾기
    images = [f for f in os.listdir('.') if 'led' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print('LED 이미지를 찾을 수 없습니다.')
        return
    
    idx = 0
    img = cv2.imread(images[idx])
    
    # 검출기 초기화
    detector = LEDEndpointDetector()
    
    # 컨트롤 윈도우들 생성
    cv2.namedWindow('LED Detection Controller', cv2.WINDOW_NORMAL)
    stages = ['Original', 'Holes Removed', 'Gamma Corrected', 'Color Removed', 
              'Gray', 'CLAHE', 'Adaptive Threshold', 'Skeleton', 'Final Detection']
    
    for stage in stages:
        cv2.namedWindow(stage, cv2.WINDOW_NORMAL)
    
    # 트랙바 생성
    params = [
        ('Min Length', 0, 100, 20),
        ('Hull Proximity', 0, 100, 20),
        ('Merge Angle', 0, 90, 10),
        ('Merge Distance', 0, 100, 25),
        ('Remove Holes', 0, 1, 1),
        ('CLAHE Clip x10', 0, 100, 20),
        ('Adaptive BS', 3, 51, 15),
        ('Adaptive C', 0, 10, 4),
        ('Gamma Corr', 0, 1, 0),
        ('Spur Rem', 0, 1, 0),
        ('RANSAC', 0, 1, 0),
        ('Hough', 0, 1, 0),
        ('Intensity', 0, 1, 0),
        ('Ridge Det', 0, 1, 0),
        ('Score Filter', 0, 1, 0),
        ('Endpoint Refine', 0, 1, 0)
    ]
    
    for name, lo, hi, df in params:
        cv2.createTrackbar(name, 'LED Detection Controller', df, hi, lambda x: None)
    
    while True:
        try:
            # 트랙바 값 읽기
            vals = {name: cv2.getTrackbarPos(name, 'LED Detection Controller') for name, *_ in params}
            
            # 파라미터 구성
            process_params = {
                'min_length': vals['Min Length'],
                'hull_proximity': vals['Hull Proximity'],
                'merge_angle': vals['Merge Angle'],
                'merge_distance': vals['Merge Distance'],
                'remove_holes': vals['Remove Holes'] == 1,
                'clahe_clip': vals['CLAHE Clip x10'] / 10.0,
                'adaptive_bs': vals['Adaptive BS'] if vals['Adaptive BS'] % 2 == 1 else vals['Adaptive BS'] + 1,
                'adaptive_c': vals['Adaptive C'],
                'use_gamma': vals['Gamma Corr'] == 1,
                'use_spur': vals['Spur Rem'] == 1,
                'use_ransac': vals['RANSAC'] == 1,
                'use_hough': vals['Hough'] == 1,
                'use_intensity': vals['Intensity'] == 1,
                'use_ridge': vals['Ridge Det'] == 1,
                'use_score_filter': vals['Score Filter'] == 1,
                'use_endpoint_refine': vals['Endpoint Refine'] == 1
            }
            
            # 이미지 처리
            results = detector.process_image(img, process_params)
            
            # 전처리 단계별 이미지 표시 (안전한 접근)
            debug_images = results.get('debug_images', {})
            
            # 각 이미지를 안전하게 표시
            for stage, key in [('Original', 'original'), ('Holes Removed', 'holes_removed'), 
                              ('Gamma Corrected', 'gamma_corrected'), ('Color Removed', 'color_removed'),
                              ('Gray', 'gray'), ('CLAHE', 'clahe'), 
                              ('Adaptive Threshold', 'adaptive'), ('Skeleton', 'skeleton')]:
                if key in debug_images:
                    cv2.imshow(stage, debug_images[key])
                else:
                    # 기본 이미지 표시
                    cv2.imshow(stage, img if key == 'original' else np.zeros_like(img))
            
            # 최종 결과 시각화
            vis = detector.visualize_endpoint_detection(
                img, results['all_lines'], results['lead_lines'], 
                results['endpoints'], results['hull'], results['centroid'],
                results['confidences']
            )
            
            # Hull proximity 영역 시각화 (hull 내부/외부 모두 proximity_thresh 거리 내)
            if results['hull'] is not None and vals['Hull Proximity'] > 0:
                gray_img = debug_images.get('gray')
                if gray_img is not None:
                    h, w = gray_img.shape
                    
                    # Hull 영역 마스크 생성
                    hull_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(hull_mask, [results['hull']], -1, 255, -1)
                    
                    # Hull 경계에서 proximity_thresh 거리 내의 영역 계산
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (2 * vals['Hull Proximity'] + 1, 
                                                      2 * vals['Hull Proximity'] + 1))
                    
                    # Hull 내부 영역 축소 (erosion)
                    eroded = cv2.erode(hull_mask, kernel)
                    inner_band = cv2.subtract(hull_mask, eroded)
                    
                    # Hull 외부 영역 확장 (dilation)
                    dilated = cv2.dilate(hull_mask, kernel)
                    outer_band = cv2.subtract(dilated, hull_mask)
                    
                    # 전체 proximity 영역 (내부 + 외부 띠)
                    proximity_band = cv2.bitwise_or(inner_band, outer_band)
                    
                    # 시각화에 반영
                    overlay = vis.copy()
                    overlay[proximity_band == 255] = (255, 0, 0)  # 빨간색으로 표시
                    vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
            
            cv2.imshow('Final Detection', vis)
            
            # 결과 출력 (안전한 접근)
            endpoint_count = len(results.get('endpoints', []))
            confidences = results.get('confidences', [])
            lead_count = len(results.get('lead_lines', []))
            merged_count = len(results.get('all_lines', []))
            
            # 상태 표시 (2개의 리드와 2개의 끝점이 이상적)
            status = "✓" if lead_count == 2 and endpoint_count == 2 else "⚠"
            
            print(f"\r{status} 선:{merged_count} 리드:{lead_count} 끝점:{endpoint_count}", end='')
            if confidences:
                avg_confidence = np.mean(confidences)
                print(f" 신뢰도:{avg_confidence:.2f}", end='')
            print("    ", end='')  # 이전 출력 지우기
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                idx = (idx + 1) % len(images)
                img = cv2.imread(images[idx])
                print(f"\n이미지 변경: {images[idx]}")
            elif key == ord('p'):
                idx = (idx - 1) % len(images)
                img = cv2.imread(images[idx])
                print(f"\n이미지 변경: {images[idx]}")
            elif key == ord('s'):
                # 결과 저장
                result_path = f"{images[idx].split('.')[0]}_result.jpg"
                cv2.imwrite(result_path, vis)
                print(f"\n결과 저장됨: {result_path}")
        
        except Exception as e:
            logger.error(f"메인 루프에서 오류 발생: {e}")
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()