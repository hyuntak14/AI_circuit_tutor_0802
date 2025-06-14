import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
import math
import os
import itertools

def remove_red_blue_lines(img):
    """
    원본 BGR 이미지에서 빨간색과 파란색 선을 마스크하고, 주변 픽셀 색상으로 인페인팅하여 제거합니다.
    img: BGR 컬러 이미지 (np.ndarray)
    반환: 선이 제거된 이미지
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 빨간색 범위 (두 구간)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    # 파란색 범위
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # 합친 마스크
    mask = cv2.bitwise_or(mask_red, mask_blue)
    # 라인 너비 보강
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    # 인페인팅으로 제거
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def detect_lines_ransac(img, threshold, max_trials):
    """
    RANSAC으로 이미지 상의 주요 직선을 하나 검출해 반환하는 예시 구현.
    • img: 그레이스케일(또는 단일 채널) 이미지 (np.ndarray)
    • threshold: RANSAC inlier 최대 오차 (픽셀 단위)
    • max_trials: RANSAC 최대 시도 횟수
    반환: [(x1, y1, x2, y2), …] 형태의 선 리스트
    """
    # 1) 에지 검출
    edges = cv2.Canny(img, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 2:
        return []

    # 2) RANSAC 회귀
    X = xs.reshape(-1, 1)
    y = ys
    ransac = RANSACRegressor(residual_threshold=threshold,
                             max_trials=max_trials)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_

    # 3) 모델 계수로 이미지 폭 전구간의 선 방정식 구하기
    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.intercept_)
    h, w = img.shape[:2]
    x0, x1 = 0, w
    y0, y1 = a * x0 + b, a * x1 + b

    return [(int(x0), int(y0), int(x1), int(y1))]


def ransac_extrapolate_endpoints(lines):
    if len(lines) < 2:
        return []
    # fit to all endpoints
    pts = np.array([(x1,y1) for x1,y1,x2,y2 in lines] + [(x2,y2) for x1,y1,x2,y2 in lines])
    X = pts[:,0].reshape(-1,1)
    y = pts[:,1]
    model = RANSACRegressor(random_state=0, max_trials=100)

    model.fit(X, y)
    xs = np.array([[X.min()],[X.max()]])
    ys = model.predict(xs)
    return [(int(xs[0,0]),int(ys[0])), (int(xs[1,0]),int(ys[1]))]

# Skeleton spur removal
def remove_spurs(skel_img, length=10):
    # remove small objects
    clean = remove_small_objects(skel_img.astype(bool), min_size=length)
    return (clean.astype(np.uint8) * 255)

# 1D intensity profile verify
def verify_intensity_endpoint(pt, gray_img, window=5):
    x,y = pt
    h,w = gray_img.shape
    vals = []
    for dx in range(-window,window+1):
        xx = min(max(x+dx,0),w-1)
        vals.append(gray_img[y,xx])
    grad = np.abs(np.diff(vals))
    if grad.max() < 10:
        return False
    return True

# subpixel refinement
def subpixel_refine(pt, grad_mag):
    x,y = pt
    # simple 3-point quadratic fit
    if x<=0 or x>=grad_mag.shape[1]-1: return pt
    g = grad_mag[y,x-1:x+2]
    denom = 2*(g[0] - 2*g[1] + g[2])
    if denom==0: return pt
    offset = (g[0] - g[2]) / denom
    return (x+offset, y)

# detect via Canny + Hough
def detect_hough_lines(gray_img, min_length=20):
    edges = cv2.Canny(gray_img,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=min_length,maxLineGap=5)
    if lines is None: return []
    return [tuple(l[0]) for l in lines]
# --- New utility functions ---
def apply_gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def detect_lines_from_ridge(gray_img, min_length=20):
    # Compute Hessian matrix components with explicit use_gaussian_derivatives=False
    hxx, hxy, hyy = hessian_matrix(gray_img, sigma=1.5, use_gaussian_derivatives=False)
    # Stack components for eigendecomposition
    H_elems = np.stack((hxx, hxy, hyy), axis=0)
    # Compute eigenvalues of Hessian: returns array shape (2, h, w)
    eigs = hessian_matrix_eigvals(H_elems)
    i1, i2 = eigs[0], eigs[1]

    # Create ridge mask: pixels where |i1| exceeds threshold
    thresh = np.mean(np.abs(i1)) + 1.5 * np.std(np.abs(i1))
    ridge_img = (np.abs(i1) > thresh).astype(np.uint8) * 255

    # Skeletonize ridge mask
    skel = skeletonize(ridge_img // 255).astype(np.uint8) * 255

    # Detect line segments on skeleton using LSD
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(skel)
    if lines is None:
        return []

    # Filter by minimum length
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        if np.hypot(x2 - x1, y2 - y1) >= min_length:
            filtered.append((x1, y1, x2, y2))
    return filtered

# --- Remaining code unchanged ---
# (Other functions and main pipeline)

def score_lead_candidate(line, hull, centroid):
    x1, y1, x2, y2 = line

    # proximity
    dist1 = cv2.pointPolygonTest(hull, (float(x1), float(y1)), True)
    dist2 = cv2.pointPolygonTest(hull, (float(x2), float(y2)), True)
    inner_dist = min(abs(dist1), abs(dist2))
    outer_dist = max(abs(dist1), abs(dist2))
    score_proximity = (1 / (inner_dist + 1)) * outer_dist
    # direction
    inner_pt, outer_pt = ((x1, y1), (x2, y2)) if dist1 > dist2 else ((x2, y2), (x1, y1))
    vec_out = (outer_pt[0] - centroid[0], outer_pt[1] - centroid[1])
    vec_line = (outer_pt[0] - inner_pt[0], outer_pt[1] - inner_pt[1])
    score_direction = max(np.dot(vec_out, vec_line), 0)
    # weighted sum
    return score_proximity * 0.5 + score_direction * 1.5


def select_best_pair(scored_lines, min_distance=10):
    # scored_lines: list of (line, score) sorted desc
    lines = [l for l, s in scored_lines]
    if len(lines) < 2:
        return lines
    # find first pair sufficiently apart
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            l1, l2 = lines[i], lines[j]
            m1 = ((l1[0]+l1[2]) / 2, (l1[1]+l1[3]) / 2)
            m2 = ((l2[0]+l2[2]) / 2, (l2[1]+l2[3]) / 2)
            if np.hypot(m1[0]-m2[0], m1[1]-m2[1]) > min_distance:
                return [l1, l2]
    return lines[:2]


def refine_endpoint(endpoint, gray_img, search_radius=5):
    x, y = endpoint
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    max_val = -1
    best = endpoint
    h, w = gray_img.shape
    for yy in range(max(0, y-search_radius), min(h, y+search_radius)):
        for xx in range(max(0, x-search_radius), min(w, x+search_radius)):
            if grad_mag[yy, xx] > max_val:
                max_val = grad_mag[yy, xx]
                best = (xx, yy)
    return best

# --- 전처리 및 몸체/선 검출 함수 (기존과 동일) ---
def remove_breadboard_holes(img, kernel_size=5, iterations=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=3, maxRadius=15)
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]: cv2.circle(mask, (i[0], i[1]), i[2] + 5, 0, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return cv2.bitwise_and(img, img, mask=mask)

def apply_clahe(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)

def remove_color_regions(img, remove_red=True, remove_blue=True):
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

def detect_led_body(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    body_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask1, red_mask2), cv2.bitwise_or(green_mask, yellow_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    M = cv2.moments(hull)
    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else None
    return hull, centroid

def detect_lines_lsd(img_gray, min_length=30):
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(img_gray)
    if lines is None: return []
    return [(int(x1), int(y1), int(x2), int(y2)) for line in lines for x1, y1, x2, y2 in [line[0]] if np.hypot(x2 - x1, y2 - y1) >= min_length]

def filter_common_lines(lines1, lines2, angle_thresh, dist_thresh):
    common = []
    for x1, y1, x2, y2 in lines1:
        theta1 = math.atan2(y2 - y1, x2 - x1)
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        for a1, b1, a2, b2 in lines2:
            theta2 = math.atan2(b2 - b1, a2 - a1)
            mid2 = ((a1 + a2) / 2, (b1 + b2) / 2)
            if abs(theta1 - theta2) < angle_thresh and math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1]) < dist_thresh:
                common.append((int((x1+a1)/2), int((y1+b1)/2), int((x2+a2)/2), int((y2+b2)/2)))
                break
    return common

def merge_similar_lines(lines, angle_thresh, dist_thresh):
    merged = []
    for line in lines:
        matched = False
        for i, merged_line in enumerate(merged):
            angle1 = math.atan2(line[3] - line[1], line[2] - line[0])
            angle2 = math.atan2(merged_line[3] - merged_line[1], merged_line[2] - merged_line[0])
            mid1 = ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2)
            mid2 = ((merged_line[0] + merged_line[2]) / 2, (merged_line[1] + merged_line[3]) / 2)
            if abs(angle1 - angle2) < angle_thresh and math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1]) < dist_thresh:
                points = [line[:2], line[2:], merged_line[:2], merged_line[2:]]
                farthest_pair = max([(p1, p2) for p1 in points for p2 in points], key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]))
                merged[i] = (farthest_pair[0][0], farthest_pair[0][1], farthest_pair[1][0], farthest_pair[1][1])
                matched = True
                break
        if not matched: merged.append(line)
    return merged

import cv2
import numpy as np

# LED 리드 끝점 검출: hull proximity 밴드 안↔밖 선분 필터링 후
# 밴드 바깥에 위치한 끝점만 추출하여 반환합니다.

def select_lead_lines(lines, hull, centroid, proximity_thresh):
    """
    Args:
        lines: list of (x1, y1, x2, y2)
        hull: numpy.ndarray (cv2.convexHull 결과)
        centroid: (cx, cy) 중심점 (score 계산용)
        proximity_thresh: hull과의 거리 임계치

    Returns:
        list of (x1, y1, x2, y2) 선별된 최대 2개의 선분
    """
    if hull is None or centroid is None or not lines:
        return []

    candidates = []
    for x1, y1, x2, y2 in lines:
        d1 = cv2.pointPolygonTest(hull, (float(x1), float(y1)), True)
        d2 = cv2.pointPolygonTest(hull, (float(x2), float(y2)), True)
        inside1 = abs(d1) <= proximity_thresh
        inside2 = abs(d2) <= proximity_thresh
        outside1 = not inside1
        outside2 = not inside2
        # 한 점은 밴드 안, 다른 점은 바깥
        if (inside1 and outside2) or (inside2 and outside1):
            candidates.append((x1, y1, x2, y2))

    if not candidates:
        return []

    if len(candidates) > 2:
        scored = [(line, score_lead_candidate(line, hull, centroid)) for line in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [scored[0][0], scored[1][0]]
    else:
        selected = candidates

    return selected


def get_final_endpoints(leads, hull):
    """
    select_lead_lines가 반환한 끝점 리스트 입력 시 그대로 반환,
    선분 리스트 입력 시 단순히 두 점을 모두 반환합니다.

    Args:
        leads: list of (x1, y1, x2, y2) 또는 list of (x, y)
        hull: unused placeholder to match 호출 시그니처

    Returns:
        list of (x, y) 형태의 끝점
    """
    # select_lead_lines가 이미 끝점 리스트를 반환했으면 그대로
    if leads and isinstance(leads[0], tuple) and len(leads[0]) == 2 and len(leads[0]) != 4:
        return leads

    # 그렇지 않으면 입력이 선분 리스트라고 보고 양끝점 모두 반환
    endpoints = []
    for x1, y1, x2, y2 in leads:
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    return endpoints



# --- 시각화 함수 ---
def visualize_lines(img, lines, window_name, color=(0, 255, 0)):
    vis = img.copy()
    for x1, y1, x2, y2 in lines: cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis, f"{window_name}: {len(lines)} lines", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis

def visualize_endpoint_detection(img, all_lines, lead_lines, endpoints, hull, centroid):
    vis = img.copy()
    if hull is not None: cv2.drawContours(vis, [hull], -1, (0, 255, 255), 2)
    if centroid is not None: cv2.circle(vis, centroid, 5, (0, 255, 255), -1)
    for x1, y1, x2, y2 in all_lines: cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
    for x1, y1, x2, y2 in lead_lines: cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i, (x, y) in enumerate(endpoints):
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(vis, f"T{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis

# --- 메인 실행 함수 ---
# Main pipeline

# 메인 파이프라인
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.linear_model import RANSACRegressor
import math
import os

# --- 빨간색 및 파란색 선 제거 및 주변 색상으로 대체 ---
def remove_red_blue_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                            cv2.inRange(hsv, lower_red2, upper_red2))
    lower_blue = np.array([100, 150, 0]); upper_blue = np.array([140, 255, 255])
    mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.dilate(cv2.bitwise_or(mask_r, mask_b),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# --- RANSAC line detection ---
def detect_lines_ransac(img, threshold, max_trials):
    edges = cv2.Canny(img, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 2:
        return []
    X = xs.reshape(-1, 1); y = ys
    ransac = RANSACRegressor(residual_threshold=threshold,
                             max_trials=max_trials, random_state=0)
    ransac.fit(X, y)
    a = float(ransac.estimator_.coef_[0]); b = float(ransac.estimator_.intercept_)
    h, w = img.shape[:2]; x0, x1 = 0, w
    y0, y1 = a * x0 + b, a * x1 + b
    return [(int(x0), int(y0), int(x1), int(y1))]

# --- 스켈레톤 스퍼 제거 ---
def remove_spurs(skel_img, length=10):
    clean = remove_small_objects(skel_img.astype(bool), min_size=length)
    return (clean.astype(np.uint8) * 255)

# --- 사용자 정의 detect, filter, merge, select, endpoint 등 함수는 기존과 동일 ---
# detect_lines_lsd, detect_hough_lines, detect_lines_from_ridge,
# filter_common_lines, merge_similar_lines, detect_led_body,
# select_lead_lines, get_final_endpoints, verify_intensity_endpoint,
# refine_endpoint, apply_clahe 등




# --- 두 선 사이 각도 계산 ---
def angle_between_lines(l1,l2):
    (x1,y1,x2,y2),(x3,y3,x4,y4)=l1,l2
    a1=math.atan2(y2-y1,x2-x1); a2=math.atan2(y4-y3,x4-x3)
    diff=abs(a1-a2); diff=min(diff,2*math.pi-diff)
    return math.degrees(diff)

# --- PCA 기반 끝점 검출 ---
def pca_endpoints(mask):
    pts = np.column_stack(np.where(mask>0))
    if len(pts)<2: return []
    pca=PCA(n_components=2); pca.fit(pts)
    axis = pca.components_[0]
    proj = pts.dot(axis)
    i0,i1 = np.argmin(proj), np.argmax(proj)
    return [tuple(pts[i0][::-1]), tuple(pts[i1][::-1])]  # (x,y)

# --- 컨투어 근사화 기반 끝점 검출 ---
def contour_endpoints(mask):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours: return []
    cnt=max(contours,key=cv2.contourArea)
    epsilon=0.01*cv2.arcLength(cnt,True)
    approx=cv2.approxPolyDP(cnt,epsilon,True).squeeze()
    pts=approx if approx.ndim==2 else cnt.reshape(-1,2)
    # convex hull
    hull=cv2.convexHull(pts)
    # farthest pair
    maxd=0; pair=(None,None)
    for i in range(len(hull)):
        for j in range(i+1,len(hull)):
            d=np.linalg.norm(hull[i][0]-hull[j][0])
            if d>maxd: maxd,pair=(d,(tuple(hull[i][0][::-1]),tuple(hull[j][0][::-1])))
    return [pair[0],pair[1]] if pair[0] else []


# --- 스켈레톤 엔드포인트 직접 추출 ---
def skeleton_endpoints(skel):
    # skel: uint8 image with 0 or 255
    bin_img = (skel > 0).astype(np.uint8)
    # 8-neighbor kernel without center
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(bin_img, -1, kernel)
    endpoints_mask = (bin_img == 1) & (neighbor_count == 1)
    coords = np.column_stack(np.where(endpoints_mask))  # (row, col)
    # return list of (x, y) and mask
    pts = [(int(c[1]), int(c[0])) for c in coords]
    return pts, endpoints_mask.astype(np.uint8)

# --- 거리 변환 기반 끝점 검출 ---
def distance_transform_endpoints(mask):
    # mask: binary uint8 image
    dst = cv2.distanceTransform((mask>0).astype(np.uint8), cv2.DIST_L2, 5)
    flat = dst.flatten()
    if flat.size < 2:
        return []
    idxs = np.argpartition(flat, -2)[-2:]
    h, w = mask.shape
    coords = [(int(i % w), int(i // w)) for i in idxs]
    return coords

# --- PCA 기반 끝점 검출 ---
def pca_endpoints(mask):
    pts = np.column_stack(np.where(mask > 0))  # (row, col)
    if pts.shape[0] < 2:
        return []
    pca = PCA(n_components=2)
    pca.fit(pts)
    axis = pca.components_[0]
    proj = pts.dot(axis)
    i0, i1 = int(np.argmin(proj)), int(np.argmax(proj))
    return [(int(pts[i0][1]), int(pts[i0][0])), (int(pts[i1][1]), int(pts[i1][0]))]

# --- 컨투어 근사화 기반 끝점 검출 ---
def contour_endpoints(mask):
    contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx.reshape(-1, 2)
    hull = cv2.convexHull(pts)
    maxd = 0
    pair = None
    for i in range(len(hull)):
        for j in range(i+1, len(hull)):
            pt1 = hull[i][0]
            pt2 = hull[j][0]
            d = np.linalg.norm(pt1 - pt2)
            if d > maxd:
                maxd = d
                pair = ((int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])))
    return list(pair) if pair else []

# --- RANSAC line detection ---
def detect_lines_ransac(img, threshold, max_trials):
    edges = cv2.Canny(img, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 2:
        return []
    X = xs.reshape(-1, 1)
    y = ys
    model = RANSACRegressor(residual_threshold=threshold, max_trials=max_trials, random_state=0)
    model.fit(X, y)
    a = float(model.estimator_.coef_[0])
    b = float(model.estimator_.intercept_)
    h, w = img.shape[:2]
    x0, x1 = 0, w
    y0, y1 = a * x0 + b, a * x1 + b
    return [(int(x0), int(y0), int(x1), int(y1))]

# --- 스켈레톤 스퍼 제거 ---
def remove_spurs(skel_img, length=10):
    clean = remove_small_objects(skel_img.astype(bool), min_size=length)
    return (clean.astype(np.uint8) * 255)

# --- 두 선 사이 각도 계산 ---
def angle_between_lines(l1, l2):
    x11, y11, x12, y12 = l1
    x21, y21, x22, y22 = l2
    a1 = math.atan2(y12 - y11, x12 - x11)
    a2 = math.atan2(y22 - y21, x22 - x21)
    diff = abs(a1 - a2)
    diff = min(diff, 2 * math.pi - diff)
    return math.degrees(diff)

# --- 사용자 정의 함수 placeholders ---
# apply_clahe, detect_lines_lsd, detect_hough_lines, detect_lines_from_ridge,
# filter_common_lines, merge_similar_lines, detect_led_body,
# select_lead_lines, get_final_endpoints, verify_intensity_endpoint,
# refine_endpoint 등은 기존 구현 사용

# --- 메인 파이프라인 ---
def main():
    images = [f for f in os.listdir('.') if 'led' in f.lower() and f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not images:
        print('No LED images found.')
        return
    idx = 0
    img = cv2.imread(images[idx])

    # 창 및 트랙바 설정
    windows = ['Original', 'Preprocessed', 'Gray', 'CLAHE', 'Adaptive Threshold', 'Skeleton',
               'Skeleton Endpoints', 'PCA Endpoints', 'Contour Endpoints', 'Distance Endpoints', 'Final Detection']
    for w in windows + ['LED Detection Controller']:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    params = [
        ('Min Length', 0, 100, 20), ('Hull Proximity', 0, 100, 20),
        ('Merge Angle', 0, 90, 10), ('Merge Distance', 0, 100, 25),
        ('CLAHE Clip x10', 0, 100, 20), ('Adaptive BS', 3, 51, 15), ('Adaptive C', 0, 20, 10),
        ('Spike Remove', 0, 1, 0), ('RANSAC', 0, 1, 0), ('RANSAC Iter', 1, 500, 100), ('RANSAC Thresh', 1, 100, 10)
    ]
    for name, _, vmax, init in params:
        cv2.createTrackbar(name, 'LED Detection Controller', init, vmax, lambda x: None)

    while True:
        # 트랙바 값 읽기
        vals = {n: cv2.getTrackbarPos(n, 'LED Detection Controller') for n, *_ in params}

        # 1) 빨강/파랑 라인 제거 및 전처리
        orig = remove_red_blue_lines(img.copy())
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        clahe_img = apply_clahe(gray, clipLimit=vals['CLAHE Clip x10'] / 10.0)
        bs = vals['Adaptive BS']
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        adaptive = cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, bs, vals['Adaptive C']
        )
        skel = skeletonize(adaptive // 255).astype(np.uint8) * 255
        if vals['Spike Remove']:
            skel = remove_spurs(skel)

        # 2) 여러 방식 엔드포인트 추출
        # 2a) 스켈레톤 이웃 기반
        sk_pts, sk_mask = skeleton_endpoints(skel)
        # 2b) PCA 기반
        pca_pts = pca_endpoints(adaptive)
        # 2c) 컨투어 기반
        cnt_pts = contour_endpoints(adaptive)
        # 2d) 거리 변환 기반
        dist_pts = distance_transform_endpoints(adaptive)

        # 시각화: 입력 및 중간 단계
        cv2.imshow('Original', orig)
        cv2.imshow('Gray', gray)
        cv2.imshow('CLAHE', clahe_img)
        cv2.imshow('Adaptive Threshold', adaptive)
        cv2.imshow('Skeleton', skel)
        cv2.imshow('Skeleton Endpoints', sk_mask * 255)

        # PCA mask
        pca_vis = orig.copy()
        for x, y in pca_pts:
            cv2.circle(pca_vis, (x, y), 5, (255, 0, 255), -1)
        cv2.imshow('PCA Endpoints', pca_vis)

        # Contour mask
        cnt_vis = orig.copy()
        for x, y in cnt_pts:
            cv2.circle(cnt_vis, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow('Contour Endpoints', cnt_vis)

        # Distance mask
        dist_vis = orig.copy()
        for x, y in dist_pts:
            cv2.circle(dist_vis, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Distance Endpoints', dist_vis)

        # 3) 기존 병합 선 및 리드 끝점
        hull, centroid = detect_led_body(orig)
        lines1 = detect_lines_lsd(clahe_img, vals['Min Length'])
        lines2 = detect_lines_lsd(adaptive, vals['Min Length'])
        common = filter_common_lines(lines2, lines1,
                                     np.deg2rad(vals['Merge Angle']),
                                     vals['Merge Distance'])
        merged = merge_similar_lines(common,
                                     np.deg2rad(vals['Merge Angle']),
                                     vals['Merge Distance'])
        leads = select_lead_lines(merged, hull, centroid, vals['Hull Proximity'])
        # 각도 필터
        filtered = []
        for l1, l2 in itertools.combinations(leads, 2):
            ang = angle_between_lines(l1, l2)
            if 20 <= ang <= 120:
                filtered = [l1, l2]
                break
        if filtered:
            leads = filtered
        endpoints = get_final_endpoints(leads, hull)
        if vals.get('Intensity', 0):
            endpoints = [pt for pt in endpoints if verify_intensity_endpoint(pt, gray)]
        if vals.get('Endpoint Refine', 0):
            endpoints = [refine_endpoint(pt, gray) for pt in endpoints]

        # 4) RANSAC 끝점
        endpoints_r = []
        if vals['RANSAC']:
            lines_r = detect_lines_ransac(clahe_img,
                                          vals['RANSAC Thresh'],
                                          vals['RANSAC Iter'])
            leads_r = select_lead_lines(lines_r, hull, centroid, vals['Hull Proximity'])
            filtered_r = []
            for l1, l2 in itertools.combinations(leads_r, 2):
                ang = angle_between_lines(l1, l2)
                if 20 <= ang <= 120:
                    filtered_r = [l1, l2]
                    break
            if filtered_r:
                leads_r = filtered_r
            endpoints_r = get_final_endpoints(leads_r, hull)

        # 5) band 영역
        band = None
        if hull is not None and vals['Hull Proximity'] > 0:
            h, w = skel.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [hull], -1, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (2 * vals['Hull Proximity'] + 1,) * 2)
            band = cv2.subtract(cv2.dilate(mask, kernel), mask)

        # 6) 최종 시각화
        final = orig.copy()
        if hull is not None:
            cv2.drawContours(final, [hull], -1, (0, 255, 255), 2)
        if centroid is not None:
            cv2.circle(final, centroid, 5, (0, 255, 255), -1)
        for x1, y1, x2, y2 in merged:
            cv2.line(final, (x1, y1), (x2, y2), (128, 128, 128), 1)
        for x1, y1, x2, y2 in leads:
            cv2.line(final, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i, (x, y) in enumerate(endpoints):
            color = (0, 0, 255) if band is not None and band[y, x] == 255 else (255, 0, 0)
            cv2.circle(final, (x, y), 8, color, -1)
            cv2.putText(final, f"M{i+1}", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for i, (x, y) in enumerate(endpoints_r):
            color = (0, 0, 255) if band is not None and band[y, x] == 255 else (255, 0, 0)
            cv2.circle(final, (x, y), 10, color, -1)
            cv2.putText(final, f"R{i+1}", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('Final Detection', final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('n'):
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx])
        if key == ord('p'):
            idx = (idx - 1) % len(images)
            img = cv2.imread(images[idx])

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
