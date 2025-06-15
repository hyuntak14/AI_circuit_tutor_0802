import cv2
import numpy as np
import math
import os
from sklearn.cluster import DBSCAN

# --- 빨간색 및 파란색 선 제거 및 주변 색상으로 대체 ---
def remove_red_blue_lines(img):
    """원본 BGR 이미지에서 빨간색과 파란색 선을 마스크하고, 주변 픽셀 색상으로 인페인팅하여 제거합니다."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    lower_blue = np.array([100, 150, 0]); upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_or(mask_red, mask_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# --- CLAHE 적용 ---
def apply_clahe(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)

# --- 작은 영역 제거 ---
def remove_small_regions(bin_img, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    output = np.zeros_like(bin_img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255
    return output

# --- 스켈레톤화 ---
def skeletonize(bin_img):
    """이진 이미지에서 모폴로지 연산을 통해 스켈레톤 추출"""
    skel = np.zeros(bin_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = bin_img.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

# --- LSD Line Detection ---
def detect_lines_lsd(img_gray, min_length=10):
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(img_gray)
    if lines is None:
        return []
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if np.hypot(x2 - x1, y2 - y1) >= min_length:
            filtered.append((int(x1), int(y1), int(x2), int(y2)))
    return filtered

# --- DBSCAN 병합 ---
def merge_lines_dbscan(lines, angle_thresh, dist_thresh, min_samples=2):
    if not lines:
        return []
    w = dist_thresh / angle_thresh
    feats = []
    for x1, y1, x2, y2 in lines:
        angle = math.atan2(y2 - y1, x2 - x1)
        feats.append([(x1 + x2) / 2, (y1 + y2) / 2, angle * w])
    feats = np.array(feats)
    labels = DBSCAN(eps=dist_thresh, min_samples=min_samples).fit_predict(feats)
    merged = []
    for lbl in set(labels):
        idxs = np.where(labels == lbl)[0]
        if lbl == -1:
            for i in idxs:
                merged.append(lines[i])
            continue
        pts = []
        for i in idxs:
            x1, y1, x2, y2 = lines[i]
            pts.extend([(x1, y1), (x2, y2)])
        max_d, best = 0, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if d > max_d:
                    max_d, best = d, (pts[i], pts[j])
        if best:
            (sx, sy), (ex, ey) = best
            merged.append((sx, sy, ex, ey))
    return merged

# --- 교집합 선 추출 ---
def intersect_similar_lines(lines1, lines2, angle_thresh, dist_thresh):
    inter = []
    for x1, y1, x2, y2 in lines1:
        angle1 = math.atan2(y2 - y1, x2 - x1)
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        for xx1, yy1, xx2, yy2 in lines2:
            angle2 = math.atan2(yy2 - yy1, xx2 - xx1)
            if min(abs(angle1 - angle2), 2 * math.pi - abs(angle1 - angle2)) < angle_thresh:
                mid2 = ((xx1 + xx2) / 2, (yy1 + yy2) / 2)
                if math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1]) < dist_thresh:
                    inter.append((x1, y1, x2, y2))
                    break
    return inter

# --- 시각화 함수 ---
def visualize_lines(img, lines, window_name, color=(0, 255, 0)):
    vis = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis, f"{window_name}: {len(lines)} lines", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis

# --- 메인 함수 ---
def main():
    images = [f for f in os.listdir('.') if 'led' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print('No LED images found in current directory.')
        return
    idx = 0
    img = cv2.imread(images[idx])
    if img is None:
        print(f"Failed to load image: {images[idx]}")
        return
    windows = [
        'Original', 'Preprocessed', 'Gray', 'CLAHE',
        'Otsu', 'Adaptive', 'Clean Otsu', 'Clean Adaptive',
        'Canny Otsu', 'Canny Adaptive', 'Skeleton Otsu', 'Skeleton Adaptive',
        'Lines Gray', 'Lines Otsu', 'Lines Adaptive', 'Merged Lines'
    ]
    for w in windows:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controller', cv2.WINDOW_NORMAL)
    params = [
        ('Min Length', 10, 100, 10),
        ('CLAHE Clip x10', 20, 100, 20),
        ('Adaptive BS', 11, 101, 15),
        ('Adaptive C', 2, 20, 10),
        ('Merge Angle', 5, 45, 10),
        ('Merge Distance', 10, 100, 25),
        ('Min Area', 50, 1000, 100)
    ]
    for name, default, max_val, _ in params:
        cv2.createTrackbar(name, 'Controller', default, max_val, lambda x: None)
    while True:
        min_length = cv2.getTrackbarPos('Min Length', 'Controller')
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip x10', 'Controller') / 10.0
        adaptive_bs = cv2.getTrackbarPos('Adaptive BS', 'Controller')
        if adaptive_bs % 2 == 0:
            adaptive_bs += 1
        adaptive_bs = max(adaptive_bs, 3)
        adaptive_c = cv2.getTrackbarPos('Adaptive C', 'Controller')
        merge_angle = np.deg2rad(cv2.getTrackbarPos('Merge Angle', 'Controller'))
        merge_dist = cv2.getTrackbarPos('Merge Distance', 'Controller')
        min_area = cv2.getTrackbarPos('Min Area', 'Controller')
        preprocessed = remove_red_blue_lines(img.copy())
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        clahe_img = apply_clahe(gray, clipLimit=clahe_clip)
        _, otsu_binary = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_binary = cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, adaptive_bs, adaptive_c
        )
        clean_otsu = remove_small_regions(otsu_binary, min_area)
        clean_adapt = remove_small_regions(adaptive_binary, min_area)
        canny_otsu = cv2.Canny(clean_otsu, 50, 150)
        canny_adapt = cv2.Canny(clean_adapt, 50, 150)
        skel_otsu = skeletonize(canny_otsu)
        skel_adapt = skeletonize(canny_adapt)
        lines_gray = detect_lines_lsd(clahe_img, min_length)
        lines_otsu = detect_lines_lsd(otsu_binary, min_length)
        lines_adaptive = detect_lines_lsd(adaptive_binary, min_length)
        inter_all = intersect_similar_lines(lines_gray, lines_adaptive, merge_angle, merge_dist)
        merged_lines = merge_lines_dbscan(inter_all, merge_angle, merge_dist, min_samples=2)
        merged_lines = merge_lines_dbscan(merged_lines, merge_angle, merge_dist, min_samples=2)
        cv2.imshow('Original', img)
        cv2.imshow('Preprocessed', preprocessed)
        cv2.imshow('Gray', gray)
        cv2.imshow('CLAHE', clahe_img)
        cv2.imshow('Otsu', otsu_binary)
        cv2.imshow('Adaptive', adaptive_binary)
        cv2.imshow('Clean Otsu', clean_otsu)
        cv2.imshow('Clean Adaptive', clean_adapt)
        cv2.imshow('Canny Otsu', canny_otsu)
        cv2.imshow('Canny Adaptive', canny_adapt)
        cv2.imshow('Skeleton Otsu', skel_otsu)
        cv2.imshow('Skeleton Adaptive', skel_adapt)
        cv2.imshow('Lines Gray', visualize_lines(preprocessed, lines_gray, 'Gray'))
        cv2.imshow('Lines Otsu', visualize_lines(preprocessed, lines_otsu, 'Otsu', (255, 0, 0)))
        cv2.imshow('Lines Adaptive', visualize_lines(preprocessed, lines_adaptive, 'Adaptive', (0, 0, 255)))
        cv2.imshow('Merged Lines', visualize_lines(preprocessed, merged_lines, 'Merged Lines', (255, 255, 0)))
        print(f"\rImage: {images[idx]} | Gray: {len(lines_gray)} | Otsu: {len(lines_otsu)} | Adaptive: {len(lines_adaptive)} | Merged: {len(merged_lines)} lines", end='')
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx])
            print()
        elif key == ord('p'):
            idx = (idx - 1) % len(images)
            img = cv2.imread(images[idx])
            print()
        elif key == ord('s'):
            base_name = os.path.splitext(images[idx])[0]
            cv2.imwrite(f"{base_name}_merged_lines.png", visualize_lines(preprocessed, merged_lines, 'Merged Lines'))
            print(f"\nSaved: {base_name}_merged_lines.png")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
