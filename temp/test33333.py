import cv2
import numpy as np
import math
import os

# --- 빨간색 및 파란색 선 제거 및 주변 색상으로 대체 ---
def remove_red_blue_lines(img):
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
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# --- CLAHE 적용 ---
def apply_clahe(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)

# --- LSD Line Detection ---
def detect_lines_lsd(img_gray, min_length=10):
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(img_gray)
    if lines is None:
        return []
    filtered = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if np.hypot(x2-x1, y2-y1) >= min_length:
            filtered.append((int(x1),int(y1),int(x2),int(y2)))
    return filtered

# --- Standard Hough Transform ---
def detect_lines_hough(img_gray, rho=1, theta=np.pi/180, threshold=150):
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    if lines is None:
        return []
    result = []
    for rho, theta in lines[:,0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*(a))
        result.append((x1, y1, x2, y2))
    return result

# --- Probabilistic Hough Transform ---
def detect_lines_houghp(img_gray, min_length=30, max_gap=5, threshold=80):
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=min_length, maxLineGap=max_gap)
    if lines is None:
        return []
    return [(x1, y1, x2, y2) for [[x1, y1, x2, y2]] in lines]

# --- Fast Line Detector (ximgproc) ---
def detect_lines_fast(img_gray, length=10, distance=1.4142):
    try:
        fld = cv2.ximgproc.createFastLineDetector(length, distance)
    except AttributeError:
        return []  # OpenCV ximgproc가 없으면 빈 리스트 반환
    lines = fld.detect(img_gray)
    if lines is None:
        return []
    return [(int(x1), int(y1), int(x2), int(y2)) for [[x1, y1, x2, y2]] in lines]

# --- 선 시각화 ---
def visualize_lines(img, lines, window_name, color=(0, 255, 0)):
    vis = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis, f"{window_name}: {len(lines)} lines", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis

# --- 메인 함수 ---
def main():
    # LED 이미지 찾기
    images = [f for f in os.listdir('.') if 'led' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print('No LED images found in current directory.')
        return
    print(f"Found {len(images)} LED images")
    idx = 0

    img = cv2.imread(images[idx])
    if img is None:
        print(f"Failed to load image: {images[idx]}")
        return

    # 윈도우 생성
    win_names = [
        'Original', 'Preprocessed', 'Gray', 'CLAHE',
        'LSD', 'Hough', 'HoughP', 'Fast', 'Merged'
    ]
    for w in win_names:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controller', cv2.WINDOW_NORMAL)

    # 트랙바 설정
    params = [
        ('Min Length', 10, 200, 30),
        ('CLAHE Clip x10', 20, 100, 20),
        ('Adaptive BS', 11, 101, 15),
        ('Adaptive C', 2, 20, 10),
        ('Merge Angle', 5, 45, 10),
        ('Merge Distance', 10, 100, 25)
    ]
    for name, default, max_val, _ in params:
        cv2.createTrackbar(name, 'Controller', default, max_val, lambda x: None)

    while True:
        # 트랙바 값 읽기
        min_length = cv2.getTrackbarPos('Min Length', 'Controller')
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip x10', 'Controller') / 10.0
        adaptive_bs = cv2.getTrackbarPos('Adaptive BS', 'Controller')
        adaptive_c = cv2.getTrackbarPos('Adaptive C', 'Controller')
        merge_angle = cv2.getTrackbarPos('Merge Angle', 'Controller')
        merge_dist = cv2.getTrackbarPos('Merge Distance', 'Controller')
        if adaptive_bs % 2 == 0: adaptive_bs += 1
        adaptive_bs = max(adaptive_bs, 3)

        # 1. 전처리
        pre = remove_red_blue_lines(img.copy())
        gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        clahe_img = apply_clahe(gray, clipLimit=clahe_clip)

        # 2. LSD, Hough, HoughP, Fast 검출
        lines_lsd = detect_lines_lsd(clahe_img, min_length)
        lines_hough = detect_lines_hough(clahe_img)
        lines_houghp = detect_lines_houghp(clahe_img, min_length, max_gap=5)
        lines_fast = detect_lines_fast(clahe_img)

        # 3. 병합 (LSD 결과를 예시로)
        angle_thresh = np.deg2rad(merge_angle)
        merged = lines_lsd + lines_houghp
        # 간단 병합: 여기서는 DBSCAN 병합 대신 라디안+거리 기준 필터링 적용 가능

        # 4. 시각화
        cv2.imshow('Original', img)
        cv2.imshow('Preprocessed', pre)
        cv2.imshow('Gray', gray)
        cv2.imshow('CLAHE', clahe_img)
        cv2.imshow('LSD', visualize_lines(pre, lines_lsd, 'LSD', (0,255,0)))
        cv2.imshow('Hough', visualize_lines(pre, lines_hough, 'Hough', (255,0,0)))
        cv2.imshow('HoughP', visualize_lines(pre, lines_houghp, 'HoughP', (0,0,255)))
        cv2.imshow('Fast', visualize_lines(pre, lines_fast, 'Fast', (255,255,0)))
        cv2.imshow('Merged', visualize_lines(pre, merged, 'Merged', (255,255,255)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord('n'):
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx]); print()
        elif key == ord('p'):
            idx = (idx - 1) % len(images)
            img = cv2.imread(images[idx]); print()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
