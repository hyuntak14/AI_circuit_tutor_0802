import os
import cv2
import numpy as np
from resistor_detector import ResistorEndpointDetector
from skimage.morphology import skeletonize
from skimage.measure import LineModelND, ransac

# --- 개선 메서드 함수 정의 ---
def denoise_color(image):
    """컬러 이미지 노이즈 제거 (Non-Local Means)"""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def binarize(gray, method='otsu', fixed_thresh=120, block_size=11, C=2):
    """그레이스케일 이진화"""
    if method == 'otsu':
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'fixed':
        _, bw = cv2.threshold(gray, fixed_thresh, 255, cv2.THRESH_BINARY_INV)
    elif method == 'adaptive':
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, C)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw

def morph_ops(bw, opening=True, closing=True, kernel_size=3, iterations=1):
    """모폴로지 연산"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if opening:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if closing:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return bw

def skeletonize_mask(bw):
    """스켈레톤화"""
    return (skeletonize(bw // 255).astype(np.uint8) * 255)

def find_skel_endpoints(skel):
    """스켈레톤 기준 끝점 검출"""
    endpoints = []
    h, w = skel.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y, x] > 0:
                neigh = np.sum(skel[y-1:y+2, x-1:x+2] > 0) - 1
                if neigh == 1:
                    endpoints.append((x, y))
    if len(endpoints) >= 2:
        max_d, best = -1, None
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                dx = endpoints[i][0] - endpoints[j][0]
                dy = endpoints[i][1] - endpoints[j][1]
                d = dx*dx + dy*dy
                if d > max_d:
                    max_d, best = d, (endpoints[i], endpoints[j])
        return list(best)
    return []

def ransac_endpoints(skel):
    """RANSAC 기반 주축 끝점 검출"""
    pts = np.column_stack(np.where(skel > 0))
    if len(pts) < 2:
        return []
    model, inliers = ransac(data=pts, model_class=LineModelND,
                            min_samples=2, residual_threshold=1, max_trials=100)
    origin, direction = model.params
    t = np.dot(pts - origin, direction)
    i_min, i_max = np.argmin(t), np.argmax(t)
    p1 = (int(pts[i_min][1]), int(pts[i_min][0]))
    p2 = (int(pts[i_max][1]), int(pts[i_max][0]))
    return [p1, p2]

def hough_endpoints(bw, threshold=50, min_length_ratio=0.5, max_gap=10):
    """HoughLinesP 기반 끝점 검출"""
    edges = cv2.Canny(bw, 50, 150)
    h, w = bw.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold,
                             minLineLength=int(w * min_length_ratio),
                             maxLineGap=max_gap)
    if lines is None:
        return []
    lines = lines.reshape(-1, 4)
    x1, y1, x2, y2 = max(lines, key=lambda l: (l[0]-l[2])**2 + (l[1]-l[3])**2)
    return [(x1, y1), (x2, y2)]

def detect_endpoints(image, flags, detector):
    """전체 파이프라인으로 엔드포인트 검출"""
    img = image.copy()
    if flags['denoise']:
        img = denoise_color(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = 'adaptive' if flags['adaptive'] else 'otsu'
    bw = binarize(gray, method=method, fixed_thresh=flags['fixed_thresh'],
                  block_size=flags['block_size'], C=flags['C'])
    bw = morph_ops(bw, opening=flags['opening'], closing=flags['closing'],
                   kernel_size=flags['kernel_size'], iterations=flags['iterations'])
    skel = skeletonize_mask(bw)
    filtered_skel, comps = detector.filter_resistor_skeleton(skel)
    eps = []
    if flags['pca'] and comps:
        eps = detector.find_principal_axis_endpoints(filtered_skel)
    if not eps and flags['skeleton']:
        eps = find_skel_endpoints(filtered_skel)
    if not eps and flags['ransac']:
        eps = ransac_endpoints(filtered_skel)
    if not eps and flags['hough']:
        eps = hough_endpoints(bw,
                              threshold=flags['hough_thresh'],
                              min_length_ratio=flags['min_length_ratio'],
                              max_gap=flags['max_line_gap'])
    if eps:
        x1, y1 = flags['bbox'][0], flags['bbox'][1]
        eps = [(x1 + eps[0][0], y1 + eps[0][1]), (x1 + eps[1][0], y1 + eps[1][1])]
    return eps

# --- 메인 ---
def main():
    current_dir = os.getcwd()
    files = [f for f in os.listdir(current_dir)
             if "resistor" in f.lower() and f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not files:
        print("resistor가 포함된 이미지 파일이 없습니다.")
        return
    files = sorted(files)

    # 썸네일 크기 설정
    thumb_h, thumb_w = 200, 200

    ctrl = 'Controls'
    cv2.namedWindow(ctrl)
    def nothing(x): pass
    cv2.createTrackbar('Denoise', ctrl, 0, 1, nothing)
    cv2.createTrackbar('Adaptive', ctrl, 0, 1, nothing)
    cv2.createTrackbar('Opening', ctrl, 1, 1, nothing)
    cv2.createTrackbar('Closing', ctrl, 1, 1, nothing)
    cv2.createTrackbar('PCA', ctrl, 1, 1, nothing)
    cv2.createTrackbar('Skeleton', ctrl, 1, 1, nothing)
    cv2.createTrackbar('RANSAC', ctrl, 0, 1, nothing)
    cv2.createTrackbar('Hough', ctrl, 1, 1, nothing)
    cv2.createTrackbar('KernelSize', ctrl, 3, 31, nothing)
    cv2.createTrackbar('Iterations', ctrl, 1, 5, nothing)
    cv2.createTrackbar('BlockSize', ctrl, 11, 51, nothing)
    cv2.createTrackbar('C', ctrl, 2, 10, nothing)
    cv2.createTrackbar('FixedThr', ctrl, 120, 255, nothing)
    cv2.createTrackbar('HoughThr', ctrl, 50, 200, nothing)
    cv2.createTrackbar('MinLenRatio', ctrl, 50, 100, nothing)
    cv2.createTrackbar('MaxGap', ctrl, 10, 50, nothing)

    detector = ResistorEndpointDetector(visualize=False)
    batch_size = 10
    idx = 0

    while True:
        flags = {
            'denoise': bool(cv2.getTrackbarPos('Denoise', ctrl)),
            'adaptive': bool(cv2.getTrackbarPos('Adaptive', ctrl)),
            'opening': bool(cv2.getTrackbarPos('Opening', ctrl)),
            'closing': bool(cv2.getTrackbarPos('Closing', ctrl)),
            'pca': bool(cv2.getTrackbarPos('PCA', ctrl)),
            'skeleton': bool(cv2.getTrackbarPos('Skeleton', ctrl)),
            'ransac': bool(cv2.getTrackbarPos('RANSAC', ctrl)),
            'hough': bool(cv2.getTrackbarPos('Hough', ctrl)),
            'kernel_size': max(3, cv2.getTrackbarPos('KernelSize', ctrl) // 2 * 2 + 1),
            'iterations': cv2.getTrackbarPos('Iterations', ctrl),
            'block_size': max(3, cv2.getTrackbarPos('BlockSize', ctrl) // 2 * 2 + 1),
            'C': cv2.getTrackbarPos('C', ctrl),
            'fixed_thresh': cv2.getTrackbarPos('FixedThr', ctrl),
            'hough_thresh': cv2.getTrackbarPos('HoughThr', ctrl),
            'min_length_ratio': cv2.getTrackbarPos('MinLenRatio', ctrl) / 100.0,
            'max_line_gap': cv2.getTrackbarPos('MaxGap', ctrl)
        }

        batch = files[idx:idx+batch_size]
        vis_list = []
        for fname in batch:
            img = cv2.imread(fname)
            if img is None:
                vis = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            else:
                h, w = img.shape[:2]
                flags['bbox'] = (0, 0, w, h)
                eps = detect_endpoints(img, flags, detector)
                vis = img.copy()
                if eps:
                    for p in eps:
                        cv2.circle(vis, p, 5, (0, 255, 0), -1)
                    cv2.line(vis, eps[0], eps[1], (255, 0, 0), 2)
                vis = cv2.resize(vis, (thumb_w, thumb_h))
            vis_list.append(vis)

        # 그리드 생성 (2x5)
        rows = []
        for r in range(2):
            row_imgs = vis_list[r*5:(r+1)*5]
            if len(row_imgs) < 5:
                blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                row_imgs += [blank] * (5 - len(row_imgs))
            rows.append(np.hstack(row_imgs))
        grid = np.vstack(rows)
        cv2.imshow('Batch Results', grid)

        key = cv2.waitKey(200)
        if key == ord('n'):
            idx = (idx + batch_size) % len(files)
        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
