import os
import cv2
import numpy as np
from glob import glob

# 전역 변수 초기값
threshold = 100       # 검은색 임계값
min_hull_area = 50    # convex hull 최소 면적


def nothing(x):
    pass


def get_masks(gray, thresh):
    """
    다양한 이진화 마스크 생성: 고정(thresh), Otsu, adaptive
    """
    _, mask_fixed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_fixed = cv2.morphologyEx(mask_fixed, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_adaptive = cv2.morphologyEx(mask_adaptive, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask_fixed, mask_otsu, mask_adaptive


def draw_adaptive_hull(image, mask_adaptive, min_area):
    """
    Adaptive Threshold 마스크 기반으로, min_area 이상의 모든 hull을 합쳐
    하나의 convex hull만 그려서 반환
    """
    contours, _ = cv2.findContours(mask_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull) >= min_area:
            kept.append(hull)

    output = image.copy()
    if kept:
        all_pts = np.vstack(kept)
        merged = cv2.convexHull(all_pts)
        cv2.drawContours(output, [merged], -1, (255, 0, 0), 2)  # 파란색으로 표시
    return output


def draw_fixed_hull(image, mask_fixed, min_area):
    """고정 임계값 마스크 기반으로 작은 영역 제거 후 합쳐진 hull 그리기"""
    contours, _ = cv2.findContours(mask_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept, removed = [], []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull) >= min_area:
            kept.append(hull)
        else:
            removed.append(hull)
    output = image.copy()
    for rh in removed:
        cv2.drawContours(output, [rh], -1, (0, 0, 255), 2)
    if kept:
        all_pts = np.vstack(kept)
        merged = cv2.convexHull(all_pts)
        cv2.drawContours(output, [merged], -1, (0, 255, 0), 2)
    return output


def draw_otsu_hull(image, mask_otsu, min_area):
    """
    Otsu 이진화 마스크 기반으로, min_area 이상의 모든 hull을 합쳐
    하나의 convex hull만 그려서 반환
    """
    contours, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull) >= min_area:
            kept.append(hull)

    output = image.copy()
    if kept:
        all_pts = np.vstack(kept)
        merged = cv2.convexHull(all_pts)
        cv2.drawContours(output, [merged], -1, (0, 255, 0), 2)

    return output



def main():
    global threshold, min_hull_area
    # 'cap' 포함된 이미지 검색
    exts = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff')
    files = sum([glob(ext) for ext in exts], [])
    cap_files = [f for f in files if 'cap' in f.lower()]
    if not cap_files:
        print("'cap'이 포함된 파일을 찾을 수 없습니다.")
        return

    idx = 0
    win = 'Preview'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Threshold', win, threshold, 255, nothing)
    cv2.createTrackbar('MinHullArea', win, min_hull_area, 1000, nothing)

    while True:
        threshold = cv2.getTrackbarPos('Threshold', win)
        min_hull_area = cv2.getTrackbarPos('MinHullArea', win)

        filepath = cap_files[idx]
        image = cv2.imread(filepath)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {filepath}")
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 마스크 생성
        mask_fixed, mask_otsu, mask_adaptive = get_masks(gray, threshold)
        # 결과 그림
        fixed_out = draw_fixed_hull(image, mask_fixed, min_hull_area)
        #otsu_out = draw_otsu_hull(mask_otsu, min_hull_area)
        otsu_out = draw_otsu_hull(image, mask_otsu, min_hull_area)
        #adaptive_out = cv2.cvtColor(mask_adaptive, cv2.COLOR_GRAY2BGR)
        adaptive_out = draw_adaptive_hull(image, mask_adaptive, min_hull_area)

        disp_fixed = cv2.cvtColor(mask_fixed, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([
            image, disp_fixed, otsu_out, adaptive_out, fixed_out
        ])
        cv2.imshow(win, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'):
            idx = (idx + 1) % len(cap_files)
        elif key == ord('p'):
            idx = (idx - 1) % len(cap_files)
        elif key == ord('s'):
            out_dir = 'output_hull'
            os.makedirs(out_dir, exist_ok=True)
            name, ext = os.path.splitext(os.path.basename(filepath))
            out_path = os.path.join(out_dir, f"{name}_merged_hull{ext}")
            cv2.imwrite(out_path, fixed_out)
            print(f"저장 완료: {out_path}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
