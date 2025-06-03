# run_gui_led_detector.py

import os
import glob
import cv2
import numpy as np
from skimage import morphology
from detector import ImprovedLEDEndpointDetector  # RGB 입력을 기대하는 detector

# 트랙바 콜백 (아무 동작 안 함)
def nothing(x):
    pass

def setup_trackbars():
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('min_contour_area','Controls',  50, 500, nothing)
    cv2.createTrackbar('blur_kernel',      'Controls',   3, 31, nothing)
    cv2.createTrackbar('canny_low',        'Controls',  50, 255, nothing)
    cv2.createTrackbar('canny_high',       'Controls', 150, 255, nothing)
    cv2.createTrackbar('min_object_size',  'Controls',  30, 500, nothing)
    cv2.createTrackbar('dyn_pct',          'Controls',   0, 100, nothing)  # 0→비활성, 1–100→percentile
    cv2.createTrackbar('Binarization', 'Controls', 0, 2, nothing) 

def get_trackbar_vals():
    mca = cv2.getTrackbarPos('min_contour_area','Controls')
    bk  = cv2.getTrackbarPos('blur_kernel',     'Controls')
    if bk % 2 == 0: bk += 1
    cl  = cv2.getTrackbarPos('canny_low',       'Controls')
    ch  = cv2.getTrackbarPos('canny_high',      'Controls')
    mos = cv2.getTrackbarPos('min_object_size', 'Controls')
    dp  = cv2.getTrackbarPos('dyn_pct',         'Controls')
    dp_val = dp if dp>0 else None
    bin_method_val = cv2.getTrackbarPos('Binarization', 'Controls')
    bin_method = {0:'otsu', 1:'adaptive', 2:'sauvola'}.get(bin_method_val, 'otsu')
    return mca, bk, cl, ch, mos, dp, bin_method

def visualize(img_bgr, detector):
    # BGR→RGB 변환 후 detector에 전달
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray, bin_raw, bin_clean, edges = detector.preprocess_image_advanced(img)
    metal_mask = detector.detect_metal_pins(img)

    # 스켈레톤
    bw = bin_clean > 0
    skeleton = morphology.skeletonize(bw)

    # 끝점
    endpoints = detector.detect_endpoints(img)

    # 윈도우에 출력
    cv2.imshow('Original RGB',    img)
    cv2.imshow('Grayscale',       gray)
    cv2.imshow('Binary - Raw',    bin_raw)
    cv2.imshow('Binary - Cleaned',bin_clean)
    cv2.imshow('Edges',           edges)
    cv2.imshow('Metal Mask',      metal_mask)
    cv2.imshow('Skeleton',        (skeleton*255).astype(np.uint8))

    # Detection
    disp = img.copy()
    if endpoints:
        (x1,y1),(x2,y2) = endpoints
        cv2.circle(disp, (x1,y1), 6, (255,0,0), -1)
        cv2.circle(disp, (x2,y2), 6, (0,255,0), -1)
        cv2.line(disp, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow('Detection', disp)

def main(input_dir: str = '.'):
    # 이미지 리스트
    exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
    paths = sum((glob.glob(os.path.join(input_dir, e)) for e in exts), [])
    paths = [p for p in paths if 'led' in os.path.basename(p).lower()]
    paths.sort()
    if not paths:
        print("➤ 'led' 포함 이미지가 없습니다.")
        return

    setup_trackbars()
    idx = 0
    while True:
        img_bgr = cv2.imread(paths[idx])
        if img_bgr is None:
            print(f"❌ 로드 실패: {paths[idx]}")
            idx = (idx + 1) % len(paths)
            continue

        mca, bk, cl, ch, mos, dp = get_trackbar_vals()
        det = ImprovedLEDEndpointDetector(
            min_contour_area=mca,
            blur_kernel_size=bk,
            canny_low=cl,
            canny_high=ch,
            min_object_size=mos,
            dynamic_percentile=dp,
            debug_mode=False,
            binarization_method=bin_method
        )
        visualize(img_bgr, det)

        key = cv2.waitKey(100) & 0xFF
        if key == 27:       # ESC
            break
        elif key in (83, ord('n')):  # → or 'n'
            idx = (idx + 1) % len(paths)
        elif key in (81, ord('p')):  # ← or 'p'
            idx = (idx - 1) % len(paths)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, default='.', help='LED 이미지 폴더')
    args = p.parse_args()
    main(args.input_dir)
