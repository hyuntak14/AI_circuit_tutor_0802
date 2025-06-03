# test_led_endpoint_detector_with_vis.py

import os
import cv2
import numpy as np
from led_detector import LedEndpointDetector

# 시각화를 위해 중간 이미지들도 저장하는 함수
def save_intermediate_images(gray, gamma, bw, out_prefix):
    cv2.imwrite(f"{out_prefix}_gray.png", gray)
    cv2.imwrite(f"{out_prefix}_gamma.png", gamma)
    cv2.imwrite(f"{out_prefix}_bw.png", bw)

# LedEndpointDetector에서 _preprocess를 접근할 수 있도록 약간의 해킹
# 또는, 아예 led_detector.py에서 _preprocess를 public으로 바꿔도 됨.
def preprocess_and_extract(detector, img, bbox, holes, out_prefix):
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    gray, gamma, bw = detector._preprocess(roi)

    # 중간 결과 시각화 저장
    save_intermediate_images(gray, gamma, bw, out_prefix)

    # 이어서 기존 extract 과정
    for hx, hy in holes:
        if x1 <= hx < x2 and y1 <= hy < y2:
            mx, my = hx - x1, hy - y1
            cv2.circle(bw, (mx, my), detector.hole_mask_radius, 0, -1)

    from skimage.morphology import skeletonize
    skel = skeletonize(bw//255).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    clean = np.zeros_like(skel)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= detector.min_skel_area:
            clean[labels == i] = 255
    skel = clean

    # 엔드포인트 검출
    eps = []
    h, w = skel.shape
    for yy in range(1, h-1):
        for xx in range(1, w-1):
            if skel[yy, xx]:
                neigh = np.sum(skel[yy-1:yy+2, xx-1:xx+2] > 0) - 1
                if neigh == 1:
                    eps.append((xx, yy))
    if len(eps) < 2:
        return None

    # 최장거리 두 점 선택
    pts = np.array(eps)
    maxd, p1, p2 = -1, None, None
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.sum((pts[i] - pts[j])**2)
            if d > maxd:
                maxd, p1, p2 = d, tuple(pts[i]), tuple(pts[j])
    endpoints = [(p1[0]+x1, p1[1]+y1), (p2[0]+x1, p2[1]+y1)]

    # 홀 매핑
    mapped = []
    for ex, ey in endpoints:
        dists = [((ex-hx)**2 + (ey-hy)**2, idx) for idx, (hx, hy) in enumerate(holes)]
        d2, idx = min(dists)
        if np.sqrt(d2) <= detector.max_hole_dist:
            mapped.append(idx)
        else:
            mapped.append(None)

    return {'endpoints': endpoints, 'holes': mapped}

# 1️⃣ Detector 초기화
detector = LedEndpointDetector(visualize=False)

# 2️⃣ 현재 폴더의 'led'가 포함된 이미지 파일 검색
img_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))
              and 'led' in f.lower()]

if not img_files:
    print("led가 포함된 이미지 파일이 없습니다.")
else:
    print(f"다음 이미지들에서 LED 끝점 및 중간 결과를 시각화합니다: {img_files}")

    for fname in img_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"파일 '{fname}'은(는) 열 수 없습니다.")
            continue

        # 3️⃣ LED 검출 영역 (전체 이미지)
        height, width = img.shape[:2]
        bbox = (0, 0, width, height)

        # 4️⃣ 테스트용으로 holes는 비워둡니다
        holes = []

        # 5️⃣ 중간 이미지 저장 + extract
        out_prefix = os.path.splitext(fname)[0]
        result = preprocess_and_extract(detector, img, bbox, holes, out_prefix)

        if result is None:
            print(f"[{fname}] LED 끝점을 찾지 못했습니다.")
        else:
            print(f"[{fname}] 검출된 끝점: {result['endpoints']}")

            # 결과 그리기 및 저장
            detector.draw(img, result, holes)
            out_name = f"result_{fname}"
            cv2.imwrite(out_name, img)
            print(f"[{fname}] 결과를 '{out_name}'으로 저장했습니다.")
