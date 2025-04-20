import cv2
import numpy as np
from skimage.morphology import skeletonize

# 테스트할 저항 이미지와 bbox 설정
IMAGE_PATH = r"resistor4.jpg"
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"이미지 로드 실패: {IMAGE_PATH}")
h, w = image.shape[:2]
bbox = (0, 0, w, h)  # 전체 이미지를 ROI로 사용

# ROI 추출
x1, y1, x2, y2 = bbox
roi = image[y1:y2, x1:x2]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 빈 윈도우와 트랙바 생성
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controls', 400, 300)
# Canny
cv2.createTrackbar('Canny Low', 'Controls', 50, 255, lambda x: None)
cv2.createTrackbar('Canny High', 'Controls', 150, 255, lambda x: None)
# HoughLinesP
cv2.createTrackbar('Hough Thresh', 'Controls', 50, 200, lambda x: None)
cv2.createTrackbar('MinLen %', 'Controls', 50, 100, lambda x: None)  # % of width
cv2.createTrackbar('MaxGap', 'Controls', 10, 100, lambda x: None)
# Morphology
cv2.createTrackbar('Morph K', 'Controls', 3, 15, lambda x: None)
cv2.createTrackbar('Morph It', 'Controls', 1, 5, lambda x: None)
cv2.createTrackbar('MinSkel', 'Controls', 20, 200, lambda x: None)

while True:
    # 트랙바 값 읽기
    low = cv2.getTrackbarPos('Canny Low', 'Controls')
    high = cv2.getTrackbarPos('Canny High', 'Controls')
    h_th = cv2.getTrackbarPos('Hough Thresh', 'Controls')
    min_len = cv2.getTrackbarPos('MinLen %', 'Controls') / 100.0
    max_gap = cv2.getTrackbarPos('MaxGap', 'Controls')
    k = cv2.getTrackbarPos('Morph K', 'Controls') or 1
    iters = cv2.getTrackbarPos('Morph It', 'Controls')
    min_skel = cv2.getTrackbarPos('MinSkel', 'Controls')

    # 1) 이진화 + 모폴로지
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=iters)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=iters)

    # 2) 스켈레톤 + 작은 객체 제거
    skel = skeletonize(bw//255).astype(np.uint8)*255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel)
    clean = np.zeros_like(skel)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_skel:
            clean[labels==i] = 255
    skel = clean

    # 3) 엔드포인트 수집
    endpoints = []
    hh, ww = skel.shape
    for yy in range(1, hh-1):
        for xx in range(1, ww-1):
            if skel[yy, xx] and np.sum(skel[yy-1:yy+2, xx-1:xx+2])//255 - 1 == 1:
                endpoints.append((xx, yy))

    # 4) Canny + HoughLinesP
    edges = cv2.Canny(bw, low, high)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, h_th,
                            minLineLength=int(ww*min_len),
                            maxLineGap=max_gap)
    hough_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    pts = []
    if lines is not None:
        for x3,y3,x4,y4 in lines.reshape(-1,4):
            cv2.line(hough_img, (x3,y3), (x4,y4), (255,0,0), 1)
            pts.extend([(x3,y3),(x4,y4)])

    # 5) 최장 거리 2점 계산
    data_pts = np.array(endpoints) if len(endpoints)>=2 else (np.array(pts) if len(pts)>=2 else None)
    centers = []
    if data_pts is not None:
        max_d = -1
        for i in range(len(data_pts)):
            for j in range(i+1, len(data_pts)):
                d = np.sum((data_pts[i]-data_pts[j])**2)
                if d>max_d:
                    max_d = d; p1, p2 = tuple(data_pts[i]), tuple(data_pts[j])
        centers = [p1, p2]

    # 시각화 그리드 구성
    def bgr(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim==2 else img.copy()
    viz = [cv2.resize(roi,(ww,hh)), bgr(bw), bgr(skel), bgr(edges),
           cv2.resize(hough_img,(ww,hh)), bgr(gray)]
    # 엔드포인트 & 중심
    ep_img = viz[3].copy(); cen_img = viz[5].copy()
    for ex,ey in endpoints: cv2.circle(ep_img,(ex,ey),3,(0,0,255),-1)
    for cx,cy in centers: cv2.circle(cen_img,(cx,cy),5,(0,255,0),-1)
    viz[3], viz[5] = ep_img, cen_img
    grid = np.vstack([np.hstack(viz[:3]), np.hstack(viz[3:])])

    cv2.imshow('Process Overview', grid)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
