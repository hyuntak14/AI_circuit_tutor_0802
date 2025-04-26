# test_led_detector_interactive.py

import cv2
import numpy as np
from skimage.morphology import skeletonize
from hole_detector import HoleDetector
from new_led_detector import ImprovedLedEndpointDetector

def nothing(x):
    pass

def main():
    img_path = 'led1.jpg'  # 테스트용 이미지 경로
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {img_path}")

    # 1) 브레드보드 구멍 검출
    hole_det = HoleDetector()
    holes = hole_det.detect_holes(img)
    print(f"검출된 홀 개수: {len(holes)}")

    # 2) ROI 선택 (LED 바운딩 박스)
    r = cv2.selectROI('Select LED ROI', img, showCrosshair=True, fromCenter=False)
    x1, y1, w, h = map(int, r)
    cv2.destroyWindow('Select LED ROI')
    if w == 0 or h == 0:
        print("ROI가 선택되지 않았습니다.")
        return
    bbox = (x1, y1, x1 + w, y1 + h)

    # 3) 트랙바 창 생성
    cv2.namedWindow('Params', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('BlockSize','Params',25,50,nothing)      # 실제 blockSize = val*2+1
    cv2.createTrackbar('C','Params',50,100,nothing)             # 실제 C = val - 50
    cv2.createTrackbar('CannyMin','Params',50,255,nothing)
    cv2.createTrackbar('CannyMax','Params',150,255,nothing)
    cv2.createTrackbar('HoughTh','Params',50,200,nothing)
    cv2.createTrackbar('MinLen','Params',20,200,nothing)
    cv2.createTrackbar('MaxGap','Params',5,50,nothing)
    cv2.createTrackbar('MaskRad','Params',5,30,nothing)

    # 4) 디텍터 초기화 (파라미터는 루프 안에서 덮어씌웁니다)
    led_det = LedEndpointDetector(visualize=False)

    while True:
        # 트랙바 값 읽기
        bs = cv2.getTrackbarPos('BlockSize','Params')*2 + 1
        c  = cv2.getTrackbarPos('C','Params') - 50
        cn_min = cv2.getTrackbarPos('CannyMin','Params')
        cn_max = cv2.getTrackbarPos('CannyMax','Params')
        h_th    = cv2.getTrackbarPos('HoughTh','Params')
        ml      = cv2.getTrackbarPos('MinLen','Params')
        mg      = cv2.getTrackbarPos('MaxGap','Params')
        mr      = cv2.getTrackbarPos('MaskRad','Params')

        # 디텍터 파라미터 업데이트
        led_det.adapt_block    = bs
        led_det.adapt_C        = c
        led_det.hough_th       = h_th
        led_det.min_skel_area  = 20    # 고정
        led_det.max_hole_dist  = 15    # 고정
        led_det.hole_mask_radius = mr

        # ROI 잘라내서 전처리
        x1, y1, x2, y2 = bbox
        roi = img[y1:y2, x1:x2]
        gray, gamma, bw = led_det._preprocess(roi)

        # 이진화에 blockSize/C 반영
        # (내부 _preprocess가 읽은 adapt_block, adapt_C로 반영됨)

        # 1) Binary → 마스킹
        bw_mask = bw.copy()
        for hx, hy in holes:
            if x1 <= hx < x2 and y1 <= hy < y2:
                mx, my = hx - x1, hy - y1
                cv2.circle(bw_mask, (mx, my), mr, 0, -1)

        # 2) Skeleton
        skel = skeletonize(bw_mask//255).astype(np.uint8)*255

        # 3) Canny
        edges = cv2.Canny(gray, cn_min, cn_max)

        # 4) HoughLinesP
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            h_th, minLineLength=ml, maxLineGap=mg
        )
        vis_hough = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for x1l,y1l,x2l,y2l in lines[:,0]:
                cv2.line(vis_hough, (x1l,y1l), (x2l,y2l), (0,0,255), 1)

        # 5) 최종 추출
        result = led_det.extract(img, bbox, holes)

        # 6) 화면 표시
        cv2.imshow('1.Gray (CLAHE)', gray)
        cv2.imshow('2.Gamma', gamma)
        cv2.imshow('3.Binary', bw)
        cv2.imshow('4.Masked Binary', bw_mask)
        cv2.imshow('5.Skeleton', skel)
        cv2.imshow('6.Edges', edges)
        cv2.imshow('7.Hough Lines', vis_hough)

        # 최종 결과
        vis_final = img.copy()
        cv2.rectangle(vis_final, (x1,y1), (x2,y2), (0,255,255), 2)
        for (hx,hy) in holes:
            cv2.circle(vis_final, (hx,hy), 3, (0,255,0), -1)
        led_det.draw(vis_final, result, holes)
        cv2.imshow('8.Final Result', vis_final)

        # ESC 누르면 종료
        if cv2.waitKey(100) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
