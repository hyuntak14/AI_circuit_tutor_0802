import cv2
import numpy as np
import sys

# detector 패키지가 있는 최상위 폴더 경로 설정
sys.path.append(r'd:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\temp')

from wire_detector import WireDetector

def nothing(x):
    pass

def run_interactive(img_path):
    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")
    h, w = img.shape[:2]

    # Settings 윈도우와 트랙바 생성
    cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Settings', 400, 200)
    cv2.createTrackbar('Kernel Size', 'Settings', 3, 15, nothing)
    cv2.createTrackbar('Endpoint eps', 'Settings', 20, 100, nothing)
    cv2.createTrackbar('L Threshold', 'Settings', 160, 255, nothing)
    cv2.createTrackbar('Canny Low', 'Settings', 20, 100, nothing)
    cv2.createTrackbar('Canny High', 'Settings', 80, 200, nothing)

    while True:
        # 트랙바 값 읽기
        k = cv2.getTrackbarPos('Kernel Size', 'Settings')
        eps = cv2.getTrackbarPos('Endpoint eps', 'Settings')
        l_thresh = cv2.getTrackbarPos('L Threshold', 'Settings')
        canny_low = cv2.getTrackbarPos('Canny Low', 'Settings')
        canny_high = cv2.getTrackbarPos('Canny High', 'Settings')

        # kernel_size는 홀수, 최소 1
        if k < 1: k = 1
        if k % 2 == 0: k += 1
        # Canny 범위 보정
        if canny_high < canny_low + 1:
            canny_high = canny_low + 1

        # WireDetector 초기화
        detector = WireDetector(kernel_size=k, endpoint_eps=eps)

        # custom extract_white_wire_mask 적용
        def custom_extract(image, bbox):
            x1, y1, x2, y2 = bbox
            roi = image[y1:y2, x1:x2]
            # LAB 기반 마스크
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            _, l_mask = cv2.threshold(l_channel, l_thresh, 255, cv2.THRESH_BINARY)
            ab_mask = cv2.inRange(
                cv2.merge([a_channel, b_channel]),
                np.array([120, 120]), np.array([140, 150])
            )
            ab_mask = cv2.bitwise_not(ab_mask)
            mask = cv2.bitwise_and(l_mask, ab_mask)
            # Canny 엣지 추가
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, canny_low, canny_high)
            mask = cv2.bitwise_or(mask, edges)
            # 모폴로지 정리
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, detector.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, detector.kernel)
            return detector.remove_holes(mask, roi)

        detector.extract_white_wire_mask = custom_extract

        # configure_white_thresholds 우회 (흰 마스크가 있어야 내부 호출 방지)
        detector.configure_white_thresholds = lambda img: None
        detector.full_white_mask = np.zeros((h, w), dtype=np.uint8)

        # 전체 영역 흰 전선 마스크
        white_mask = detector.extract_white_wire_mask(img, (0, 0, w, h))

        # Black/Red wire 검출
        segs = detector.detect_wires(img)
        endpoints, channel = detector.select_best_endpoints(segs)

        # 결과 오버레이 준비
        overlay = img.copy()
        mask_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(overlay, 0.7, mask_bgr, 0.3, 0)

        # 엔드포인트 그리기 (white 채널 기준)
        for (x, y) in endpoints:
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)

        # 파라미터 정보 표시
        info = f"K={k}, eps={eps}, Lth={l_thresh}, Cny[{canny_low},{canny_high}], Ch={channel}"
        cv2.putText(overlay, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 화면 출력
        cv2.imshow('Result', overlay)

        # 'q' 누르면 종료
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_path = 'wire1.jpg'
    run_interactive(img_path)
