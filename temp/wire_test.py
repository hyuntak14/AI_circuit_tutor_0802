import cv2
import matplotlib.pyplot as plt
import sys

# detector 패키지가 있는 breadboard_project 상위 폴더를 추가
sys.path.append(r'd:\Hyuntak\연구실\AR 회로 튜터\breadboard_project')

from wire_detector import WireDetector

def test_wire_detector(img_path):
    # 전체 이미지 로드
    full_img = cv2.imread(img_path)
    if full_img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

    cropped = full_img.copy()
    detector = WireDetector(kernel_size=5)
    # configure_white_thresholds 우회
    detector.configure_white_thresholds = lambda img: setattr(
        detector, 'full_white_mask',
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    )
    detector.full_white_mask = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

    # 전체 영역 bbox
    h, w = full_img.shape[:2]
    bbox = (0, 0, w, h)
    white_mask = detector.extract_white_wire_mask(full_img, bbox)

    # black/red wire mask 및 skeleton 계산
    wire_segments = detector.detect_wires(cropped)
    endpoints, channel = detector.select_best_endpoints(wire_segments)

    # -- 시각화: 1~4 --
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title('Cropped Image')
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('White Wire Mask')
    plt.imshow(white_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Black Wire Mask')
    plt.imshow(wire_segments['black']['mask'], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title(f"Skeleton & Endpoints ({channel})")
    skel = wire_segments[channel]['skeleton']
    plt.imshow(skel, cmap='gray')
    for (x, y) in endpoints:
        plt.scatter(x, y, s=50, c='red', marker='o')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # -- 최종 검출 결과: 원본에 오버레이 --
    overlay = full_img.copy()
    for (x, y) in endpoints:
        cv2.circle(overlay, (x, y), 8, (0, 0, 255), 2)
    cv2.putText(overlay, f"Channel: {channel}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # OpenCV 윈도우로 보기
    cv2.imshow("Final Detection Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_path = "wire2.jpg"
    test_wire_detector(img_path)
