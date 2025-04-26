from ic_chip_detector import ICChipPinDetector
import cv2

# 1. 검출기 초기화
detector = ICChipPinDetector(gray_thresh=60, min_area=2000, visualize=True)

# 2. 이미지 로드 (브레드보드가 원근 보정된 상태여야 더욱 정확합니다)
img = cv2.imread('IC2.jpg')

# 3. IC 칩 및 핀 위치 검출
detections = detector.detect(img)

# 4. 결과 시각화
detector.draw(img, detections)
cv2.imshow('IC Pin Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. 검출된 핀 좌표 출력
for det in detections:
    print('IC 박스 좌표:', det['box'])
    print('핀 위치 8개:', det['pin_points'])
