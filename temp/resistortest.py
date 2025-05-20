import os
import cv2
import numpy as np
from resistor_detector import ResistorEndpointDetector
#from diode_detector import ResistorEndpointDetector
from led_detector import LedEndpointDetector

# 경로 설정: 테스트할 저항 이미지 파일 경로
IMAGE_PATH = r"resistor3.jpg"

# 1) 이미지 로드
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"이미지 로드 실패: {IMAGE_PATH}")

# 2) 바운딩 박스 수동 지정 (x1, y1, x2, y2)
#    전체 이미지에 저항만 담긴 경우, 전체 이미지 영역을 bbox로 사용 가능합니다.
h, w = image.shape[:2]
bbox = (0, 0, w, h)

# 3) 엔드포인트 디텍터 초기화 및 추출
detector = ResistorEndpointDetector(visualize=True)
endpoints = detector.extract(image, bbox)
if endpoints is None:
    print("엔드포인트를 검출하지 못했습니다.")
else:
    pt1, pt2 = endpoints
    print(f"검출된 엔드포인트: {pt1}, {pt2}")

    # 4) 시각화
    vis = image.copy()
    detector.draw(vis, endpoints, color=(0,255,0), radius=7)
    cv2.imshow("Resistor Endpoint Test", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
