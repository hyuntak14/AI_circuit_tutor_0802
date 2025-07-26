import os
import cv2
import numpy as np
from resistor_detector import ResistorEndpointDetector
#from diode_detector import ResistorEndpointDetector
from led_detector import LedEndpointDetector
from Improved_resistor_det import ImprovedResistorEndpointDetector

# 현재 경로에서 'resistor'가 포함된 파일명 찾기
current_dir = os.getcwd()
files = [f for f in os.listdir(current_dir) if "resistor" in f.lower() and f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not files:
    print("resistor가 포함된 이미지 파일이 없습니다.")
else:
    # 엔드포인트 디텍터 초기화 (한 번만 생성)
    detector = ResistorEndpointDetector(visualize=True)

    for file_name in files:
        print(f"처리 중: {file_name}")
        image = cv2.imread(file_name)
        if image is None:
            print(f"이미지 로드 실패: {file_name}")
            continue

        # 전체 이미지 영역을 bbox로 사용
        h, w = image.shape[:2]
        bbox = (0, 0, w, h)

        # 엔드포인트 추출
        endpoints = detector.extract(image, bbox)
        if endpoints is None:
            print(f"{file_name}: 엔드포인트를 검출하지 못했습니다.")
            continue

        pt1, pt2 = endpoints
        print(f"{file_name}: 검출된 엔드포인트: {pt1}, {pt2}")

        # 시각화
        vis = image.copy()
        detector.draw(vis, endpoints, color=(0,255,0), radius=7)
        cv2.imshow(f"Resistor Endpoint: {file_name}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
