import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog

from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.hole_detector import HoleDetector
from detector.resistor_detector import ResistorEndpointDetector
from detector.led_detector import LedEndpointDetector
from detector.wire_detector import WireDetector
from ui.perspective_editor import select_and_transform
from ui.manual_labeler import draw_and_label, choose_class_gui
from circuit_generator import generate_circuit

# 소자별 색상 (data.yaml 기준)
class_colors = {
    'Breadboard': (0, 128, 255),
    'Capacitor':  (255, 0, 255),
    'Diode':      (0, 255, 0),
    'IC':         (204, 102, 255),
    'LED':        (102, 0, 102),
    'Line_area':  (255, 0, 0),
    'Resistor':   (200, 170, 0)
}

def imread_unicode(path):
    """유니코드 경로 지원을 위한 이미지 로드 함수"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def modify_detections(image, detections):
    # (기존 modify_detections 구현 유지)
    return detections


def main():
    root = tk.Tk()
    root.withdraw()

    # 1) 모델 및 검출기 초기화
    detector     = FasterRCNNDetector(r'D:/Hyuntak/연구실/AR 회로 튜터/breadboard_project/model/fasterrcnn.pt')
    hole_det     = HoleDetector()
    wire_det     = WireDetector(kernel_size=4)
    resistor_det = ResistorEndpointDetector()
    led_det      = LedEndpointDetector(max_hole_dist=15, visualize=False)

    # 2) 이미지 로드 및 브레드보드 영역 투영
    img = imread_unicode(r'D:/Hyuntak/연구실/AR 회로 튜터/개발/breadboard6.jpg')
    comps = detector.detect(img)
    bb = next((b for c,_,b in comps if c.lower()=='breadboard'), None)
    if bb is None:
        raise ValueError('Breadboard 미검출')
    warped, _  = select_and_transform(img.copy(), bb)
    warped_raw = warped.copy()

    # 3) 컴포넌트 검출 + 수동 라벨링
    auto      = [c for c in detector.detect(warped) if c[0].lower()!='breadboard']
    manual    = draw_and_label(warped)
    all_comps = [(cls,conf,box) for cls,conf,box in auto] \
                + [(cls,1.0,box) for cls,box in manual]
    all_comps = modify_detections(warped, all_comps)

    # 4) 구멍 검출
    holes = hole_det.detect_holes(warped_raw)

    # 5) 와이어 검출 및 플러그된 위치 파악
    segments = wire_det.detect_wires(warped_raw)
    wire_connections = []
    for color, seg in segments.items():
        eps = seg.get('endpoints', [])
        # 유효한 두 개의 끝점이 있는 경우
        if wire_det.is_good_endpoints(eps):
            # 가장 먼 두 점(farthest-pair) 선택
            max_d = 0
            pair = (eps[0], eps[0])
            for i in range(len(eps)):
                for j in range(i+1, len(eps)):
                    dx = eps[i][0] - eps[j][0]
                    dy = eps[i][1] - eps[j][1]
                    d = dx*dx + dy*dy
                    if d > max_d:
                        max_d, pair = d, (eps[i], eps[j])
            # 엔드포인트 좌표를 가장 가까운 구멍 위치에 매핑
            conn = []
            for pt in pair:
                distances = [(h, (pt[0]-h[0])**2 + (pt[1]-h[1])**2) for h in holes]
                nearest = min(distances, key=lambda x: x[1])[0]
                conn.append(nearest)
            wire_connections.append({'color':color, 'holes':conn})
    # 와이어가 연결된 구멍 쌍 리스트
    wires = [wc['holes'] for wc in wire_connections]
    print(f'Detected wire connections (holes): {wires}')

    # 6) 소자 핀 추출
    component_pins = []
    for cls, conf, box in all_comps:
        if cls == 'Resistor':
            pins = resistor_det.extract(warped, box)
            resistor_det.draw(warped_raw, pins)
        elif cls == 'LED':
            pins = led_det.extract(warped, box, holes)
            led_det.draw(warped_raw, pins, holes)
        else:
            continue
        component_pins.append({'class':cls, 'box':box, 'pins':pins})

    # 7) 전압 및 소자값 입력
    voltage = simpledialog.askfloat("전압 입력", "전원 전압을 입력하세요 (V):", initialvalue=10.0) or 10.0
    for comp in component_pins:
        comp['value'] = simpledialog.askfloat(
            f"{comp['class']} 값 입력",
            f"{comp['class']} 값을 Ω 단위로 입력하세요:",
            initialvalue=100.0
        ) or 0.0

    # 8) 회로 생성 호출
    generate_circuit(
        r'D:/Hyuntak/연구실/AR 회로 튜터/개발/breadboard6.jpg',
        component_pins,
        holes,
        wires,
        voltage,
        'circuit.spice',
        'circuit.jpg'
    )

    # 9) 결과 시각화
    for comp in component_pins:
        print(f"{comp['class']} @ {comp['box']} → pins={comp['pins']}, value={comp['value']}Ω")

    cv2.imshow('Result', warped_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
