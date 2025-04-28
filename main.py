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
from circuit_generator import generate_circuit
from detector.diode_detector import ResistorEndpointDetector as DiodeEndpointDetector
from detector.ic_chip_detector import ICChipPinDetector

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
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def manual_pin_selection(image, box, expected_count):
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2].copy()
    pins = []
    win = 'Manual Pin Selection'
    def mouse_cb(event, x, y, flags, param):
        nonlocal pins
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pins) < expected_count:
                pins.append((x1 + x, y1 + y))
                cv2.circle(roi, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(win, roi)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # remove last
            if pins:
                pins.pop()
                # redraw
                temp = image[y1:y2, x1:x2].copy()
                for px, py in pins:
                    cv2.circle(temp, (px - x1, py - y1), 5, (0, 0, 255), -1)
                roi[:] = temp

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)
    cv2.imshow(win, roi)
    while True:
        if len(pins) == expected_count:
            break
        if cv2.waitKey(1) == 27:  # ESC to cancel
            break
    cv2.destroyWindow(win)
    return pins


def unified_labeler(image, class_colors, initial_labels=None):
    root = tk.Tk()
    root.withdraw()
    win = 'Label Editor'
    labels = initial_labels.copy() if initial_labels else []
    drawing = False
    ix, iy = -1, -1
    canvas = image.copy()
    class_list = list(class_colors.keys())

    def redraw():
        nonlocal canvas
        canvas = image.copy()
        for cls, (x1,y1,x2,y2) in labels:
            color = class_colors.get(cls, (0,255,255))
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 2)
            cv2.putText(canvas, cls, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow(win, canvas)

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, labels
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            tmp = canvas.copy()
            cv2.rectangle(tmp, (ix,iy), (x,y), (255,255,255), 1)
            cv2.imshow(win, tmp)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1_, x2_ = sorted([ix, x])
            y1_, y2_ = sorted([iy, y])
            if abs(x2_-x1_) < 10 or abs(y2_-y1_) < 10:
                redraw()
                return
            prompt = "클래스를 선택하세요:\n" + \
                     "\n".join(f"{i+1}. {c}" for i,c in enumerate(class_list))
            idx = simpledialog.askinteger(
                "새 클래스 선택", prompt,
                minvalue=1, maxvalue=len(class_list)
            )
            if idx:
                cls = class_list[idx-1]
                labels.append((cls, (x1_,y1_,x2_,y2_)))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (cls, (x1_,y1_,x2_,y2_)) in enumerate(labels):
                if x1_ <= x <= x2_ and y1_ <= y <= y2_:
                    action = simpledialog.askstring(
                        "삭제/수정",
                        f"'{cls}' 선택됨:\n'd'→삭제, 'r'→이름 변경"
                    )
                    if action == 'd':
                        labels.pop(i)
                    elif action == 'r':
                        new_name = simpledialog.askstring("이름 변경", "새 이름:")
                        if new_name:
                            labels[i] = (new_name, (x1_,y1_,x2_,y2_))
                    break
            redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)
    redraw()
    cv2.waitKey(0)
    cv2.destroyWindow(win)
    return labels
def modify_detections(image, detections):
    """
    • image: 원본 영상 (np.ndarray)
    • detections: [(cls, score, (x1,y1,x2,y2)), ...]
    
    자동 검출된 결과 위에, 
    • 좌클릭 드래그 → 새 박스 그리기 
    • 우클릭 클릭 → 클래스 목록에서 선택 후 추가
    • 'q' 키 누르면 완료
    """
    import cv2
    import tkinter as tk
    from tkinter import simpledialog

    # 전역 class_colors 사용
    class_list = list(class_colors.keys())

    # 복사본에 그리기
    canvas = image.copy()
    for cls, score, box in detections:
        x1,y1,x2,y2 = box
        color = class_colors.get(cls, (0,255,255))
        cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 2)
        cv2.putText(canvas, cls, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 드래그 변수
    drawing = False
    ix, iy = -1, -1

    def redraw():
        nonlocal canvas
        canvas = image.copy()
        for cls, score, (x1,y1,x2,y2) in detections:
            color = class_colors.get(cls, (0,255,255))
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 2)
            cv2.putText(canvas, cls, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow(win, canvas)

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, detections

        # 드래그 시작
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        # 드래그 중
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            tmp = canvas.copy()
            cv2.rectangle(tmp, (ix,iy), (x,y), (255,255,255), 1)
            cv2.imshow(win, tmp)

        # 드래그 종료 → 박스 확정 후 클래스 선택
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1_, x2_ = sorted([ix, x])
            y1_, y2_ = sorted([iy, y])
            if abs(x2_-x1_) < 10 or abs(y2_-y1_) < 10:
                redraw()
                return

            # tkinter 창 띄워서 클래스 선택
            root = tk.Tk()
            root.withdraw()
            prompt = "추가할 클래스 선택:\n" + \
                     "\n".join(f"{i+1}. {c}" for i,c in enumerate(class_list))
            idx = simpledialog.askinteger("Class 선택", prompt,
                                          minvalue=1, maxvalue=len(class_list))
            root.destroy()
            if idx:
                cls = class_list[idx-1]
                detections.append((cls, 1.0, (x1_, y1_, x2_, y2_)))
            redraw()

    win = "Modify Detections"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)
    redraw()

    # 'q' 를 누를 때까지 대기
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(win)
    return detections

def main():
    component_pins = []
    root = tk.Tk()
    root.withdraw()

    # Detector 초기화
    detector     = FasterRCNNDetector(r'D:/Hyuntak/연구실/AR 회로 튜터/breadboard_project/model/fasterrcnn.pt')
    hole_det     = HoleDetector()
    wire_det     = WireDetector(kernel_size=4)
    resistor_det = ResistorEndpointDetector()
    led_det      = LedEndpointDetector(max_hole_dist=15, visualize=False)
    diode_det    = DiodeEndpointDetector()  # diode endpoints detector
    ic_det       = ICChipPinDetector()       # IC 칩 핀 위치 detector

    # 이미지 로드 및 브레드보드 검출
    img = imread_unicode(r'D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\breadboard16.jpg')
    comps = detector.detect(img)
    bb = next((b for c,_,b in comps if c.lower()=='breadboard'), None)
    if bb is None:
        raise ValueError('Breadboard 미검출')
    warped, _  = select_and_transform(img.copy(), bb)
    warped_raw = warped.copy()

    # 객체 검출 및 라벨링
    detections = detector.detect(warped)
    auto_comps = [(cls, box) for cls,_,box in detections if cls.lower()!='breadboard']

    vis_img = warped.copy()
    for cls, box in auto_comps:
        color = class_colors.get(cls, (0,255,255))
        x1,y1,x2,y2 = box
        cv2.rectangle(vis_img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis_img, cls, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    final_labels = unified_labeler(vis_img, class_colors, auto_comps[:])
    all_comps = [(cls, 1.0, box) for cls, box in final_labels]
    all_comps = modify_detections(warped, all_comps)

    # 핀 검출 및 wire endpoints 처리
    holes = hole_det.detect_holes(warped_raw)

    for cls, _, box in all_comps:
        x1, y1, x2, y2 = box
        expected = 8 if cls == 'IC' else 2
        pins = []

        if cls == 'Resistor':
            pins = resistor_det.extract(warped, box)
        elif cls == 'LED':
            pins = led_det.extract(warped, box, holes)
        elif cls == 'Diode':
            pins = diode_det.extract(warped, box)
        elif cls == 'IC':
            roi = warped[y1:y2, x1:x2]
            ic_detections = ic_det.detect(roi)
            if ic_detections:
                pts = ic_detections[0]['pin_points']
                pins = [(x1 + px, y1 + py) for px, py in pts]
        elif cls == 'Line_area':
            # 선영역: WireDetector로 endpoints 2개 추출
            roi = warped_raw[y1:y2, x1:x2]
            segs = wire_det.detect_wires(roi)
            endpoints, _ = wire_det.select_best_endpoints(segs)
            if endpoints:
                pins = [(x1 + pt[0], y1 + pt[1]) for pt in endpoints]
        # 자동 검출 미달 시 None/tuple -> list 변환
        # 2) 튜플·None 체크 → 리스트로 정규화
        if pins is None:
            pins_list = []
        elif isinstance(pins, tuple):
           pins_list = list(pins)
        else:
            pins_list = pins

        # 3) 유효 좌표 검증: 개수 및 None 제거
        valid = (
            len(pins_list) == expected and
            all(isinstance(pt, tuple) and None not in pt for pt in pins_list)
        )
        if not valid:
            # 수동 입력으로 재보완 :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
            pins_list = manual_pin_selection(warped_raw, box, expected)

        pins = pins_list

        # 결과 시각화
        for pt in pins:
            cv2.circle(warped_raw, pt, 5, (0, 255, 0), -1)
        component_pins.append({'class':cls, 'box':box, 'pins':pins})

    # 인덱스 및 값 입력
    for idx, comp in enumerate(component_pins):
        comp['num_idx'] = idx+1

    voltage = simpledialog.askfloat("전압 입력", "전원 전압을 입력하세요 (V):", initialvalue=10.0) or 10.0
    for comp in component_pins:
        if comp['class'] == 'Resistor':
            comp['value'] = simpledialog.askfloat(
                "저항 값 입력",
                "저항 값을 Ω 단위로 입력하세요:",
                initialvalue=100.0
            ) or 0.0
        else:
            comp['value'] = 10  # 또는 원하는 기본값
    orig = warped.copy()           # “warped_raw” 가 아니라 핀 찍기 전 이미지
    warped_raw = orig.copy()       # 이후 임시 표시용

    # … (컴포넌트 핀 검출 및 pins_list 채우는 코드)

    # ② Fix Pins 윈도우의 베이스 이미지는 깔끔한 orig 로
    result_base = orig.copy()

    def redraw():
        img = result_base.copy()
        for comp in component_pins:
            x1,y1,x2,y2 = comp['box']
            color = class_colors.get(comp['class'], (0,255,255))
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            for px,py in comp['pins']:
                cv2.circle(img, (px,py), 5, (0,255,0), -1)
        cv2.imshow('Fix Pins', img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for comp in component_pins:
                x1,y1,x2,y2 = comp['box']
                if x1<=x<=x2 and y1<=y<=y2:
                    # (1) 기존 점 즉시 삭제 및 화면 반영
                    comp['pins'] = []
                    redraw()

                    # (2) 수동 입력
                    expected = 8 if comp['class']=='IC' else 2
                    new_pins = manual_pin_selection(orig, comp['box'], expected)

                    # (3) 새 점만 저장 및 재그리기
                    comp['pins'] = new_pins or []
                    redraw()
                    break

    cv2.namedWindow('Fix Pins')
    cv2.setMouseCallback('Fix Pins', on_mouse)
    redraw()
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Fix Pins')


    # 최종 수정 및 회로도 생성
    dets = [(comp['class'], 1.0, comp['box']) for comp in component_pins]
    new_dets = modify_detections(warped, dets)
    for comp, (_,_,box) in zip(component_pins, new_dets):
        comp['box'] = box

    generate_circuit(
        r'D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\breadboard16.jpg',
        component_pins,
        holes,
        [],  # wire connections
        voltage,
        'circuit.spice',
        'circuit.jpg'
    )

    for comp in component_pins:
        print(f"{comp['class']} @ {comp['box']} → pins={comp['pins']}, value={comp['value']}Ω")


if __name__ == '__main__':
    main()
