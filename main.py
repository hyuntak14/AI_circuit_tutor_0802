import os
import matplotlib
matplotlib.use('Qt5Agg')   # 또는 'Qt5Agg'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from sklearn.cluster import DBSCAN
from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.hole_detector import HoleDetector
from detector.resistor_detector import ResistorEndpointDetector
from detector.led_detector import LedEndpointDetector
from detector.wire_detector import WireDetector
from ui.perspective_editor import select_and_transform
from circuit_generator import generate_circuit
from detector.diode_detector import ResistorEndpointDetector as DiodeEndpointDetector
from detector.ic_chip_detector import ICChipPinDetector
from checker.error_checker import ErrorChecker

import random
WINDOW = 'AR Tutor'

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

def visualize_component_nets(img, component_pins, hole_to_net, parent, find):
    """
    • img: 회로판 원본(BGR)
    • component_pins: [{'class','box','pins',…}, …]
    • hole_to_net: {(x,y): raw_net_id, …}
    • parent, find: Union-Find 자료구조 함수
    """
    import cv2
    import numpy as np

    # 1) Net ID별 고유 색상 생성
    #    (Union-Find으로 최종 병합된 ID set)
    final_nets = set(find(n) for n in hole_to_net.values())
    rng = np.random.default_rng(1234)
    net_colors = {
        net_id: tuple(int(c) for c in rng.integers(0,256,3))
        for net_id in final_nets
    }

    overlay = img.copy()
    # 2) 각 컴포넌트 핀마다, nearest_net → 최종 Net ID 찾고, 원 그리기
    for comp in component_pins:
        for pt in comp['pins']:
            # raw 위치(pt)와 hole_to_net 키 중 최소거리로 매핑
            closest = min(
                hole_to_net.keys(),
                key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2
            )
            raw_net = hole_to_net[closest]
            net_id = find(raw_net)
            color = net_colors[net_id]
            cv2.circle(overlay, pt, 6, color, -1)
            # Net ID 텍스트 표시
            cv2.putText(overlay, str(net_id), (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3) 반투명하게 합성
    alpha = 0.6
    out = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    # 4) 창에 띄우기
    cv2.imshow('Component ↔ Net Mapping', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Component ↔ Net Mapping')


def visualize_cluster_connections(row_nets, component_pins):
    # 1) 전체 net_id 추출 & 색상 매핑
    net_ids = {entry['net_id']
               for _, clusters in row_nets
               for entry in clusters}
    colors = {nid: (random.random(), random.random(), random.random())
              for nid in net_ids}

    plt.figure(figsize=(8,6))
    # 2) 클러스터(구멍) 시각화
    for row_idx, clusters in row_nets:
        for entry in clusters:
            nid = entry['net_id']
            pts = entry['pts']              # pts: [(x,y), …]
            xs = [int(round(x)) for x,y in pts]
            ys = [int(round(y)) for x,y in pts]
            plt.scatter(xs, ys,
                        c=[colors[nid]],
                        s=20,
                        label=f'Net {nid}')

    # 3) 소자 핀 연결 시각화
    for comp in component_pins:
        name = comp['class']
        pins = comp['pins']               # [(x1,y1),(x2,y2)]
        xs = [p[0] for p in pins]
        ys = [p[1] for p in pins]
        plt.plot(xs, ys,
                 marker='x',
                 linestyle='-',
                 linewidth=1)
        plt.text(xs[0], ys[0], name,
                 fontsize=8, va='bottom')

    # 4) 축 뒤집기 & 범례
    plt.gca().invert_yaxis()    
    # 중복 레이블 제거
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    # 1) 파일로 저장
    out_path = 'cluster_connections.png'
    plt.savefig(out_path, dpi=200)
    plt.close()

    img = cv2.imread(out_path)
    if img is not None:
        cv2.imshow('Cluster Connections', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"이미지 로드 실패: {out_path}")


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
        cv2.imshow(WINDOW, canvas)
        

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, labels
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            tmp = canvas.copy()
            cv2.rectangle(tmp, (ix,iy), (x,y), (255,255,255), 1)
            cv2.imshow(WINDOW, tmp)
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


    cv2.setMouseCallback(WINDOW, mouse_cb)
    redraw()
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
        cv2.imshow(WINDOW, canvas)

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

    '''win = "Modify Detections"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)
    redraw()

    # 'q' 를 누를 때까지 대기
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(win)'''
    return detections



def main():
    
    component_pins = []
    root = tk.Tk()
    root.withdraw()

    # Detector 초기화
    detector     = FasterRCNNDetector(r'D:/Hyuntak/연구실/AR 회로 튜터/breadboard_project/model/fasterrcnn.pt')
    hole_det = HoleDetector(
    template_csv_path='detector/template_holes_complete.csv',
    template_image_path='detector/breadboard18.jpg',
    max_nn_dist=20.0
)

    wire_det     = WireDetector(kernel_size=4)
    resistor_det = ResistorEndpointDetector()
    led_det      = LedEndpointDetector(max_hole_dist=15, visualize=False)
    diode_det    = DiodeEndpointDetector()  # diode endpoints detector
    ic_det       = ICChipPinDetector()       # IC 칩 핀 위치 detector

    # 이미지 로드 및 브레드보드 검출
    img = imread_unicode(r'D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\breadboard99.jpg')
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

    cv2.namedWindow(WINDOW)

    final_labels = unified_labeler(vis_img, class_colors, auto_comps[:])
    all_comps = [(cls, 1.0, box) for cls, box in final_labels]
    all_comps = modify_detections(warped, all_comps)

    # 1) 구멍 좌표 검출
    holes = hole_det.detect_holes(warped_raw)
    # 2) 전체 넷 클러스터링
    nets, row_nets  = hole_det.get_board_nets(holes,base_img=warped_raw, show=False)

    # 2-1) 행별 그룹(cluster) 생성 (template alignment 적용된 points 기준)
    #hole_det.visualize_clusters(base_img=warped_raw,clusters=nets,affine_pts=holes )

    

    # hole_to_net 맵 생성
    hole_to_net = {}
    for row_idx, clusters in row_nets:
        for entry in clusters:
            net_id = entry['net_id']
            for x, y in entry['pts']:
                hole_to_net[(int(round(x)), int(round(y)))] = net_id


     # ─── 2) Union-Find 초기화 & net 색상 매핑 ────────────
    parent = { net_id: net_id for net_id in set(hole_to_net.values()) }
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    import numpy as np
    rng = np.random.default_rng(1234)
    final_nets = set(find(n) for n in hole_to_net.values())
    net_colors = {
        net_id: tuple(int(c) for c in rng.integers(0, 256, 3))
        for net_id in final_nets
    }



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
            x1, y1, x2, y2 = box  # 수정된 부분
            roi = warped_raw[y1:y2, x1:x2]
            ics = ic_det.detect(roi)
            if ics:
                det = ics[0]
                pins = [(x1 + px, y1 + py) for px, py in det['pin_points']]




            
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
        # (a) 컴포넌트 박스와 현재 핀 위치 그리기
        for comp in component_pins:
            x1,y1,x2,y2 = comp['box']
            color = class_colors.get(comp['class'], (0,255,255))
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            for px,py in comp['pins']:
                cv2.circle(img, (px,py), 5, (0,255,0), -1)

        # (b) net 매핑 오버레이
        overlay = img.copy()
        for comp in component_pins:
            for pt in comp['pins']:
                # 가장 가까운 hole_to_net 키 찾기
                closest = min(
                    hole_to_net.keys(),
                    key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2
                )
                raw_net = hole_to_net[closest]
                net_id = find(raw_net)
                c = net_colors.get(net_id, (255,255,255))
                cv2.circle(overlay, pt, 8, c, -1)
                cv2.putText(overlay, str(net_id),
                            (pt[0]+8, pt[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, c, 2)

        # (c) 블렌딩 후 화면에 띄우기
        alpha = 0.6
        blended = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        cv2.imshow(WINDOW, blended)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for comp in component_pins:
                x1,y1,x2,y2 = comp['box']
                if x1<=x<=x2 and y1<=y<=y2:
                    # 1) 기존 핀 삭제
                    comp['pins'] = []
                    redraw()
                    # 2) 수동 핀 선택
                    expected = 8 if comp['class']=='IC' else 2
                    new_pins = manual_pin_selection(orig, comp['box'], expected)
                    comp['pins'] = new_pins or []
                    # 3) 갱신된 핀+netz 다시 그림
                    redraw()
                    break

    #cv2.namedWindow('Fix Pins')
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #cv2.destroyWindow('Fix Pins')


    def nearest_net(pt):
        closest = min(hole_to_net.keys(),
                    key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2)
        return find(hole_to_net[closest])  # 이미 Union-Find 상에서 병합된 global net_id 반환

    parent = { net: net for net in set(hole_to_net.values()) }
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    wires = []
    for comp in component_pins:
        if comp['class']=='Line_area' and len(comp['pins'])==2:
            # 두 endpoint를 nearest_net 로 매핑
            net1 = nearest_net(comp['pins'][0])
            net2 = nearest_net(comp['pins'][1])
            # 서로 다른 넷 사이만 연결 정보로 추가
            if net1 != net2:
                wires.append((net1, net2))

    # 최종 수정 및 회로도 생성
    dets = [(comp['class'], 1.0, comp['box']) for comp in component_pins]
    new_dets = modify_detections(warped, dets)
    for comp, (_,_,box) in zip(component_pins, new_dets):
        comp['box'] = box

    
    # ———————————————— 전원 단자 클릭 입력 ————————————————
    from diagram import get_n_clicks
    # ———— ① 모든 엔드포인트 수집 ————
    # component_pins 안의 'pins' 리스트(와이어 끝점, 소자 핀 위치)를 모두 합칩니다.
    all_endpoints = [
        pt
        for comp in component_pins
        for pt   in comp['pins']
    ]

    # ———— ② UI로 필요한 전원 소스 개수 입력받기 ————
    root = tk.Tk()
    root.withdraw()
    source_count = simpledialog.askinteger(
        "전원 개수 입력",
        "필요한 전원 소스 수를 입력하세요:",
        minvalue=1, initialvalue=1
    ) or 1
    root.destroy()

    # ———— ③ source_count에 맞춰 +/– 클릭 프롬프트 리스트 생성 ————
    prompts = []
    for i in range(1, source_count + 1):
        prompts.append(f"Click + terminal of source {i}")
        prompts.append(f"Click - terminal of source {i}")

    # ———— ④ 다중 클릭으로 좌표 수집 ————
    all_pts = get_n_clicks(
        warped_raw,
        WINDOW,
        prompts
    )

    # ———— ⑤ 짝수 인덱스 → plus_pts, 홀수 인덱스 → minus_pts ————
    plus_pts  = all_pts[0::2]
    minus_pts = all_pts[1::2]

    power_pairs = []
    img_w = warped_raw.shape[1]
    comp_count = len([c for c in component_pins if c['class'] != 'Line_area'])
    grid_width = comp_count * 2 + 2

    for plus_pt, minus_pt in zip(plus_pts, minus_pts):
        # 클릭한 위치에서 가장 가까운 실제 엔드포인트 찾기
        closest_plus = min(all_endpoints, key=lambda p: (p[0]-plus_pt[0])**2 + (p[1]-plus_pt[1])**2)
        closest_minus = min(all_endpoints, key=lambda p: (p[0]-minus_pt[0])**2 + (p[1]-minus_pt[1])**2)

        # 네트워크 ID 매핑
        net_plus = nearest_net(closest_plus)
        net_minus = nearest_net(closest_minus)

        # schemdraw용 그리드 좌표로 변환
        x_plus_grid = closest_plus[0]  / img_w * grid_width
        x_minus_grid = closest_minus[0] / img_w * grid_width

        # 리스트에 추가
        power_pairs.append((net_plus, x_plus_grid, net_minus, x_minus_grid))

        for i, (net_p, x_p, net_m, x_m) in enumerate(power_pairs, start=1):
            component_pins.append({
                'class': 'VoltageSource',
                'pins': [],  # 좌표는 사용되지 않으므로 빈 리스트로 둠
                'value': voltage,
                'box': (0,0,0,0),  # 임의값
                'num_idx': 100+i,  # 충돌 피하기 위해 임의 인덱스
                'name': f'V{i}'
            })
        # 전원 소자도 component_pins에 추가 (V+, V- 클래스로)


        # 전원 소자도 component_pins에 추가 (단일 VoltageSource)

        # 시각화 (디버그용 — 필요시 생략 가능)
        cv2.circle(warped_raw, plus_pt,        5, (0,0,255),  -1)
        cv2.circle(warped_raw, closest_plus,   7, (255,0,0),  -1)
        cv2.line(  warped_raw, plus_pt, closest_plus, (0,255,0), 2)

        cv2.circle(warped_raw, minus_pt,       5, (0,0,255),  -1)
        cv2.circle(warped_raw, closest_minus,  7, (255,0,0),  -1)
        cv2.line(  warped_raw, minus_pt, closest_minus, (0,255,0), 2)

    cv2.imshow(WINDOW, warped_raw)
    cv2.waitKey(0)
    #cv2.destroyWindow(WINDOW)

    # ———— ③ 네트워크 ID로 매핑 ————
    net_plus  = nearest_net(closest_plus)
    net_minus = nearest_net(closest_minus)

    # ———— ④ schemdraw 그리드 x 좌표 변환 ————
    img_w      = warped_raw.shape[1]
    # Line_area 는 제외한 컴포넌트 수
    comp_count = len([c for c in component_pins if c['class']!='Line_area'])
    grid_width = comp_count * 2 + 2
    x_plus_grid  = closest_plus[0]  / img_w * grid_width
    x_minus_grid = closest_minus[0] / img_w * grid_width

    #클래스명 변경(generate_circuit에 맞게게)
  

    ## ———————————————— 회로도 생성 (전원 위치 넘김) ————————————————
    components, nets = generate_circuit(
        component_pins,
        holes, wires,
        voltage,
        'circuit.spice',
        'circuit.jpg',
        hole_to_net,
        power_pairs
    )

        # 2) hole_to_net → net_id: [comp_name,…] 형태로 역색인 생성
    nets_mapping = {}
    for comp in components:
        n1, n2 = comp['nodes']
        nets_mapping.setdefault(n1, []).append(comp['name'])
        nets_mapping.setdefault(n2, []).append(comp['name'])

    # 3) power_pairs(plus/minus net) 정보를 이용해 VoltageSource 컴포넌트 추가
    #    (power_pairs 는 사용자가 클릭해서 얻은 리스트: [(net_p, x_p, net_m, x_m), …])
    for i, (net_p, _, net_m, _) in enumerate(power_pairs, start=1):
        vs_name = f"V{i}"
        vs_comp = {
            'name': vs_name,
            'class': 'VoltageSource',
            'value': voltage,       # 사용자 입력 전압 변수
            'nodes': (net_p, net_m)
        }
        components.append(vs_comp)
        nets_mapping.setdefault(net_p, []).append(vs_name)
        nets_mapping.setdefault(net_m, []).append(vs_name)

    # 4) ground_net 을 minus 단자(net_m) 으로 지정
    ground_net = power_pairs[0][2]

    # 5) 수정된 mapping 으로 ErrorChecker 생성
    checker = ErrorChecker(components, nets_mapping, ground_nodes={ground_net})
    errors = checker.run_all_checks()
    if errors:
        print("=== Wiring Errors Detected ===")
        for e in errors:
            print("·", e)
    else:
        print("No wiring errors detected.")


    for comp in component_pins:
        print(f"{comp['class']} @ {comp['box']} → pins={comp['pins']}, value={comp['value']}Ω")

    

if __name__ == '__main__':
    main()
