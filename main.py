import os
import matplotlib
matplotlib.use('Qt5Agg')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox

from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.hole_detector import HoleDetector
from detector.resistor_detector import ResistorEndpointDetector
from detector.led_detector import LedEndpointDetector
from detector.wire_detector import WireDetector
from detector.diode_detector import ResistorEndpointDetector as DiodeEndpointDetector
from detector.ic_chip_detector import ICChipPinDetector
from ui.perspective_editor import select_and_transform
from circuit_generator import generate_circuit
from checker.error_checker import ErrorChecker

<<<<<<< HEAD
class SimpleCircuitConverter:
    def __init__(self):
        self.detector = FasterRCNNDetector(r'D:/Hyuntak/lab/AR_circuit_tutor/breadboard_project/model/fasterrcnn.pt')
        self.hole_det = HoleDetector(
            template_csv_path='detector/template_holes_complete.csv',
            template_image_path='detector/breadboard18.jpg',
            max_nn_dist=20.0
=======
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
    detector     = FasterRCNNDetector(r'D:/Hyuntak/lab/AR_circuit_tutor/breadboard_project/model/fasterrcnn.pt')
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
    img = imread_unicode(r'D:\Hyuntak\lab\AR_circuit_tutor\breadboard_project\breadboard99.jpg')
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
    holes = hole_det.detect_holes(warped_raw,visualize_nets=True)
    # 2) 전체 넷 클러스터링
    #nets, row_nets  = hole_det.get_board_nets(holes,base_img=warped_raw, show=True)

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
>>>>>>> 43c99fd46a94b88ad6ea9a12de5345dc93c72d7d
        )
        self.wire_det = WireDetector(kernel_size=4)
        self.resistor_det = ResistorEndpointDetector()
        self.led_det = LedEndpointDetector(max_hole_dist=15, visualize=False)
        self.diode_det = DiodeEndpointDetector()
        self.ic_det = ICChipPinDetector()
        
        # 컴포넌트 색상
        self.class_colors = {
            'Breadboard': (0, 128, 255),
            'Capacitor': (255, 0, 255),
            'Diode': (0, 255, 0),
            'IC': (204, 102, 255),
            'LED': (102, 0, 102),
            'Line_area': (255, 0, 0),
            'Resistor': (200, 170, 0)
        }

    def load_image(self):
        """이미지 파일 선택 및 로드"""
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="브레드보드 이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()
        
        if not file_path:
            return None
            
        return cv2.imread(file_path)

    def auto_detect_and_transform(self, img):
        """자동 브레드보드 검출 및 변환"""
        print("🔍 브레드보드 자동 검출 중...")
        comps = self.detector.detect(img)
        bb = next((b for c, _, b in comps if c.lower() == 'breadboard'), None)
        
        if bb is None:
            print("❌ 브레드보드를 찾을 수 없습니다.")
            return None
            
        print("✅ 브레드보드 검출 완료")
        warped, _ = select_and_transform(img.copy(), bb)
        return warped

    def quick_component_detection(self, warped):
        """빠른 컴포넌트 검출 및 간단한 수정"""
        print("🔍 컴포넌트 자동 검출 중...")
        detections = self.detector.detect(warped)
        all_comps = [(cls, 1.0, box) for cls, _, box in detections if cls.lower() != 'breadboard']
        
        # 시각화
        vis_img = warped.copy()
        for i, (cls, _, box) in enumerate(all_comps):
            x1, y1, x2, y2 = box
            color = self.class_colors.get(cls, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{i+1}:{cls}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('검출된 컴포넌트 (Enter: 확인, Space: 수정모드)', vis_img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord(' '):  # Space키로 수정모드
            all_comps = self.manual_edit_components(warped, all_comps)
            
        print(f"✅ {len(all_comps)}개 컴포넌트 확인됨")
        return all_comps

    def manual_edit_components(self, warped, components):
        """간단한 수동 컴포넌트 편집"""
        print("🛠️ 수동 편집 모드")
        print("- 좌클릭: 새 컴포넌트 추가 (드래그)")
        print("- 우클릭: 컴포넌트 삭제")
        print("- 키보드 'd': 번호로 삭제")
        print("- Enter: 완료")
        
        editing = True
        drawing = False
        start_point = None
        window_name = '컴포넌트 편집'
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, components
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # 새 컴포넌트 추가 시작
                drawing = True
                start_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                # 드래그 중 임시 박스 표시
                temp_img = warped.copy()
                for i, (cls, _, box) in enumerate(components):
                    x1, y1, x2, y2 = box
                    color = self.class_colors.get(cls, (0, 255, 255))
                    cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(temp_img, f"{i+1}:{cls}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 임시 박스
                cv2.rectangle(temp_img, start_point, (x, y), (255, 255, 255), 2)
                cv2.imshow(window_name, temp_img)
                
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                # 새 컴포넌트 추가 완료
                drawing = False
                x1, x2 = sorted([start_point[0], x])
                y1, y2 = sorted([start_point[1], y])
                
                # 최소 크기 확인
                if abs(x2-x1) > 20 and abs(y2-y1) > 20:
                    # 컴포넌트 클래스 선택
                    root = tk.Tk()
                    root.withdraw()
                    
                    classes = ['Resistor', 'LED', 'Diode', 'IC', 'Line_area', 'Capacitor']
                    class_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(classes))
                    choice = simpledialog.askinteger(
                        "컴포넌트 선택", 
                        f"추가할 컴포넌트를 선택하세요:\n{class_str}",
                        minvalue=1, maxvalue=len(classes)
                    )
                    
                    if choice:
                        new_class = classes[choice-1]
                        components.append((new_class, 1.0, (x1, y1, x2, y2)))
                        print(f"✅ {new_class} 추가됨")
                    
                    root.destroy()
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # 클릭한 위치의 컴포넌트 찾기
                for i, (cls, _, box) in enumerate(components):
                    x1, y1, x2, y2 = box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        root = tk.Tk()
                        root.withdraw()
                        choice = simpledialog.askinteger(
                            "삭제/수정", 
                            f"1. 삭제\n2. 수정\n선택하세요 (1-2):",
                            minvalue=1, maxvalue=2
                        )
                        if choice == 1:
                            components.pop(i)
                            print(f"✅ {cls} 삭제됨")
                        elif choice == 2:
                            classes = ['Resistor', 'LED', 'Diode', 'IC', 'Line_area', 'Capacitor']
                            class_str = "\n".join(f"{j+1}. {c}" for j, c in enumerate(classes))
                            new_choice = simpledialog.askinteger(
                                "새 클래스 선택",
                                f"새 컴포넌트를 선택하세요:\n{class_str}",
                                minvalue=1, maxvalue=len(classes)
                            )
                            if new_choice:
                                new_class = classes[new_choice-1]
                                components[i] = (new_class, 1.0, box)
                                print(f"✅ {cls} → {new_class}로 변경됨")
                        root.destroy()
                        break

        
        # 먼저 윈도우를 생성하고 이미지를 표시
        vis_img = warped.copy()
        for i, (cls, _, box) in enumerate(components):
            x1, y1, x2, y2 = box
            color = self.class_colors.get(cls, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{i+1}:{cls}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow(window_name, vis_img)
        
        # 윈도우가 생성된 후 마우스 콜백 설정
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while editing:
            vis_img = warped.copy()
            for i, (cls, _, box) in enumerate(components):
                x1, y1, x2, y2 = box
                color = self.class_colors.get(cls, (0, 255, 255))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_img, f"{i+1}:{cls}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow(window_name, vis_img)
            key = cv2.waitKey(30) & 0xFF  # 30ms 대기로 변경
            
            if key == 13:  # Enter
                editing = False
            elif key == ord('d'):  # 'd' 키로 번호 삭제
                if components:  # 컴포넌트가 있을 때만
                    root = tk.Tk()
                    root.withdraw()
                    idx = simpledialog.askinteger("삭제", f"삭제할 컴포넌트 번호 (1-{len(components)}):")
                    if idx and 1 <= idx <= len(components):
                        removed = components.pop(idx-1)
                        print(f"✅ 컴포넌트 {idx} ({removed[0]}) 삭제됨")
                    root.destroy()

            elif key == ord('c'):  # 'c' 키로 번호별 클래스 변경
                if components:
                    root = tk.Tk()
                    root.withdraw()
                    idx = simpledialog.askinteger("클래스 변경", f"변경할 컴포넌트 번호 (1-{len(components)}):")
                    if idx and 1 <= idx <= len(components):
                        classes = ['Resistor', 'LED', 'Diode', 'IC', 'Line_area', 'Capacitor']
                        class_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(classes))
                        choice = simpledialog.askinteger(
                            "새 클래스 선택",
                            f"새 컴포넌트를 선택하세요:\n{class_str}",
                            minvalue=1, maxvalue=len(classes)
                        )
                        if choice:
                            new_class = classes[choice-1]
                            old_class = components[idx-1][0]
                            box = components[idx-1][2]
                            components[idx-1] = (new_class, 1.0, box)
                            print(f"✅ {idx}번 컴포넌트: {old_class} → {new_class} 변경됨")
                    root.destroy()

        
        cv2.destroyAllWindows()
        return components

    def auto_pin_detection(self, warped, components):
        """자동 핀 검출 (실패 시 기본값 사용)"""
        print("📍 컴포넌트 핀 자동 검출 중...")
        
        # 구멍 검출
        holes = self.hole_det.detect_holes(warped)
        print(f"✅ {len(holes)}개 구멍 검출됨")
        
        component_pins = []
        for i, (cls, _, box) in enumerate(components):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1  # 이 부분을 먼저 계산
            expected = 8 if cls == 'IC' else 2
            pins = []
            
            try:
                if cls == 'Resistor':
                    result = self.resistor_det.extract(warped, box)
                    if result and result[0] is not None and result[1] is not None:
                        pins = list(result)
                elif cls == 'LED':
                    result = self.led_det.extract(warped, box, holes)
                    if result and 'endpoints' in result:
                        pins = result['endpoints']
                elif cls == 'Diode':
                    result = self.diode_det.extract(warped, box)
                    if result and result[0] is not None and result[1] is not None:
                        pins = list(result)
                elif cls == 'IC':
                    roi = warped[y1:y2, x1:x2]
                    ics = self.ic_det.detect(roi)
                    if ics:
                        pins = [(x1 + px, y1 + py) for px, py in ics[0]['pin_points']]
                elif cls == 'Line_area':
                    roi = warped[y1:y2, x1:x2]
                    segs = self.wire_det.detect_wires(roi)
                    endpoints, _ = self.wire_det.select_best_endpoints(segs)
                    if endpoints:
                        pins = [(x1 + pt[0], y1 + pt[1]) for pt in endpoints]
            except Exception as e:
                print(f"⚠️ {cls} 핀 검출 실패: {e}")
            
            # 실패시 기본 위치 사용
            if len(pins) != expected:
                print(f"⚠️ {cls} 핀 자동검출 실패, 기본값 사용")
                if cls == 'IC':
                    # IC 핀은 좌상단부터 시계방향으로 8개
                    pins = [
                        (x1+w//4, y1+h//4), (x1+w//2, y1+h//4), (x1+3*w//4, y1+h//4),
                        (x1+3*w//4, y1+h//2), (x1+3*w//4, y1+3*h//4), (x1+w//2, y1+3*h//4),
                        (x1+w//4, y1+3*h//4), (x1+w//4, y1+h//2)
                    ]
                else:
                    # 2핀 컴포넌트는 양쪽 끝
                    pins = [(x1+10, y1+h//2), (x2-10, y1+h//2)]
            
            component_pins.append({
                'class': cls,
                'box': box,
                'pins': pins,
                'value': 100.0 if cls == 'Resistor' else 0.0,
                'num_idx': i+1
            })
        
        print(f"✅ 모든 컴포넌트 핀 처리 완료")
        return component_pins, holes

    def manual_pin_verification_and_correction(self, warped, component_pins, holes):
        """핀 위치 확인 및 수정 단계"""
        print("📍 핀 위치 확인 및 수정 단계")
        print("- 좌클릭: 컴포넌트 선택하여 핀 수정")
        print("- 'v': 전체 핀 위치 확인 모드")
        print("- Enter: 다음 단계로")
        
        # hole_to_net 맵 생성 (핀 시각화용)
        nets, row_nets = self.hole_det.get_board_nets(holes, base_img=warped, show=False)
        hole_to_net = {}
        for row_idx, clusters in row_nets:
            for entry in clusters:
                net_id = entry['net_id']
                for x, y in entry['pts']:
                    hole_to_net[(int(round(x)), int(round(y)))] = net_id
        
        # Union-Find 초기화
        parent = {net_id: net_id for net_id in set(hole_to_net.values())}
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        # Net 색상 매핑
        rng = np.random.default_rng(1234)
        final_nets = set(find(n) for n in hole_to_net.values())
        net_colors = {
            net_id: tuple(int(c) for c in rng.integers(0, 256, 3))
            for net_id in final_nets
        }
        
        def redraw_pins(verification_mode=False):
            """핀 위치와 네트워크 매핑을 시각화"""
            img = warped.copy()
            
            if verification_mode:
                # 확인 모드: 더 상세한 정보 표시
                
                # 1) 모든 구멍을 네트 색상으로 표시
                for (hx, hy), net_id in hole_to_net.items():
                    final_net = find(net_id)
                    hole_color = net_colors.get(final_net, (128, 128, 128))
                    cv2.circle(img, (int(hx), int(hy)), 3, hole_color, -1)
                
                # 2) 컴포넌트와 핀을 더 자세히 표시
                for i, comp in enumerate(component_pins):
                    x1, y1, x2, y2 = comp['box']
                    color = self.class_colors.get(comp['class'], (0, 255, 255))
                    
                    # 컴포넌트 박스 (두껍게)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # 컴포넌트 정보 (더 자세히)
                    expected = 8 if comp['class'] == 'IC' else 2
                    actual = len(comp['pins'])
                    status = "✓" if actual == expected else "⚠"
                    info_text = f"{i+1}:{comp['class']} {status}({actual}/{expected})"
                    cv2.putText(img, info_text, (x1, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 핀을 더 크게 표시
                    for j, (px, py) in enumerate(comp['pins']):
                        if hole_to_net:
                            closest = min(hole_to_net.keys(), 
                                        key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
                            raw_net = hole_to_net[closest]
                            net_id = find(raw_net)
                            pin_color = net_colors.get(net_id, (255, 255, 255))
                            
                            # 핀과 구멍 사이 연결선 표시
                            cv2.line(img, (int(px), int(py)), closest, (255, 255, 255), 1)
                        else:
                            pin_color = (0, 255, 0)
                        
                        # 핀을 더 크게 표시
                        cv2.circle(img, (int(px), int(py)), 8, pin_color, -1)
                        cv2.circle(img, (int(px), int(py)), 8, (0, 0, 0), 2)
                        
                        # 핀 번호와 네트 ID 표시
                        cv2.putText(img, f"P{j+1}", (int(px)+10, int(py)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        if hole_to_net and net_id:
                            cv2.putText(img, f"N{net_id}", (int(px)+10, int(py)+15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, pin_color, 1)
                
                # 범례 추가
                legend_y = 50
                cv2.putText(img, "Legend:", (10, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, "Small dots: Holes", (10, legend_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(img, "Large circles: Component pins", (10, legend_y + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(img, "Lines: Pin-to-hole connections", (10, legend_y + 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
            else:
                # 일반 모드: 간단한 표시
                for i, comp in enumerate(component_pins):
                    x1, y1, x2, y2 = comp['box']
                    color = self.class_colors.get(comp['class'], (0, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{i+1}:{comp['class']}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 핀을 간단히 표시
                    for j, (px, py) in enumerate(comp['pins']):
                        if hole_to_net:
                            closest = min(hole_to_net.keys(), 
                                        key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
                            raw_net = hole_to_net[closest]
                            net_id = find(raw_net)
                            pin_color = net_colors.get(net_id, (255, 255, 255))
                        else:
                            pin_color = (0, 255, 0)
                        
                        cv2.circle(img, (int(px), int(py)), 6, pin_color, -1)
                        cv2.circle(img, (int(px), int(py)), 6, (0, 0, 0), 2)
                        
                        # 핀 번호만 간단히 표시
                        cv2.putText(img, str(j+1), (int(px)+8, int(py)-8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return img
        
        def manual_pin_selection(comp_idx):
            """특정 컴포넌트의 핀을 수동으로 재설정"""
            comp = component_pins[comp_idx]
            x1, y1, x2, y2 = comp['box']
            expected = 8 if comp['class'] == 'IC' else 2
            
            print(f"📍 {comp['class']} #{comp_idx+1}의 핀 {expected}개를 선택하세요")
            
            # ROI 추출 및 확대
            roi = warped[y1:y2, x1:x2].copy()
            h, w = roi.shape[:2]
            scale = 3
            roi_resized = cv2.resize(roi, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
            
            pins = []
            window_name = f'{comp["class"]} Pin Selection'
            
            def mouse_cb(event, x, y, flags, param):
                nonlocal pins, roi_resized
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    if len(pins) < expected:
                        # 클릭 좌표를 원본 크기로 변환
                        x_orig = int(x / scale)
                        y_orig = int(y / scale)
                        pins.append((x1 + x_orig, y1 + y_orig))
                        
                        # 시각화
                        cv2.circle(roi_resized, (x, y), 8, (0, 0, 255), -1)
                        cv2.putText(roi_resized, str(len(pins)), (x+10, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow(window_name, roi_resized)
                        
                elif event == cv2.EVENT_RBUTTONDOWN:
                    # 마지막 핀 제거
                    if pins:
                        pins.pop()
                        # 이미지 다시 그리기
                        roi_resized = cv2.resize(warped[y1:y2, x1:x2].copy(), 
                                               (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
                        for idx, (px, py) in enumerate(pins):
                            disp_x = (px - x1) * scale
                            disp_y = (py - y1) * scale
                            cv2.circle(roi_resized, (int(disp_x), int(disp_y)), 8, (0, 0, 255), -1)
                            cv2.putText(roi_resized, str(idx+1), (int(disp_x)+10, int(disp_y)-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow(window_name, roi_resized)
            
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, mouse_cb)
            cv2.imshow(window_name, roi_resized)
            
            print(f"좌클릭: 핀 추가 ({len(pins)}/{expected}), 우클릭: 마지막 핀 제거, ESC: 완료")
            
            while len(pins) < expected:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
            
            cv2.destroyWindow(window_name)
            
            if len(pins) == expected:
                comp['pins'] = pins
                print(f"✅ {comp['class']} #{comp_idx+1} 핀 {len(pins)}개 업데이트됨")
            else:
                print(f"⚠️ 핀 개수 부족 ({len(pins)}/{expected}), 기존 핀 유지")
        
        # 메인 루프
        window_name = '핀 위치 확인 및 수정'
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 클릭한 위치의 컴포넌트 찾기
                for i, comp in enumerate(component_pins):
                    x1, y1, x2, y2 = comp['box']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        manual_pin_selection(i)
                        break
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        verification_mode = False
        
        while True:
            if verification_mode:
                # 전체 핀 확인 모드 - 상세 정보 표시
                img = redraw_pins(verification_mode=True)
                cv2.putText(img, "VERIFICATION MODE - Detailed pin analysis", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, "Press 'v' to exit verification mode", 
                           (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # 일반 모드 - 간단한 표시
                img = redraw_pins(verification_mode=False)
                cv2.putText(img, "EDIT MODE - Click component to edit pins", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, "'v': detailed view | Enter: next step", 
                           (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, img)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 13:  # Enter
                break
            elif key == ord('v'):  # 확인 모드 토글
                verification_mode = not verification_mode
                print(f"{'📋 상세 확인 모드' if verification_mode else '✏️ 편집 모드'}")
            elif key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        print("✅ 핀 위치 확인 및 수정 완료")
        
        return component_pins

    def quick_value_input(self, component_pins):
        """개별 저항값 입력"""
        resistors = [(i, comp) for i, comp in enumerate(component_pins) if comp['class'] == 'Resistor']
        
        if not resistors:
            print("✅ 저항이 없습니다.")
            return
        
        print(f"📝 {len(resistors)}개 저항의 값을 입력하세요")
        
        root = tk.Tk()
        root.withdraw()
        
        for idx, (comp_idx, comp) in enumerate(resistors):
            # 현재 저항 정보 표시
            x1, y1, x2, y2 = comp['box']
            
            value = simpledialog.askfloat(
                f"저항값 입력 ({idx+1}/{len(resistors)})", 
                f"저항 R{idx+1} (위치: {x1},{y1}) 값을 입력하세요 (Ω):",
                initialvalue=100.0,
                minvalue=0.1
            )
            
            if value is not None:
                comp['value'] = value
                print(f"✅ R{idx+1}: {value}Ω")
            else:
                print(f"⚠️ R{idx+1}: 기본값 100Ω 사용")
                comp['value'] = 100.0
        
        root.destroy()
        print(f"✅ 모든 저항값 입력 완료")

    def quick_power_selection(self, warped, component_pins):
        """간단한 전원 선택"""
        print("⚡ 전원 단자를 선택하세요")
        print("- 첫 번째 클릭: 양극(+)")
        print("- 두 번째 클릭: 음극(-)")
        
        # 모든 핀 위치 수집
        all_endpoints = [pt for comp in component_pins for pt in comp['pins']]
        
        # 전원 전압 입력
        root = tk.Tk()
        root.withdraw()
        voltage = simpledialog.askfloat("전원 전압", "전원 전압을 입력하세요 (V):", initialvalue=5.0)
        root.destroy()
        
        if not voltage:
            voltage = 5.0
        
        # 클릭으로 전원 단자 선택
        selected_points = []
        
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 2:
                selected_points.append((x, y))
                # 가장 가까운 실제 핀 찾기
                closest = min(all_endpoints, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
                cv2.circle(warped, closest, 8, (0, 0, 255), -1)
                cv2.putText(warped, f"{'+'if len(selected_points)==1 else '-'}", 
                           (closest[0]+10, closest[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('전원 선택', warped)
        
        cv2.imshow('전원 선택', warped)
        cv2.setMouseCallback('전원 선택', on_click)
        
        while len(selected_points) < 2:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        
        if len(selected_points) == 2:
            # 가장 가까운 실제 핀들 찾기
            plus_pt = min(all_endpoints, key=lambda p: (p[0]-selected_points[0][0])**2 + (p[1]-selected_points[0][1])**2)
            minus_pt = min(all_endpoints, key=lambda p: (p[0]-selected_points[1][0])**2 + (p[1]-selected_points[1][1])**2)
            
            print(f"✅ 전원 설정: +{plus_pt}, -{minus_pt}, {voltage}V")
            return voltage, plus_pt, minus_pt
        else:
            # 기본값 사용
            print("⚠️ 전원 선택 실패, 기본값 사용")
            return voltage, all_endpoints[0], all_endpoints[-1]

    def generate_final_circuit(self, component_pins, holes, voltage, plus_pt, minus_pt, warped):
        """최종 회로 생성"""
        print("🔄 회로도 생성 중...")
        
        # hole_to_net 맵 생성
        nets, row_nets = self.hole_det.get_board_nets(holes, base_img=warped, show=False)
        hole_to_net = {}
        for row_idx, clusters in row_nets:
            for entry in clusters:
                net_id = entry['net_id']
                for x, y in entry['pts']:
                    hole_to_net[(int(round(x)), int(round(y)))] = net_id
        
        # nearest_net 함수 정의
        def nearest_net(pt):
            closest = min(hole_to_net.keys(), key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2)
            return hole_to_net[closest]
        
        # 전원 매핑
        net_plus = nearest_net(plus_pt)
        net_minus = nearest_net(minus_pt)
        
        # 와이어 연결 처리
        wires = []
        for comp in component_pins:
            if comp['class'] == 'Line_area' and len(comp['pins']) == 2:
                net1 = nearest_net(comp['pins'][0])
                net2 = nearest_net(comp['pins'][1])
                if net1 != net2:
                    wires.append((net1, net2))
        
        # schemdraw 그리드 좌표 변환
        img_w = warped.shape[1]
        comp_count = len([c for c in component_pins if c['class'] != 'Line_area'])
        grid_width = comp_count * 2 + 2
        x_plus_grid = plus_pt[0] / img_w * grid_width
        x_minus_grid = minus_pt[0] / img_w * grid_width
        
        power_pairs = [(net_plus, x_plus_grid, net_minus, x_minus_grid)]
        
        # 회로 생성
        try:
            mapped, final_hole_to_net = generate_circuit(
                all_comps=component_pins,
                holes=holes,
                wires=wires,
                voltage=voltage,
                output_spice='circuit.spice',
                output_img='circuit.jpg',
                hole_to_net=hole_to_net,
                power_pairs=power_pairs
            )
            
            print("✅ 회로도 생성 완료!")
            print("📁 생성된 파일:")
            print("  - circuit.jpg (회로도)")
            print("  - circuit.spice (SPICE 넷리스트)")
            
            return True
            
        except Exception as e:
            print(f"❌ 회로 생성 실패: {e}")
            return False

    def run(self):
        """전체 프로세스 실행"""
        print("=" * 50)
        print("🔌 간소화된 브레드보드 → 회로도 변환기")
        print("=" * 50)
        
        # 1. 이미지 로드
        img = self.load_image()
        if img is None:
            print("❌ 이미지를 선택하지 않았습니다.")
            return
        
        # 2. 브레드보드 자동 검출 및 변환
        warped = self.auto_detect_and_transform(img)
        if warped is None:
            return
        
        # 3. 컴포넌트 검출
        components = self.quick_component_detection(warped)
        if not components:
            print("❌ 컴포넌트가 검출되지 않았습니다.")
            return
        
        # 4. 핀 검출
        component_pins, holes = self.auto_pin_detection(warped, components)
        
        # 5. 핀 위치 확인 및 수정 단계 추가
        component_pins = self.manual_pin_verification_and_correction(warped, component_pins, holes)
        
        # 6. 값 입력
        self.quick_value_input(component_pins)
        
        # 7. 전원 선택
        voltage, plus_pt, minus_pt = self.quick_power_selection(warped, component_pins)
        
        # 8. 회로 생성
        success = self.generate_final_circuit(component_pins, holes, voltage, plus_pt, minus_pt, warped)
        
        if success:
            print("\n🎉 변환 완료!")
            print("generated files:")
            print("  - circuit.jpg")
            print("  - circuit.spice")
            
            # 결과 보기
            try:
                result_img = cv2.imread('circuit.jpg')
                if result_img is not None:
                    cv2.imshow('생성된 회로도', result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except:
                pass
        else:
            print("❌ 변환에 실패했습니다.")

if __name__ == "__main__":
    converter = SimpleCircuitConverter()
    converter.run()