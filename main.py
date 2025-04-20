import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import numpy as np
import tkinter as tk

from detector.component_detector import ComponentPinDetector
from detector.pin_detector import PinDetector
from ui.perspective_editor import select_and_transform
from mapper.pin_mapper import ComponentToPinMapper
from ui.manual_labeler import draw_and_label, choose_class_gui
from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.wire_detector import WireDetector
from detector.resistor_detector import ResistorEndpointDetector

# 소자별 색상 (data.yaml 기준)
class_colors = {
    'Breadboard': (0, 128, 255),
    'Capacitor': (255, 0, 255),
    'Diode': (0, 255, 0),
    'IC': (204, 102, 255),
    'LED': (102, 0, 102),
    'Line_area': (255, 0, 0),
    'Resistor': (200, 170, 0)
}


def imread_unicode(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def wait_for_next():
    root = tk.Tk()
    root.title('다음 단계로 진행')
    root.geometry('200x100')
    tk.Button(root, text='다음', font=('Arial', 14), command=root.destroy).pack(expand=True)
    root.mainloop()


def modify_detections(image, detections):
    win = 'object change'
    updated = detections.copy()

    def draw_boxes():
        tmp = image.copy()
        for i, (cls, conf, box) in enumerate(updated):
            x1, y1, x2, y2 = box
            cv2.rectangle(tmp, (x1, y1), (x2, y2), class_colors.get(cls, (255,255,255)), 2)
            cv2.putText(tmp, f'[{i}] {cls}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors.get(cls, (255,255,255)), 2)
        cv2.imshow(win, tmp)

    def prompt(cls):
        choice = {'cmd': None}
        def chg(): choice['cmd']='change'; dlg.destroy()
        def dlt(): choice['cmd']='delete'; dlg.destroy()
        def cnl(): choice['cmd']=None; dlg.destroy()
        dlg = tk.Toplevel(); dlg.title('object change')
        tk.Label(dlg, text=f'현재 클래스: {cls}\n어떻게 하시겠습니까?', font=('Arial',12)).pack(padx=10,pady=10)
        frm = tk.Frame(dlg); frm.pack(padx=10,pady=10)
        tk.Button(frm, text='Change', width=10, command=chg).grid(row=0,column=0,padx=5)
        tk.Button(frm, text='Delete', width=10, command=dlt).grid(row=0,column=1,padx=5)
        tk.Button(frm, text='Cancel', width=10, command=cnl).grid(row=0,column=2,padx=5)
        dlg.wait_window()
        if choice['cmd']=='change':
            new = choose_class_gui()
            return new or cls
        if choice['cmd']=='delete': return 'delete'
        return None

    cv2.namedWindow(win)
    draw_boxes()
    cv2.setMouseCallback(win, lambda e,x,y,f,p: (on_click(e,x,y) if e==cv2.EVENT_LBUTTONDOWN else None))
    def on_click(event,x,y):
        for idx,(cls,conf,box) in enumerate(updated):
            x1,y1,x2,y2 = box
            if x1<=x<=x2 and y1<=y<=y2:
                res = prompt(cls)
                if res=='delete': updated.pop(idx)
                elif res: updated[idx]=(res,conf,box)
                break
        draw_boxes()

    print("수정 후 'q' 누르세요.")
    while True:
        if cv2.waitKey(0)&0xFF==ord('q'): break
    cv2.destroyWindow(win)
    return updated


def main():
    # 모델 및 검출기 초기화
    detector = FasterRCNNDetector(r'D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\fasterrcnn.pt')
    pin_det = PinDetector()
    mapper = ComponentToPinMapper()
    comp_pin = ComponentPinDetector()
    wire_det = WireDetector(kernel_size=4)
    resistor_det = ResistorEndpointDetector()

    # 이미지 로드 및 브레드보드 영역 투영
    img = imread_unicode(r'D:\Hyuntak\연구실\AR 회로 튜터\개발\breadboard7.jpg')
    comps = detector.detect(img)
    bb = next((b for c,_,b in comps if c.lower()=='breadboard'), None)
    if bb is None: raise ValueError('Breadboard 미검출')
    warped, _ = select_and_transform(img.copy(), bb)
    warped_raw = warped.copy()

    # 컴포넌트 검출 + 수동 라벨링 + 수정
    auto = [c for c in detector.detect(warped) if c[0].lower()!='breadboard']
    manual = draw_and_label(warped)
    all_comps = [(cls,conf,box) for cls,conf,box in auto] + [(cls,1.0,box) for cls,box in manual]
    all_comps = modify_detections(warped, all_comps)

    # 와이어 시각화 및 엔드포인트
    wire_det.process_line_area_wires(warped_raw, all_comps, scale_factor=1)
    wires = wire_det.detect_wires(warped)
    best, color = wire_det.select_best_endpoints(wires)
    print(f'채널: {color}, 엔드포인트: {best}')

    # 저항 끝점 추출 및 시각화
    for cls,conf,box in all_comps:
        if cls=='Resistor':
            pts = resistor_det.extract(warped, box)
            resistor_det.draw(warped_raw, pts)

    # 결과 출력
    cv2.imshow('Result', warped_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
