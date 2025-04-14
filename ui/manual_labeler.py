# manual_labeler.py

import cv2
import tkinter as tk
from tkinter import Scrollbar, Listbox
import numpy as np

# data.yaml에 기반한 클래스 목록 및 각 클래스 별 색상
class_names = ['Breadboard', 'Capacitor', 'Diode', 'IC', 'LED', 'Line_area','Resistor']
class_colors = {
    'Breadboard': (0, 128, 255),
    'Capacitor': (255, 0, 255),
    'Diode': (0, 255, 0),
    'IC': (204, 102, 255),
    'LED': (102, 0, 102),
    'Line_area': (255, 0, 0),
    'Resistor': (255, 255, 102),
}

def choose_class_gui():
    """
    클래스 선택 GUI를 띄워, 사용자가 선택한 클래스명을 반환합니다.
    반환값: 선택된 클래스명 (문자열) 또는 None.
    """
    selected_class = None

    def select():
        nonlocal selected_class
        sel = listbox.curselection()
        if sel:
            selected_class = class_names[sel[0]]
            root.destroy()

    root = tk.Tk()
    root.title("클래스 선택")
    root.geometry("300x250")
    
    scrollbar = Scrollbar(root)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    listbox = Listbox(root, yscrollcommand=scrollbar.set, font=("Arial", 14))
    for cls in class_names:
        listbox.insert(tk.END, cls)
    listbox.pack(fill=tk.BOTH, expand=True)
    
    scrollbar.config(command=listbox.yview)
    
    button = tk.Button(root, text="선택", command=select)
    button.pack(pady=5)
    
    root.mainloop()
    return selected_class

def draw_and_label(image):
    """
    이미지에 대해 사용자가 마우스로 드래그하여 객체 영역(박스)을 선택하면,
    선택한 영역에 대해 클래스 선택 GUI를 띄워, 최종적으로
    [(클래스명, (x1, y1, x2, y2)), ...] 형식의 리스트를 반환합니다.
    """
    local_boxes = []  # 사용자가 선택한 객체 정보를 저장
    local_state = {'drawing': False, 'start_point': None}

    def mouse_callback(event, x, y, flags, param):
        nonlocal local_boxes, local_state, image

        if event == cv2.EVENT_LBUTTONDOWN:
            local_state['drawing'] = True
            local_state['start_point'] = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and local_state['drawing']:
            temp = image.copy()
            cv2.rectangle(temp, local_state['start_point'], (x, y), (255, 0, 0), 2)
            cv2.imshow("Manual Labeling", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            local_state['drawing'] = False
            end_point = (x, y)
            x1, y1 = local_state['start_point']
            x2, y2 = end_point
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            
            # 영역이 정해지면 클래스 선택 GUI를 실행
            selected_class = choose_class_gui()
            if selected_class is not None:
                local_boxes.append((selected_class, (x1, y1, x2, y2)))

    cv2.namedWindow("Manual Labeling")
    cv2.setMouseCallback("Manual Labeling", mouse_callback)

    print("🖱️ 마우스로 드래그해서 객체 영역을 지정하세요. ESC 키를 누르면 종료됩니다.")
    while True:
        temp = image.copy()
        # 현재까지 선택한 박스를 표시
        for cls_name, (x1, y1, x2, y2) in local_boxes:
            color = class_colors.get(cls_name, (255, 255, 255))
            cv2.rectangle(temp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(temp, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Manual Labeling", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    return local_boxes
