import cv2
import tkinter as tk
from tkinter import simpledialog, Scrollbar, Listbox
import numpy as np

# 클래스 이름 (data.yaml 기반)
class_names = ['Breadboard', 'Capacitor', 'Diode', 'IC', 'LED', 'Line_area', 'Line_point', 'Resistor']

# 각 클래스에 대해 고유 색상 지정 (BGR)
class_colors = {
    'Breadboard': (0, 128, 255),     # 오렌지
    'Capacitor': (255, 0, 255),      # 핑크
    'Diode': (0, 255, 0),            # 라임 그린
    'IC': (204, 102, 255),           # 연보라
    'LED': (102, 0, 102),            # 남색
    'Line_area': (255, 0, 0),        # 파란색
    'Resistor': (255, 255, 102),     # 하늘색
}

boxes = []
drawing = False
start_point = None

def choose_class_gui():
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

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, boxes

    image = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = image.copy()
        cv2.rectangle(temp, start_point, (x, y), (255, 0, 0), 2)
        cv2.imshow("Manual Labeling", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        x1, y1 = start_point
        x2, y2 = end_point
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        selected_class = choose_class_gui()
        if selected_class is not None:
            boxes.append((selected_class, (x1, y1, x2, y2)))

def draw_and_label(image):
    cv2.namedWindow("Manual Labeling")
    cv2.setMouseCallback("Manual Labeling", mouse_callback, param=image)

    print("🖱️ 마우스로 드래그해서 객체 영역을 지정하세요. ESC 키를 누르면 종료됩니다.")

    while True:
        temp = image.copy()
        for cls_name, (x1, y1, x2, y2) in boxes:
            color = class_colors.get(cls_name, (255, 255, 255))
            cv2.rectangle(temp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(temp, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Manual Labeling", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    return boxes
