# component_editor.py
import cv2
import tkinter as tk
from tkinter import simpledialog

class ComponentEditor:
    def __init__(self, class_colors):
        self.class_colors = class_colors
    
    def quick_component_detection(self, warped, detector):
        """빠른 컴포넌트 검출 및 간단한 수정"""
        print("🔍 컴포넌트 자동 검출 중...")
        detections = detector.detect(warped)
        all_comps = [(cls, 1.0, box) for cls, _, box in detections if cls.lower() != 'breadboard']
        
        # 시각화
        vis_img = warped.copy()
        for i, (cls, _, box) in enumerate(all_comps):
            x1, y1, x2, y2 = box
            color = self.class_colors.get(cls, (0, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{i+1}:{cls}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Detected Components (Enter: Validate, Space: Edit mode)', vis_img)
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
        print("- 우클릭: 컴포넌트 삭제/수정")
        print("- 키보드 'd': 번호로 삭제")
        print("- 키보드 'c': 번호로 클래스 변경")
        print("- Enter: 완료")
        
        editing = True
        drawing = False
        start_point = None
        window_name = 'Edit Components'
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, components
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp_img = warped.copy()
                self._draw_components(temp_img, components)
                cv2.rectangle(temp_img, start_point, (x, y), (255, 255, 255), 2)
                cv2.imshow(window_name, temp_img)
                
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                x1, x2 = sorted([start_point[0], x])
                y1, y2 = sorted([start_point[1], y])
                
                if abs(x2-x1) > 20 and abs(y2-y1) > 20:
                    new_class = self._select_component_class()
                    if new_class:
                        components.append((new_class, 1.0, (x1, y1, x2, y2)))
                        print(f"✅ {new_class} 추가됨")
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._handle_right_click(components, x, y)
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while editing:
            vis_img = warped.copy()
            self._draw_components(vis_img, components)
            cv2.imshow(window_name, vis_img)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 13:  # Enter
                editing = False
            elif key == ord('d'):  # 삭제
                self._delete_component_by_number(components)
            elif key == ord('c'):  # 클래스 변경
                self._change_component_class(components)
        
        cv2.destroyAllWindows()
        return components
    
    def _draw_components(self, img, components):
        """컴포넌트들을 이미지에 그리기"""
        for i, (cls, _, box) in enumerate(components):
            x1, y1, x2, y2 = box
            color = self.class_colors.get(cls, (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{i+1}:{cls}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _select_component_class(self):
        """컴포넌트 클래스 선택 대화상자"""
        root = tk.Tk()
        root.withdraw()
        
        classes = ['Resistor', 'LED', 'Diode', 'IC', 'Line_area', 'Capacitor']
        class_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(classes))
        choice = simpledialog.askinteger(
            "컴포넌트 선택", 
            f"추가할 컴포넌트를 선택하세요:\n{class_str}",
            minvalue=1, maxvalue=len(classes)
        )
        
        root.destroy()
        return classes[choice-1] if choice else None
    
    def _handle_right_click(self, components, x, y):
        """우클릭 처리 - 컴포넌트 삭제/수정"""
        for i, (cls, _, box) in enumerate(components):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                root = tk.Tk()
                root.withdraw()
                choice = simpledialog.askinteger(
                    "삭제/수정", 
                    f"'{cls}' 컴포넌트:\n1. 삭제\n2. 클래스 변경\n선택하세요 (1-2):",
                    minvalue=1, maxvalue=2
                )
                if choice == 1:
                    components.pop(i)
                    print(f"✅ {cls} 삭제됨")
                elif choice == 2:
                    new_class = self._select_component_class()
                    if new_class:
                        components[i] = (new_class, 1.0, box)
                        print(f"✅ {cls} → {new_class}로 변경됨")
                root.destroy()
                break
    
    def _delete_component_by_number(self, components):
        """번호로 컴포넌트 삭제"""
        if not components:
            return
            
        root = tk.Tk()
        root.withdraw()
        idx = simpledialog.askinteger("삭제", f"삭제할 컴포넌트 번호 (1-{len(components)}):")
        if idx and 1 <= idx <= len(components):
            removed = components.pop(idx-1)
            print(f"✅ 컴포넌트 {idx} ({removed[0]}) 삭제됨")
        root.destroy()
    
    def _change_component_class(self, components):
        """번호로 컴포넌트 클래스 변경"""
        if not components:
            return
            
        root = tk.Tk()
        root.withdraw()
        idx = simpledialog.askinteger("클래스 변경", f"변경할 컴포넌트 번호 (1-{len(components)}):")
        if idx and 1 <= idx <= len(components):
            new_class = self._select_component_class()
            if new_class:
                old_class = components[idx-1][0]
                box = components[idx-1][2]
                components[idx-1] = (new_class, 1.0, box)
                print(f"✅ {idx}번 컴포넌트: {old_class} → {new_class} 변경됨")
        root.destroy()