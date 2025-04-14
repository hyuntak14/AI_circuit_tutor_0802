# main.py

from detector.pin_detector import PinDetector
from ui.perspective_editor import select_and_transform
from mapper.pin_mapper import ComponentToPinMapper
from ui.manual_labeler import draw_and_label  # 수동 라벨러 기능
from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.location_detector import KeypointDetector  # 수정된 KeypointDetector 사용
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk

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
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def wait_for_next():
    """Tkinter를 이용해 '다음' 버튼을 보여주고, 사용자가 버튼을 누를 때까지 대기합니다."""
    root = tk.Tk()
    root.title("다음 단계로 진행")
    root.geometry("200x100")
    button = tk.Button(root, text="다음", font=("Arial", 14), command=root.destroy)
    button.pack(expand=True)
    root.mainloop()

def modify_detections(image, detections):
    """
    detections: [(클래스명, confidence, bbox), ...] 리스트
    각 객체에 대해 bounding box와 현재 클래스명을 보여주며, 
    사용자가 새로운 클래스명을 입력하면 수정하고, 'delete'라고 입력하면 삭제하도록 함.
    bbox는 (x1, y1, x2, y2) 형태.
    """
    modified = []
    for idx, (cls_name, conf, bbox) in enumerate(detections):
        x1, y1, x2, y2 = bbox
        # 이미지 복사본에 현재 bbox만 강조하여 표시
        temp_img = image.copy()
        color = class_colors.get(cls_name, (255, 255, 255))
        cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(temp_img, f"현재: {cls_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("객체 수정", temp_img)
        cv2.waitKey(1)  # 이미지가 뜨게 잠시 대기
        print(f"[{idx}] Bounding Box: {bbox}, 현재 클래스: {cls_name}")
        # 사용자 입력: 새 클래스명 입력 (빈 문자열이면 그대로), 'delete' 입력 시 삭제
        user_input = input("클래스 수정 (수정할 클래스명 입력, 삭제는 'delete', 그대로는 엔터): ").strip()
        cv2.destroyWindow("객체 수정")
        if user_input.lower() == "delete":
            print("해당 객체 삭제됨.")
            continue  # 현재 객체를 리스트에 추가하지 않음
        elif user_input != "":
            # 새로운 클래스명으로 업데이트
            print(f"클래스명을 '{cls_name}'에서 '{user_input}'로 수정합니다.")
            cls_name = user_input
        modified.append((cls_name, conf, bbox))
    return modified

def main():
    # 모델 경로 설정
    model_path = r"D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\fasterrcnn.pt"
    keypoint_model_paths = {
        'Capacitor': r"D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\location detector\densenet_Capacitor_keypoint_model.h5",
        'Diode':     r"D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\location detector\resnet_Diode_keypoint_model.h5",
        'IC':        r"D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\location detector\resnet_IC_keypoint_model.h5",
        'LED':       r"D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\location detector\resnet_LED_keypoint_model.h5",
        'Line_area': r"D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\location detector\densenet_Line_area_keypoint_model.h5",
        'Resistor':  r"D:\Hyuntak\연구실\AR 회로 튜터\breadboard_project\model\location detector\resnet_Resistor_keypoint_model.h5",
    }
    
    # 모델 및 검출기 초기화
    detector = FasterRCNNDetector(model_path)
    pin_detector = PinDetector()
    mapper = ComponentToPinMapper()
    kp_detector = KeypointDetector(keypoint_model_paths)
    
    # 이미지 로드
    image_path = r"D:\Hyuntak\연구실\AR 회로 튜터\개발\breadboard7.jpg"
    image = imread_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ 이미지 로드 실패: {image_path}")
    
    # 원본 이미지에서 자동 객체 검출 수행
    components = detector.detect(image)
    # Breadboard 영역 검출 (Breadboard가 반드시 존재한다고 가정 – 확실하지 않음)
    breadboard_boxes = [box for cls_name, _, box in components if cls_name.lower() == 'breadboard']
    if not breadboard_boxes:
        raise ValueError("Breadboard 영역이 검출되지 않았습니다.")
    
    # 첫 번째 Breadboard 영역 선택 (추측입니다)
    breadboard_box = breadboard_boxes[0]
    warped_img, transform_offset = select_and_transform(image.copy(), breadboard_box)
    
    # 보정된 Breadboard 영역 내에서 자동 객체 검출 (Breadboard 제외)
    components_warped = detector.detect(warped_img)
    for cls_name, conf, box in components_warped:
        if cls_name.lower() == 'breadboard':
            continue
        x1, y1, x2, y2 = box
        color = class_colors.get(cls_name, (255, 255, 255))
        cv2.rectangle(warped_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(warped_img, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 수동 라벨링: 인식하지 못한 객체를 사용자가 드래그하여 선택
    # 반환 값 예: [("Diode", (x1, y1, x2, y2)), ...] (확실하지 않음)
    manual_labels = draw_and_label(warped_img)
    
    # 자동 검출 객체와 수동 라벨링 객체 병합
    all_components = []
    for comp in components_warped:
        if comp[0].lower() != 'breadboard':
            all_components.append(comp)
    for label in manual_labels:
        # 수동 라벨의 형식이 (클래스명, bbox)라고 가정 – confidence 값은 1.0으로 설정
        cls_name, bbox = label
        all_components.append((cls_name, 1.0, bbox))
    
    # 사용자에게 잘못 검출된 객체에 대해 클래스 수정 또는 삭제 기능 제공
    print("\n=== 검출된 객체들을 검토합니다. ===")
    all_components = modify_detections(warped_img, all_components)
    
    # '다음' 버튼을 눌러 키포인트 검출 단계로 진행
    wait_for_next()
    
    # 병합 및 수정된 모든 객체에 대해 키포인트 검출 수행
    for cls_name, conf, box in all_components:
        x1, y1, x2, y2 = box
        color = class_colors.get(cls_name, (255, 255, 255))
        try:
            # 해당 클래스가 키포인트 검출 대상일 때 수행
            if cls_name in ['Capacitor', 'Diode', 'IC', 'LED', 'Line_area', 'Resistor']:
                keypoints = kp_detector.detect_keypoint(warped_img, box, cls_name)
                for pt in keypoints:
                    cv2.circle(warped_img, pt, 4, (0, 0, 255), -1)
                    cv2.putText(warped_img, f"{cls_name} key", (pt[0] + 5, pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            print(f"{cls_name}에 대한 키포인트 검출 실패: {e}")
    
    # 최종 결과 시각화
    im_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.title("Detection + Keypoint Localization (Auto, Manual, & 수정)")
    plt.show()

if __name__ == '__main__':
    main()
