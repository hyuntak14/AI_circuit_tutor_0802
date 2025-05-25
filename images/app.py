import streamlit as st
import streamlit_drawable_canvas as dc
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from io import BytesIO

from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.pin_detector import PinDetector
from mapper.pin_mapper import ComponentToPinMapper
from main import save_yolo_labels

class_colors = {
    'Breadboard': (0, 128, 255),
    'Capacitor': (255, 0, 255),
    'Diode': (0, 255, 0),
    'IC': (204, 102, 255),
    'LED': (102, 0, 102),
    'Line_area': (255, 0, 0),
    'Resistor': (255, 255, 102),
}

class_names = ['Breadboard', 'Capacitor', 'Diode', 'IC', 'LED', 'Line_area', 'Resistor']

st.set_page_config(page_title="AI 회로도 인식기", layout="wide")
st.title("🧠 AI 기반 회로도 객체 탐지 및 핀 매핑")

model_path = "D:/Hyuntak/연구실/AR 회로 튜터/breadboard_project/model/fasterrcnn.pt"

uploaded_file = st.file_uploader("회로도 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(original_image)

    st.image(original_image, caption="업로드된 원본 이미지", use_container_width=True)

    display_image = original_image.copy()
    display_image.thumbnail((800, 800))
    display_np = np.array(display_image)

    from streamlit_drawable_canvas import st_canvas

    detector_preview = FasterRCNNDetector(model_path)
    preview_components = detector_preview.detect(display_np)
    preview_bb = next((box for cls_name, _, box in preview_components if cls_name.lower() == 'breadboard'), None)

    initial_polygon = []
    if preview_bb:
        x1, y1, x2, y2 = preview_bb
        initial_polygon = [
            {"x": x1, "y": y1},
            {"x": x2, "y": y1},
            {"x": x2, "y": y2},
            {"x": x1, "y": y2}
        ]

    st.subheader("📐 꼭짓점 보정 또는 수동 지정")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000",
        background_image=display_image,
        update_streamlit=True,
        height=display_image.size[1],
        width=display_image.size[0],
        initial_drawing={"version": "4.4.0", "objects": [{"type": "polygon", "path": [[p['x'], p['y']] for p in initial_polygon], "fill": "rgba(255, 0, 0, 0.3)", "stroke": "#000", "strokeWidth": 3}]} if initial_polygon else None,
        drawing_mode="polygon",
        key="canvas_poly"
    )

    corners = None
    if canvas_result.json_data is not None:
        try:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0 and objects[0]["type"] == "polygon":
                corners = objects[0]["path"][:4]
        except:
            corners = None

    def run_pipeline(original_np, model_path, corners=None):
        detector = FasterRCNNDetector(model_path)
        pin_detector = PinDetector()
        mapper = ComponentToPinMapper()

        components = detector.detect(original_np)
        breadboard_boxes = [box for cls_name, _, box in components if cls_name.lower() == 'breadboard']

        if not breadboard_boxes and not corners:
            st.warning("Breadboard가 감지되지 않았습니다. 캔버스에서 영역을 수동으로 지정해주세요.")
            return None, None, None

        if corners and len(corners) == 4:
            scale_x = original_np.shape[1] / 800
            scale_y = original_np.shape[0] / 800
            scaled_corners = [(int(x * scale_x), int(y * scale_y)) for x, y in corners]
            pts_src = np.float32(scaled_corners)
            pts_dst = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped_img = cv2.warpPerspective(original_np, matrix, (640, 640))
        else:
            x1, y1, x2, y2 = breadboard_boxes[0]
            warped_img = original_np[y1:y2, x1:x2]

        _, pins = pin_detector.detect_pins(warped_img)
        components = detector.detect(warped_img)

        yolo_boxes = []
        for cls_name, conf, box in components:
            if cls_name.lower() == 'breadboard':
                continue
            x1_, y1_, x2_, y2_ = box
            cv2.rectangle(warped_img, (x1_, y1_), (x2_, y2_), class_colors.get(cls_name, (255,255,255)), 2)
            cv2.putText(warped_img, f"{cls_name} {conf:.2f}", (x1_, y1_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors.get(cls_name, (255,255,255)), 2)
            yolo_boxes.append((cls_name, (x1_, y1_, x2_, y2_)))

        return warped_img, yolo_boxes, pins

    if st.button("🎯 객체 인식 및 핀 매핑 실행"):
        result_img, yolo_boxes, pins = run_pipeline(original_np, model_path, corners=corners)
        if result_img is not None:
            st.subheader("✍ 수동 라벨링: 객체 박스를 그려주세요")
            selected_class = st.selectbox("라벨 클래스 선택", class_names, index=0)
            canvas_label = st_canvas(
    fill_color="rgba(0, 255, 0, 0.3)",
    stroke_width=2,
    stroke_color="#00FF00",
    background_image=Image.fromarray(cv2.cvtColor(result_img.copy(), cv2.COLOR_BGR2RGB)),
    update_streamlit=False,
    height=result_img.shape[0],
    width=result_img.shape[1],
    drawing_mode="rect",
    key="canvas_rect"
)

            manual_boxes = []
            if canvas_label.json_data is not None:
                for obj in canvas_label.json_data["objects"]:
                    if obj["type"] == "rect":
                        x, y = int(obj["left"]), int(obj["top"])
                        w, h = int(obj["width"]), int(obj["height"])
                        manual_boxes.append((selected_class, (x, y, x + w, y + h)))

            for cls_name, (x1m, y1m, x2m, y2m) in manual_boxes:
                color = class_colors.get(cls_name, (255, 255, 255))
                cv2.rectangle(result_img, (x1m, y1m), (x2m, y2m), color, 2)
                cv2.putText(result_img, f"{cls_name} (manual)", (x1m, y1m - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                center = ((x1m + x2m) // 2, (y1m + y2m) // 2)
                mapped_pins = ComponentToPinMapper().find_nearest_pins((x1m, y1m, x2m, y2m), pins)
                for pin in mapped_pins:
                    cv2.line(result_img, center, pin, color, 1)

            all_boxes = yolo_boxes + manual_boxes
            save_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            save_yolo_labels(save_txt.name, result_img.shape, all_boxes, class_names)

            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_img_rgb, caption="결과 이미지", use_container_width=True)

            with open(save_txt.name, "r") as f:
                label_text = f.read()

            st.download_button("YOLO 라벨 다운로드", label_text, file_name="labels.txt")
