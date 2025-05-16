import streamlit as st
import cv2
import numpy as np
from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.hole_detector import HoleDetector
from detector.resistor_detector import ResistorEndpointDetector
from detector.led_detector import LedEndpointDetector
from detector.wire_detector import WireDetector
from detector.diode_detector import DiodeEndpointDetector
from detector.ic_chip_detector import ICChipPinDetector
from circuit_generator import generate_circuit
# 필요 시, schemdraw 결과를 이미지로 변환하는 헬퍼 함수 import
from diagram import draw_circuit_from_connectivity

st.set_page_config(page_title="Breadboard→Schematic", layout="wide")

@st.cache_data
def load_image(uploaded_file):
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def annotate_components(image, components):
    canvas = image.copy()
    for comp in components:
        cls = comp['class']
        # 박스 정보가 comp['box'] 또는 comp['bbox'] 형태일 수 있습니다.
        x1, y1, x2, y2 = comp.get('box', comp.get('bbox'))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return canvas


def main():
    st.title("📸 Breadboard Circuit to Schematic Converter")
    st.write("Upload a photo of your breadboard, and get both detected components and a generated schematic diagram.")

    uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Please upload an image to get started.")
        return

    image = load_image(uploaded)
    st.subheader("Input Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    if st.button("Run Detection & Generate Schematic"):
        with st.spinner("Detecting holes and clustering nets..."):
            hole_detector = HoleDetector()
            holes = hole_detector.detect(image)
            nets = hole_detector.get_board_nets(holes)
            st.write(f"🔘 Holes detected: {len(holes)} → Nets: {len(set(nets.values()))}")

        with st.spinner("Detecting components..."):
            detectors = [
                FasterRCNNDetector(model_path="path/to/fasterrcnn.pt"),
                ResistorEndpointDetector(),
                LedEndpointDetector(),
                DiodeEndpointDetector(),
                ICChipPinDetector(),
                WireDetector(),
            ]
            comps = []
            for det in detectors:
                dets = det.detect(image)
                comps.extend(dets)
            st.write(f"📦 Components detected: {len(comps)}")

        with st.spinner("Building circuit graph & drawing schematic..."):
            components_out, nets_out = generate_circuit(comps, nets)
            # TODO: diagram 함수를 이용해 schemdraw 다이어그램을 이미지로 변환
            # 예시: diagram_img = draw_circuit_from_connectivity(components_out, nets_out)

        # 결과 시각화
        st.subheader("Detected Components")
        ann = annotate_components(image, components_out)
        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.subheader("Generated Schematic Diagram")
        # st.image(diagram_img, use_column_width=True)
        st.info("Schematic rendering 기능을 구현해 주세요 (draw_circuit_from_connectivity 활용)")

        # 다운로드 링크 제공 (옵션)
        # buf = cv2.imencode('.png', diagram_img)[1].tobytes()
        # st.download_button('Download Schematic PNG', buf, file_name='schematic.png')

if __name__ == '__main__':
    main()
