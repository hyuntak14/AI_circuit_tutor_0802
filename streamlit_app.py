import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt

from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.hole_detector import HoleDetector
from detector.resistor_detector import ResistorEndpointDetector
from detector.led_detector import LedEndpointDetector
from detector.diode_detector import ResistorEndpointDetector as DiodeEndpointDetector
from detector.ic_chip_detector import ICChipPinDetector
from detector.wire_detector import WireDetector
from circuit_generator import generate_circuit
from checker.error_checker import ErrorChecker
from diagram import get_n_clicks

# 절대경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fasterrcnn.pt")

# 캔버스 크기 (main.py와 동일하게 640x640으로 고정)
DISPLAY_SIZE = 640
MAX_DISPLAY_WIDTH = DISPLAY_SIZE
MAX_DISPLAY_HEIGHT = DISPLAY_SIZE

# 전체 단계 수
TOTAL_PAGES = 12

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'processing' not in st.session_state:
    st.session_state.processing = False

# 컴포넌트 색상 매핑
COLOR_MAP = {
    'Resistor': '#e63946',
    'LED': '#f4a261',
    'Diode': '#457b9d',
    'IC': '#9d4edd',
    'Line_area': '#2a9d8f',
    'Capacitor': '#6c757d'
}

# Utility functions
@st.cache_data
def resize_image(img, target_size=DISPLAY_SIZE):
    """이미지를 정사각형으로 리사이즈하고 스케일 반환"""
    h, w = img.shape[:2]
    # 정사각형으로 만들기 위해 작은 쪽에 맞춰 크롭
    size = min(h, w)
    scale = target_size / size
    
    # 중앙 크롭
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    cropped = img[start_y:start_y+size, start_x:start_x+size]
    
    # 640x640으로 리사이즈
    resized = cv2.resize(cropped, (target_size, target_size))
    return resized, scale, (start_x, start_y)

def show_navigation(page_num, prev_enabled=True, next_enabled=True):
    """네비게이션 버튼을 표시하고 페이지 전환을 처리합니다."""
    cols = st.columns([1, 2, 1])
    
    # 이전 버튼
    if cols[0].button("◀️ Previous", key=f"prev_{page_num}", disabled=not prev_enabled or page_num <= 1):
        st.session_state.page = max(1, page_num - 1)
        st.rerun()
    
    # 진행률 표시
    with cols[1]:
        progress = page_num / TOTAL_PAGES
        st.progress(progress)
        st.write(f"Step {page_num} of {TOTAL_PAGES}")
    
    # 다음 버튼
    if cols[2].button("Next ▶️", key=f"next_{page_num}", disabled=not next_enabled):
        st.session_state.page = min(TOTAL_PAGES, page_num + 1)
        st.rerun()

# 1) 업로드 & 원본 표시
def page_1_upload():
    st.title("📸 Breadboard to Schematic")
    st.write("Upload an image of your breadboard to start the analysis.")
    
    uploaded = st.file_uploader("Choose a breadboard image", type=["jpg", "png", "jpeg"])
    
    if uploaded:
        data = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        # main.py 스타일의 이미지 처리 - 640x640으로 고정
        disp_img, scale, crop_offset = resize_image(img)
        
        st.session_state.img = img
        st.session_state.disp_img = disp_img
        st.session_state.scale = scale
        st.session_state.crop_offset = crop_offset
        
        st.success("✅ Image uploaded successfully!")
        st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), 
                caption=f"Uploaded Image (Resized to {DISPLAY_SIZE}x{DISPLAY_SIZE})", 
                use_container_width=False, width=DISPLAY_SIZE)
        
        show_navigation(1, prev_enabled=False, next_enabled=True)
    else:
        st.info("Please upload an image to proceed.")
        show_navigation(1, prev_enabled=False, next_enabled=False)

# 2) 코너 조정
def page_2_corner_adjust():
    st.subheader("Step 2: Adjust Breadboard Corners")
    
    if 'img' not in st.session_state:
        st.error("Please upload an image first.")
        show_navigation(2, next_enabled=False)
        return
    
    img = st.session_state.img
    disp_img = st.session_state.disp_img
    scale = st.session_state.scale
    crop_offset = st.session_state.crop_offset
    
    # 원본 이미지 크기
    h, w = img.shape[:2]
    
    # 크롭된 영역에서 브레드보드 검출
    crop_x, crop_y = crop_offset
    size = min(h, w)
    cropped_img = img[crop_y:crop_y+size, crop_x:crop_x+size]
    
    # Breadboard 검출 (크롭된 이미지에서)
    detector = FasterRCNNDetector(model_path=MODEL_PATH)
    dets = detector.detect(cropped_img)
    bb = next((box for cls,_,box in dets if cls.lower()=="breadboard"), None)
    
    if bb is None:
        st.error("❌ Breadboard not detected in the image.")
        show_navigation(2, next_enabled=False)
        return
    
    # 기본 코너 포인트 설정
    default_pts = [(bb[0],bb[1]),(bb[2],bb[1]),(bb[2],bb[3]),(bb[0],bb[3])]
    scaled_pts = [(int(x*scale), int(y*scale)) for x,y in default_pts]
    
    # 코너 핸들 생성
    HANDLE_SIZE = 16
    handles = []
    for cx, cy in scaled_pts:
        handles.append({
            "type":"rect","left":cx-HANDLE_SIZE//2,"top":cy-HANDLE_SIZE//2,
            "width":HANDLE_SIZE,"height":HANDLE_SIZE,
            "stroke":"red","strokeWidth":2,
            "fill":"rgba(255,0,0,0.3)","cornerColor":"red",
            "cornerSize":6,"transparentCorners":False
        })
    
    st.write("Drag the red handles to adjust the breadboard corners:")
    
    canvas = st_canvas(
        background_image=Image.fromarray(cv2.cvtColor(disp_img,cv2.COLOR_BGR2RGB)),
        width=DISPLAY_SIZE, height=DISPLAY_SIZE,
        drawing_mode="transform", initial_drawing={"objects":handles}, key="corner"
    )
    
    # 사용자 조정된 좌표 복원
    if canvas.json_data and canvas.json_data.get("objects"):
        src = [[(o["left"]+o["width"]/2)/scale, (o["top"]+o["height"]/2)/scale]
               for o in canvas.json_data["objects"]]
    else:
        src = np.float32(default_pts)
    
    # Perspective transformation (main.py와 동일하게 640x640으로)
    dst_size = DISPLAY_SIZE
    M = cv2.getPerspectiveTransform(np.float32(src), 
                                   np.float32([[0,0],[dst_size,0],[dst_size,dst_size],[0,dst_size]]))
    warped = cv2.warpPerspective(cropped_img, M, (dst_size, dst_size))
    st.session_state.warped = warped
    st.session_state.warped_raw = warped.copy()
    
    show_navigation(2, next_enabled=True)

# 3) 변환 이미지 확인
def page_3_transformed():
    st.subheader("Step 3: Perspective Corrected Image")
    
    if 'warped' in st.session_state:
        warped = st.session_state.warped
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), 
                caption=f"Perspective Corrected Breadboard ({DISPLAY_SIZE}x{DISPLAY_SIZE})", 
                use_container_width=False, width=DISPLAY_SIZE)
        st.success("✅ Image transformation completed successfully!")
        show_navigation(3, next_enabled=True)
    else:
        st.error("❌ No transformed image available. Please complete previous steps.")
        show_navigation(3, next_enabled=False)

# 4) 컴포넌트 검출 및 편집
def page_4_component_edit():
    st.subheader("Step 4: Component Detection & Manual Edit")
    
    if 'warped' not in st.session_state:
        st.error("❌ No transformed image available.")
        show_navigation(4, next_enabled=False)
        return
    
    warped = st.session_state.warped
    # warped는 이미 640x640이므로 추가 리사이즈 불필요
    disp_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # 컴포넌트 검출
    detector = FasterRCNNDetector(model_path=MODEL_PATH)
    raw = detector.detect(warped)
    comps = [{'class':c,'bbox':b} for c,_,b in raw if c.lower()!='breadboard']
    
    # 검출된 컴포넌트를 핸들로 표시 (스케일 1.0 사용)
    handles = []
    for comp in comps:
        x1,y1,x2,y2 = comp['bbox']
        col = COLOR_MAP.get(comp['class'],'#6c757d')
        handles.append({
            "type":"rect","left":x1,"top":y1,
            "width":x2-x1,"height":y2-y1,
            "stroke":col,"fill":f"{col}33","cornerColor":col,"cornerSize":6
        })
    
    st.write(f"Detected {len(comps)} components. You can adjust their positions:")
    
    canvas = st_canvas(
        background_image=Image.fromarray(disp_rgb),
        width=DISPLAY_SIZE, height=DISPLAY_SIZE,
        drawing_mode="transform", initial_drawing={"objects":handles}, key="comp"
    )
    
    # 업데이트된 컴포넌트 좌표 (스케일 1.0)
    if canvas.json_data and canvas.json_data.get("objects"):
        updated = []
        for idx, o in enumerate(canvas.json_data["objects"]):
            l,t = o['left'], o['top']
            w_box,h_box = o['width'], o['height']
            updated.append({'class': comps[idx]['class'],
                            'bbox':(int(l),int(t),int(l+w_box),int(t+h_box))})
    else:
        updated = comps
    
    st.session_state.final_comps = updated
    st.success(f"✅ {len(updated)} components ready for pin detection.")
    show_navigation(4, next_enabled=True)

# 5) 구멍 검출 및 넷 클러스터링
def page_5_hole_detection():
    st.subheader("Step 5: Hole Detection & Net Clustering")
    
    if 'warped_raw' not in st.session_state:
        st.error("❌ No transformed image available.")
        show_navigation(5, next_enabled=False)
        return
    
    warped_raw = st.session_state.warped_raw
    
    with st.spinner("🔍 Detecting holes and clustering nets..."):
        # HoleDetector 초기화
        hd = HoleDetector(
            template_csv_path=os.path.join(BASE_DIR, "detector", "template_holes_complete.csv"),
            template_image_path=os.path.join(BASE_DIR, "detector", "breadboard18.jpg"),
            max_nn_dist=20.0
        )
        
        # 구멍 검출
        holes = hd.detect_holes(warped_raw)
        
        # 넷 클러스터링
        nets, row_nets = hd.get_board_nets(holes, base_img=warped_raw, show=False)
        
        # hole_to_net 맵 생성
        hole_to_net = {}
        for row_idx, clusters in row_nets:
            for entry in clusters:
                net_id = entry['net_id']
                for x, y in entry['pts']:
                    hole_to_net[(int(round(x)), int(round(y)))] = net_id
        
        # net_colors 생성
        rng = np.random.default_rng(1234)
        net_ids = sorted(set(hole_to_net.values()))
        net_colors = {nid: tuple(int(c) for c in rng.integers(0, 256, 3)) for nid in net_ids}
    
    st.success(f"✅ Detected {len(holes)} holes and {len(nets)} net clusters")
    
    # 시각화 (이미 640x640이므로 추가 처리 불필요)
    vis = warped_raw.copy()
    for (x, y), net_id in hole_to_net.items():
        color = net_colors[net_id]
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)
    
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), 
             caption=f"Detected Holes with Net Colors ({DISPLAY_SIZE}x{DISPLAY_SIZE})", 
             use_container_width=False, width=DISPLAY_SIZE)
    
    # 세션 변수 저장
    st.session_state.holes = holes
    st.session_state.nets = nets
    st.session_state.row_nets = row_nets
    st.session_state.hole_to_net = hole_to_net
    st.session_state.net_colors = net_colors
    
    show_navigation(5, next_enabled=True)

# 6) 핀 검출 및 시각화
def page_6_pin_detection():
    st.subheader("Step 6: Component Pin Detection")
    
    required_attrs = ['warped_raw', 'final_comps', 'holes', 'hole_to_net', 'net_colors']
    if not all(hasattr(st.session_state, attr) for attr in required_attrs):
        st.error("❌ Required data not available. Please complete previous steps.")
        show_navigation(6, next_enabled=False)
        return
    
    warped = st.session_state.warped_raw
    
    # 세션 상태에 pin_results가 없으면 자동 검출 실행
    if 'pin_results' not in st.session_state:
        with st.spinner("🔍 Detecting component pins..."):
            # 핀 검출
            pin_results = []
            for i, comp in enumerate(st.session_state.final_comps):
                try:
                    cls = comp['class']
                    x1, y1, x2, y2 = comp['bbox']
                    pins = []
                    
                    # 바운딩 박스 유효성 검증
                    if x1 >= x2 or y1 >= y2:
                        st.warning(f"Invalid bbox for component {i+1}: {comp['bbox']}")
                        pin_results.append({'class': cls, 'bbox': (x1, y1, x2, y2), 'pins': []})
                        continue
                    
                    # 이미지 범위 검증
                    h, w = warped.shape[:2]
                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        st.warning(f"Component {i+1} bbox out of image bounds")
                        # 클램핑
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                    
                    # 클래스별 핀 검출
                    if cls == 'Resistor':
                        try:
                            detected = ResistorEndpointDetector().extract(warped, (x1, y1, x2, y2))
                            pins = list(detected) if detected and detected[0] is not None else []
                        except Exception as e:
                            pins = []
                            
                    elif cls == 'LED':
                        try:
                            led_detector = LedEndpointDetector()
                            result = led_detector.extract(warped, (x1, y1, x2, y2), st.session_state.holes)
                            pins = result.get('endpoints', []) if result else []
                        except Exception as e:
                            pins = []
                            
                    elif cls == 'Diode':
                        try:
                            detected = DiodeEndpointDetector().extract(warped, (x1, y1, x2, y2))
                            pins = list(detected) if detected and detected[0] is not None else []
                        except Exception as e:
                            pins = []
                            
                    elif cls == 'IC':
                        try:
                            roi = warped[y1:y2, x1:x2]
                            if roi.size > 0:
                                ic_detector = ICChipPinDetector()
                                ics = ic_detector.detect(roi)
                                pins = [(x1 + px, y1 + py) for px, py in ics[0]['pin_points']] if ics else []
                            else:
                                pins = []
                        except Exception as e:
                            pins = []
                            
                    elif cls == 'Line_area':
                        try:
                            roi = warped[y1:y2, x1:x2]
                            if roi.size > 0:
                                wire_det = WireDetector()
                                if hasattr(wire_det, 'configure_white_thresholds'):
                                    wire_det.configure_white_thresholds(warped)
                                segs = wire_det.detect_wires(roi)
                                eps, _ = wire_det.select_best_endpoints(segs)
                                pins = [(x1 + pt[0], y1 + pt[1]) for pt in eps] if eps else []
                            else:
                                pins = []
                        except Exception as e:
                            pins = []
                    else:
                        pins = []
                    
                    # 핀 유효성 검증
                    valid_pins = []
                    for pin in pins:
                        if isinstance(pin, (tuple, list)) and len(pin) == 2:
                            px, py = pin
                            if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                                valid_pins.append((float(px), float(py)))
                    
                    pin_results.append({
                        'class': cls, 
                        'bbox': (x1, y1, x2, y2), 
                        'pins': valid_pins
                    })
                    
                except Exception as e:
                    pin_results.append({
                        'class': comp.get('class', 'Unknown'), 
                        'bbox': comp.get('bbox', (0, 0, 0, 0)), 
                        'pins': []
                    })
        
        st.session_state.pin_results = pin_results
    
    # 핀 검출 결과 요약
    pin_results = st.session_state.pin_results
    
    # 핀 검출 상태 요약
    total_comps = len(pin_results)
    detected_comps = len([pr for pr in pin_results if pr['pins']])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Components", total_comps)
    with col2:
        st.metric("Auto-detected Pins", detected_comps)
    with col3:
        st.metric("Manual Required", total_comps - detected_comps)
    
    # 핀과 넷 연결 시각화
    st.subheader("Pin-to-Net Visualization")
    disp_vis = warped.copy()
    hole_to_net = st.session_state.hole_to_net
    net_colors = st.session_state.net_colors
    
    # Union-Find 함수 (main.py와 동일)
    parent = {net_id: net_id for net_id in set(hole_to_net.values())}
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    # 각 컴포넌트의 핀 표시
    for i, comp in enumerate(pin_results):
        x1, y1, x2, y2 = comp['bbox']
        # 컴포넌트 박스 그리기
        color = COLOR_MAP.get(comp['class'], '#6c757d')
        bgr_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(disp_vis, (x1, y1), (x2, y2), bgr_color, 2)
        cv2.putText(disp_vis, f"{comp['class']}_{i+1}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        # 핀 표시
        for px, py in comp['pins']:
            if hole_to_net:
                closest = min(hole_to_net.keys(), key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
                net_id = find(hole_to_net[closest])
                color = net_colors.get(net_id, (255, 255, 255))
                cv2.circle(disp_vis, (int(px), int(py)), 6, color, -1)
                cv2.putText(disp_vis, str(net_id), (int(px)+8, int(py)-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    
    st.image(cv2.cvtColor(disp_vis, cv2.COLOR_BGR2RGB), 
             caption=f"Component Pins Mapped to Nets ({DISPLAY_SIZE}x{DISPLAY_SIZE})", 
             use_container_width=False, width=DISPLAY_SIZE)
    
    # 컴포넌트별 핀 편집 섹션
    st.subheader("Pin Detection & Manual Adjustment")
    
    # 컴포넌트 선택
    options = []
    for i, comp in enumerate(pin_results):
        expected = 8 if comp['class'] == 'IC' else 2
        detected = len(comp['pins'])
        status = "✅" if detected == expected else "⚠️" if detected > 0 else "❌"
        options.append(f"{status} {i+1}: {comp['class']} ({detected}/{expected} pins)")
    
    if not options:
        st.error("No components found for pin detection.")
        show_navigation(6, next_enabled=False)
        return
    
    # 선택된 컴포넌트
    selected_idx = st.selectbox("Select component to adjust pins:", 
                               range(len(pin_results)), 
                               format_func=lambda i: options[i])
    
    comp = pin_results[selected_idx]
    expected = 8 if comp['class'] == 'IC' else 2
    detected = len(comp['pins'])
    
    st.info(f"**{comp['class']}** | Expected: {expected} pins | Detected: {detected} pins")
    
    # ROI 표시 및 핀 편집
    x1, y1, x2, y2 = comp['bbox']
    roi = warped[y1:y2, x1:x2]
    
    # ROI 확대 (더 큰 캔버스로 표시)
    roi_scale = 2.0  # 2배 확대
    roi_h, roi_w = roi.shape[:2]
    roi_disp = cv2.resize(roi, (int(roi_w * roi_scale), int(roi_h * roi_scale)))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Current Pin Configuration:**")
        
        # 현재 핀 위치를 캔버스에 표시
        pin_objects = []
        for j, (px, py) in enumerate(comp['pins']):
            # 상대 좌표로 변환 후 확대
            rel_x = (px - x1) * roi_scale
            rel_y = (py - y1) * roi_scale
            pin_objects.append({
                "type": "circle",
                "left": rel_x - 6, "top": rel_y - 6,
                "width": 12, "height": 12,
                "fill": "red", "stroke": "darkred", "strokeWidth": 2
            })
        
        # 핀 클릭 캔버스
        canvas_result = st_canvas(
            background_image=Image.fromarray(cv2.cvtColor(roi_disp, cv2.COLOR_BGR2RGB)),
            width=roi_disp.shape[1], 
            height=roi_disp.shape[0],
            drawing_mode="point", 
            point_display_radius=6,
            initial_drawing={"objects": pin_objects},
            key=f"pin_canvas_{selected_idx}"
        )
        
        # 캔버스 조작 가이드
        st.markdown("""
        **Instructions:**
        - Click to add new pins
        - Drag existing pins to move them
        - Right-click to delete pins
        """)
    
    with col2:
        st.write("**Pin Actions:**")
        
        # 업데이트된 핀 처리
        new_pins = []
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "circle":
                    # 캔버스 좌표를 원본 좌표로 변환
                    rel_x = (obj["left"] + obj["width"]/2) / roi_scale
                    rel_y = (obj["top"] + obj["height"]/2) / roi_scale
                    abs_x = rel_x + x1
                    abs_y = rel_y + y1
                    new_pins.append((abs_x, abs_y))
        
        # 핀 개수 상태 표시
        st.metric("Current Pin Count", len(new_pins), delta=len(new_pins) - expected)
        
        # 핀 좌표 미세 조정
        if new_pins:
            st.write("**Fine-tune Pin Coordinates:**")
            adjusted_pins = []
            for j, (px, py) in enumerate(new_pins):
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    new_x = st.number_input(f"Pin {j+1} X", 
                                          value=float(px), 
                                          step=0.5,
                                          key=f"pin_x_{selected_idx}_{j}")
                with sub_col2:
                    new_y = st.number_input(f"Pin {j+1} Y", 
                                          value=float(py), 
                                          step=0.5,
                                          key=f"pin_y_{selected_idx}_{j}")
                adjusted_pins.append((new_x, new_y))
            new_pins = adjusted_pins
        
        # 핀 업데이트 버튼
        if st.button(f"Update Pins for {comp['class']} #{selected_idx+1}", 
                    key=f"update_pins_{selected_idx}"):
            # 핀 개수 검증
            if len(new_pins) == expected:
                comp['pins'] = new_pins
                st.success(f"✅ Updated {len(new_pins)} pins for {comp['class']}")
                st.rerun()
            else:
                st.error(f"❌ Expected {expected} pins, got {len(new_pins)}. Please adjust.")
        
        # 자동 재검출 버튼
        if st.button(f"Auto Re-detect Pins", key=f"redetect_{selected_idx}"):
            # 해당 컴포넌트만 다시 검출
            try:
                cls = comp['class']
                if cls == 'Resistor':
                    detected = ResistorEndpointDetector().extract(warped, (x1, y1, x2, y2))
                    new_pins = list(detected) if detected and detected[0] is not None else []
                elif cls == 'LED':
                    result = LedEndpointDetector().extract(warped, (x1, y1, x2, y2), st.session_state.holes)
                    new_pins = result.get('endpoints', []) if result else []
                elif cls == 'Diode':
                    detected = DiodeEndpointDetector().extract(warped, (x1, y1, x2, y2))
                    new_pins = list(detected) if detected and detected[0] is not None else []
                elif cls == 'IC':
                    roi_img = warped[y1:y2, x1:x2]
                    ics = ICChipPinDetector().detect(roi_img)
                    new_pins = [(x1 + px, y1 + py) for px, py in ics[0]['pin_points']] if ics else []
                elif cls == 'Line_area':
                    roi_img = warped[y1:y2, x1:x2]
                    wire_det = WireDetector()
                    segs = wire_det.detect_wires(roi_img)
                    eps, _ = wire_det.select_best_endpoints(segs)
                    new_pins = [(x1 + pt[0], y1 + pt[1]) for pt in eps] if eps else []
                else:
                    new_pins = []
                
                # 유효한 핀만 필터링
                valid_pins = []
                for pin in new_pins:
                    if isinstance(pin, (tuple, list)) and len(pin) == 2:
                        px, py = pin
                        if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                            valid_pins.append((float(px), float(py)))
                
                comp['pins'] = valid_pins
                st.success(f"✅ Re-detected {len(valid_pins)} pins")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Auto re-detection failed: {str(e)}")
        
        # 핀 초기화 버튼
        if st.button(f"Clear All Pins", key=f"clear_{selected_idx}"):
            comp['pins'] = []
            st.success("✅ Cleared all pins")
            st.rerun()
    
    # 전체 핀 검출 상태 확인
    all_good = all(
        len(pr['pins']) == (8 if pr['class'] == 'IC' else 2) 
        for pr in pin_results
    )
    
    if all_good:
        st.success("🎉 All components have the correct number of pins!")
    else:
        incomplete = [
            f"{i+1}: {pr['class']} ({len(pr['pins'])}/{8 if pr['class'] == 'IC' else 2})"
            for i, pr in enumerate(pin_results)
            if len(pr['pins']) != (8 if pr['class'] == 'IC' else 2)
        ]
        st.warning(f"⚠️ Components with missing pins: {', '.join(incomplete)}")
    
    show_navigation(6, next_enabled=True)

# 7) 값 입력
def page_7_value_input():
    st.subheader("Step 7: Component Values")
    
    if 'pin_results' not in st.session_state:
        st.error("❌ No pin detection results available.")
        show_navigation(7, next_enabled=False)
        return
    
    st.write("Enter values for resistive components:")
    
    vals = {}
    for i, pr in enumerate(st.session_state.pin_results):
        if pr['class'] == 'Resistor':
            key = tuple(pr['bbox'])
            vals[key] = st.number_input(
                f"Resistor {i+1} value (Ω)",
                value=100.0, min_value=0.1, step=10.0,
                key=f"resistor_{i}"
            )
        else:
            vals[tuple(pr['bbox'])] = 0.0
    
    st.session_state.comp_values = vals
    
    # 입력된 값 요약
    resistor_count = len([pr for pr in st.session_state.pin_results if pr['class'] == 'Resistor'])
    if resistor_count > 0:
        st.success(f"✅ Values set for {resistor_count} resistors")
    else:
        st.info("No resistors detected in this circuit")
    
    show_navigation(7, next_enabled=True)

# 8) 핀 좌표 수동 조정
def page_8_manual_pin_adjustment():
    st.subheader("Step 8: Manual Pin Coordinate Adjustment")
    
    if 'pin_results' not in st.session_state:
        st.error("❌ No pin detection results available.")
        show_navigation(8, next_enabled=False)
        return
    
    st.write("Fine-tune pin coordinates if needed:")
    
    fixed = []
    for i, pr in enumerate(st.session_state.pin_results):
        with st.expander(f"{pr['class']} #{i+1} - {len(pr['pins'])} pins"):
            coords = []
            for j, (px, py) in enumerate(pr['pins']):
                col1, col2 = st.columns(2)
                with col1:
                    x = st.number_input(f"Pin {j+1} X", value=float(px), 
                                       step=1.0, key=f"x_{i}_{j}")
                with col2:
                    y = st.number_input(f"Pin {j+1} Y", value=float(py), 
                                       step=1.0, key=f"y_{i}_{j}")
                coords.append((x, y))
            
            fixed.append({
                'class': pr['class'],
                'bbox': pr['bbox'], 
                'pins': coords,
                'value': st.session_state.comp_values.get(tuple(pr['bbox']), 0.0)
            })
    
    st.session_state.fixed_pins = fixed
    st.success("✅ Pin coordinates finalized")
    
    show_navigation(8, next_enabled=True)

# 9) 전원 단자 선택
def page_9_power_selection():
    st.subheader("Step 9: Power Terminal Selection")
    
    if 'warped_raw' not in st.session_state:
        st.error("❌ No image available.")
        show_navigation(9, next_enabled=False)
        return
    
    # 전압 입력
    voltage = st.number_input("Supply Voltage (V)", value=5.0, min_value=0.1, step=0.1)
    st.session_state.voltage = voltage
    
    # 이미지에서 전원 단자 선택 (640x640)
    warped = st.session_state.warped_raw
    
    st.write("Click to select power terminals (+ and - terminals):")
    st.info("First click: Positive terminal, Second click: Negative terminal")
    
    canvas = st_canvas(
        background_image=Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)),
        width=DISPLAY_SIZE, height=DISPLAY_SIZE,
        drawing_mode="point", key="power_sel"
    )
    
    # 선택된 포인트들 (스케일 1.0)
    power_pts = []
    if canvas.json_data and canvas.json_data.get("objects"):
        for o in canvas.json_data["objects"]:
            x = o['left'] + o['width']/2
            y = o['top'] + o['height']/2
            power_pts.append((x, y))
    
    if len(power_pts) >= 2:
        st.success(f"✅ Selected {len(power_pts)} power terminals")
        st.session_state.power_points = power_pts[:2]  # 처음 2개만 사용
        show_navigation(9, next_enabled=True)
    else:
        st.warning("Please select at least 2 power terminals")
        show_navigation(9, next_enabled=False)

# 10) 회로 생성
def page_10_circuit_generation():
    st.subheader("Step 10: Circuit Generation")
    
    required_keys = ['fixed_pins', 'holes', 'hole_to_net', 'comp_values', 'power_points', 'voltage']
    missing = [k for k in required_keys if k not in st.session_state]
    
    if missing:
        st.error(f"❌ Missing required data: {missing}")
        show_navigation(10, next_enabled=False)
        return
    
    with st.spinner("⚡ Generating circuit diagram and SPICE file..."):
        try:
            # 전원 쌍 변환 및 단자 찾기 (main.py의 로직 참조)
            all_endpoints = [pt for comp in st.session_state.fixed_pins for pt in comp['pins']]
            
            power_pairs = []
            voltage = st.session_state.voltage
            
            # 클릭한 위치에서 가장 가까운 실제 엔드포인트 찾기
            for plus_pt, minus_pt in [(st.session_state.power_points[0], st.session_state.power_points[1])]:
                closest_plus = min(all_endpoints, key=lambda p: (p[0]-plus_pt[0])**2 + (p[1]-plus_pt[1])**2)
                closest_minus = min(all_endpoints, key=lambda p: (p[0]-minus_pt[0])**2 + (p[1]-minus_pt[1])**2)
                
                # nearest_net 함수 구현
                def find_nearest_net(pt):
                    hole_to_net = st.session_state.hole_to_net
                    closest = min(hole_to_net.keys(), key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2)
                    return hole_to_net[closest]
                
                net_plus = find_nearest_net(closest_plus)
                net_minus = find_nearest_net(closest_minus)
                
                # schemdraw용 그리드 좌표 변환 (640x640 기준)
                img_w = DISPLAY_SIZE
                comp_count = len([c for c in st.session_state.fixed_pins if c['class'] != 'Line_area'])
                grid_width = comp_count * 2 + 2
                x_plus_grid = closest_plus[0] / img_w * grid_width
                x_minus_grid = closest_minus[0] / img_w * grid_width
                
                power_pairs.append((net_plus, x_plus_grid, net_minus, x_minus_grid))
            
            # 와이어 연결 처리
            wires = []
            for comp in st.session_state.fixed_pins:
                if comp['class'] == 'Line_area' and len(comp['pins']) == 2:
                    net1 = find_nearest_net(comp['pins'][0])
                    net2 = find_nearest_net(comp['pins'][1])
                    if net1 != net2:
                        wires.append((net1, net2))
            
            # 회로 생성 (640x640 크기로)
            mapped, hole_to_net = generate_circuit(
                all_comps=st.session_state.fixed_pins,
                holes=st.session_state.holes,
                wires=wires,
                voltage=voltage,
                output_spice=os.path.join(BASE_DIR, "circuit.spice"),
                output_img=os.path.join(BASE_DIR, "circuit.jpg"),
                hole_to_net=st.session_state.hole_to_net,
                power_pairs=power_pairs
            )
            
            st.session_state.circuit_components = mapped
            st.success("✅ Circuit generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Circuit generation failed: {str(e)}")
            show_navigation(10, next_enabled=False)
            return
    
    # 결과 표시
    col1, col2 = st.columns(2)
    
    with col1:
        img_path = os.path.join(BASE_DIR, "circuit.jpg")
        if os.path.exists(img_path):
            circuit_img = cv2.imread(img_path)
            if circuit_img is not None:
                st.image(cv2.cvtColor(circuit_img, cv2.COLOR_BGR2RGB), 
                        caption="Generated Circuit Diagram", use_container_width=True)
                st.session_state.circuit_img = circuit_img
    
    with col2:
        spice_path = os.path.join(BASE_DIR, "circuit.spice")
        if os.path.exists(spice_path):
            st.success("SPICE file generated!")
            with open(spice_path, 'r') as f:
                spice_content = f.read()
                st.text_area("SPICE Netlist", spice_content, height=200)
            
            with open(spice_path, 'rb') as f:
                st.download_button(
                    "📥 Download SPICE File",
                    f.read(),
                    file_name="circuit.spice",
                    mime="text/plain"
                )
            st.session_state.spice_file = spice_path
    
    show_navigation(10, next_enabled=True)

# 11) 오류 검사
def page_11_error_checking():
    st.subheader("Step 11: Circuit Error Checking")
    
    if 'spice_file' not in st.session_state or not os.path.exists(st.session_state.spice_file):
        st.error("❌ No SPICE file available for error checking.")
        show_navigation(11, next_enabled=False)
        return
    
    if 'circuit_components' not in st.session_state:
        st.error("❌ No circuit components available for error checking.")
        show_navigation(11, next_enabled=False)
        return
    
    with st.spinner("🔍 Checking for circuit errors..."):
        try:
            # 컴포넌트와 넷 매핑 생성
            components = st.session_state.circuit_components
            nets_mapping = {}
            
            for comp in components:
                n1, n2 = comp['nodes']
                nets_mapping.setdefault(n1, []).append(comp['name'])
                nets_mapping.setdefault(n2, []).append(comp['name'])
            
            # 전원 정보 추가 (power_pairs에서 ground_net 추출)
            power_pairs = getattr(st.session_state, 'power_pairs', [])
            ground_nodes = {power_pairs[0][2]} if power_pairs else set()
            
            # ErrorChecker 실행
            checker = ErrorChecker(components, nets_mapping, ground_nodes=ground_nodes)
            errors = checker.run_all_checks()
            
            st.session_state.circuit_errors = errors
            
        except Exception as e:
            st.error(f"❌ Error checking failed: {str(e)}")
            show_navigation(11, next_enabled=True)  # 오류가 있어도 다음 단계로 진행 가능
            return
    
    # 오류 결과 표시
    if errors:
        st.warning(f"⚠️ Found {len(errors)} potential issues:")
        
        error_df = pd.DataFrame([
            {"Error Type": "Circuit Error", "Description": error}
            for error in errors
        ])
        st.dataframe(error_df, use_container_width=True)
        
        # 오류 유형별 분류
        error_types = {}
        for error in errors:
            if "Open circuit" in error:
                error_types.setdefault("Open Circuits", []).append(error)
            elif "Short circuit" in error:
                error_types.setdefault("Short Circuits", []).append(error)
            elif "Floating" in error:
                error_types.setdefault("Floating Components", []).append(error)
            elif "voltage source" in error.lower():
                error_types.setdefault("Power Issues", []).append(error)
            else:
                error_types.setdefault("Other Issues", []).append(error)
        
        for error_type, error_list in error_types.items():
            with st.expander(f"{error_type} ({len(error_list)})"):
                for error in error_list:
                    st.write(f"• {error}")
    else:
        st.success("✅ No circuit errors detected! Your circuit looks good.")
    
    show_navigation(11, next_enabled=True)

# 12) 최종 요약
def page_12_summary():
    st.subheader("Step 12: Project Summary")
    
    # 프로젝트 완료 메시지
    st.balloons()
    st.success("🎉 Breadboard to Schematic conversion completed!")
    
    # 요약 정보
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Analysis Results")
        
        # 컴포넌트 요약
        if 'final_comps' in st.session_state:
            comp_summary = {}
            for comp in st.session_state.final_comps:
                comp_type = comp['class']
                comp_summary[comp_type] = comp_summary.get(comp_type, 0) + 1
            
            st.write("**Detected Components:**")
            for comp_type, count in comp_summary.items():
                st.write(f"• {comp_type}: {count}")
        
        # 홀과 넷 요약
        if 'holes' in st.session_state:
            st.write(f"**Holes Detected:** {len(st.session_state.holes)}")
        
        if 'nets' in st.session_state:
            st.write(f"**Net Clusters:** {len(st.session_state.nets)}")
        
        # 전압 설정
        if 'voltage' in st.session_state:
            st.write(f"**Supply Voltage:** {st.session_state.voltage}V")
    
    with col2:
        st.markdown("### 📁 Generated Files")
        
        # 다운로드 링크들
        img_path = os.path.join(BASE_DIR, "circuit.jpg")
        spice_path = os.path.join(BASE_DIR, "circuit.spice")
        
        if os.path.exists(img_path):
            st.success("✅ Circuit diagram generated")
            with open(img_path, 'rb') as f:
                st.download_button(
                    "📥 Download Circuit Image",
                    f.read(),
                    file_name="circuit_diagram.jpg",
                    mime="image/jpeg"
                )
        
        if os.path.exists(spice_path):
            st.success("✅ SPICE netlist generated")
            with open(spice_path, 'rb') as f:
                st.download_button(
                    "📥 Download SPICE File",
                    f.read(),
                    file_name="circuit.spice",
                    mime="text/plain"
                )
        
        # GraphML 파일이 있다면
        graphml_path = os.path.join(BASE_DIR, "circuit.graphml")
        if os.path.exists(graphml_path):
            st.success("✅ Circuit graph generated")
            with open(graphml_path, 'rb') as f:
                st.download_button(
                    "📥 Download GraphML",
                    f.read(),
                    file_name="circuit.graphml",
                    mime="application/xml"
                )
    
    # 오류 요약
    if 'circuit_errors' in st.session_state:
        if st.session_state.circuit_errors:
            st.warning(f"⚠️ {len(st.session_state.circuit_errors)} potential issues detected")
            with st.expander("View Error Details"):
                for error in st.session_state.circuit_errors:
                    st.write(f"• {error}")
        else:
            st.success("✅ No circuit errors detected")
    
    # 최종 이미지 표시 (640x640)
    if 'circuit_img' in st.session_state:
        st.markdown("### 🔌 Final Circuit Diagram")
        st.image(cv2.cvtColor(st.session_state.circuit_img, cv2.COLOR_BGR2RGB), 
                caption=f"Generated Schematic ({DISPLAY_SIZE}x{DISPLAY_SIZE})", 
                use_container_width=False, width=DISPLAY_SIZE)
    
    # 재시작 버튼
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 Start New Project", key="restart", use_container_width=True):
            # 세션 상태 초기화
            for key in list(st.session_state.keys()):
                if key != 'page':  # 페이지는 1로 리셋
                    del st.session_state[key]
            st.session_state.page = 1
            st.rerun()
    
    # 이전 버튼만 표시 (다음 버튼 없음)
    cols = st.columns([1, 2, 1])
    if cols[0].button("◀️ Previous", key="prev_12"):
        st.session_state.page = 11
        st.rerun()
    
    with cols[1]:
        st.progress(1.0)
        st.write("Project Complete!")

# 메인 앱 라우팅
def main():
    st.set_page_config(
        page_title="Breadboard to Schematic", 
        page_icon="🔌",
        layout="wide"
    )
    
    # 사이드바에 페이지 네비게이션
    with st.sidebar:
        st.title("🔌 Navigation")
        page_names = [
            "1. Upload Image",
            "2. Adjust Corners", 
            "3. View Transform",
            "4. Edit Components",
            "5. Detect Holes",
            "6. Detect Pins",
            "7. Enter Values",
            "8. Adjust Pins",
            "9. Select Power",
            "10. Generate Circuit",
            "11. Check Errors",
            "12. Summary"
        ]
        
        current_page = st.session_state.page
        for i, name in enumerate(page_names, 1):
            if i == current_page:
                st.markdown(f"**➤ {name}**")
            elif i < current_page:
                st.markdown(f"✅ {name}")
            else:
                st.markdown(f"⏸️ {name}")
        
        st.markdown("---")
        st.markdown("### 📋 Progress")
        progress = (current_page - 1) / (TOTAL_PAGES - 1)
        st.progress(progress)
        st.write(f"{progress*100:.0f}% Complete")
    
    # 메인 페이지 컨텐츠
    page = st.session_state.page
    
    if page == 1:
        page_1_upload()
    elif page == 2:
        page_2_corner_adjust()
    elif page == 3:
        page_3_transformed()
    elif page == 4:
        page_4_component_edit()
    elif page == 5:
        page_5_hole_detection()
    elif page == 6:
        page_6_pin_detection()
    elif page == 7:
        page_7_value_input()
    elif page == 8:
        page_8_manual_pin_adjustment()
    elif page == 9:
        page_9_power_selection()
    elif page == 10:
        page_10_circuit_generation()
    elif page == 11:
        page_11_error_checking()
    elif page == 12:
        page_12_summary()
    else:
        st.error("Invalid page number. Restarting...")
        st.session_state.page = 1
        st.rerun()

if __name__ == "__main__":
    main()