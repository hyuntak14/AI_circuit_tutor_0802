import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
<<<<<<< HEAD
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
=======
>>>>>>> 43c99fd46a94b88ad6ea9a12de5345dc93c72d7d

from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.hole_detector import HoleDetector
from detector.resistor_detector import ResistorEndpointDetector
from detector.led_detector import LedEndpointDetector
from detector.diode_detector import ResistorEndpointDetector as DiodeEndpointDetector
from detector.ic_chip_detector import ICChipPinDetector
from detector.wire_detector import WireDetector
from circuit_generator import generate_circuit
from checker.error_checker import ErrorChecker
<<<<<<< HEAD
from diagram import get_n_clicks
=======
>>>>>>> 43c99fd46a94b88ad6ea9a12de5345dc93c72d7d

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
<<<<<<< HEAD
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
=======

# Utility: 이미지 축소
@st.cache_data
def resize_image(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale
>>>>>>> 43c99fd46a94b88ad6ea9a12de5345dc93c72d7d

# 1) 업로드 & 원본 표시
def page_1_upload():
    st.title("📸 Breadboard to Schematic")
<<<<<<< HEAD
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
    
    # 초기 컴포넌트 검출 (한 번만 실행)
    if 'detected_comps' not in st.session_state:
        with st.spinner("🔍 Detecting components..."):
            detector = FasterRCNNDetector(model_path=MODEL_PATH)
            raw = detector.detect(warped)
            detected_comps = [{'class':c,'bbox':b} for c,_,b in raw if c.lower()!='breadboard']
            st.session_state.detected_comps = detected_comps
    
    # 수정 가능한 컴포넌트 목록 (세션에서 관리)
    if 'editable_comps' not in st.session_state:
        st.session_state.editable_comps = st.session_state.detected_comps.copy()
    
    editable_comps = st.session_state.editable_comps
    
    # 컴포넌트 클래스 옵션
    CLASS_OPTIONS = ['Resistor', 'LED', 'Diode', 'IC', 'Line_area', 'Capacitor']
    
    # 두 가지 모드: 위치 조정 및 새 컴포넌트 추가
    mode = st.radio("🛠️ Editing Mode", ["Adjust Positions", "Add New Component"], horizontal=True)
    
    if mode == "Adjust Positions":
        # 위치 조정 모드 - transform으로 기존 컴포넌트 위치 수정
        st.write("**Drag components to adjust their positions:**")
        
        # 검출된 컴포넌트를 핸들로 표시 (스케일 1.0 사용)
        handles = []
        for i, comp in enumerate(editable_comps):
            x1,y1,x2,y2 = comp['bbox']
            col = COLOR_MAP.get(comp['class'],'#6c757d')
            handles.append({
                "type":"rect","left":x1,"top":y1,
                "width":x2-x1,"height":y2-y1,
                "stroke":col,"fill":f"{col}33","cornerColor":col,"cornerSize":6
            })
        
        canvas = st_canvas(
            background_image=Image.fromarray(disp_rgb),
            width=DISPLAY_SIZE, height=DISPLAY_SIZE,
            drawing_mode="transform", 
            initial_drawing={"objects":handles}, 
            key="comp_position"
        )
        
        # 업데이트된 컴포넌트 좌표 (스케일 1.0)
        if canvas.json_data and canvas.json_data.get("objects"):
            for idx, o in enumerate(canvas.json_data["objects"]):
                if idx < len(editable_comps):
                    l,t = o['left'], o['top']
                    w_box,h_box = o['width'], o['height']
                    editable_comps[idx]['bbox'] = (int(l),int(t),int(l+w_box),int(t+h_box))
    
    else:
        # 새 컴포넌트 추가 모드 - rect로 새 영역 그리기
        st.write("**Draw rectangles to add new components:**")
        
        # 기존 컴포넌트는 배경에 표시
        vis_img = warped.copy()
        for comp in editable_comps:
            x1,y1,x2,y2 = comp['bbox']
            col = COLOR_MAP.get(comp['class'],'#6c757d')
            bgr_color = tuple(int(col.lstrip('#')[i:i+2], 16) for i in (4,2,0))
            cv2.rectangle(vis_img, (x1,y1), (x2,y2), bgr_color, 2)
            cv2.putText(vis_img, comp['class'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        canvas = st_canvas(
            background_image=Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)),
            width=DISPLAY_SIZE, height=DISPLAY_SIZE,
            drawing_mode="rect",
            stroke_width=2,
            stroke_color="#ff0000",
            fill_color="rgba(255,0,0,0.1)",
            key="comp_add"
        )
        
        # 새로 그린 컴포넌트 추가
        if canvas.json_data and canvas.json_data.get("objects"):
            for obj in canvas.json_data["objects"]:
                x1 = int(obj["left"])
                y1 = int(obj["top"])
                x2 = int(obj["left"] + obj["width"])
                y2 = int(obj["top"] + obj["height"])
                
                # 너무 작은 박스는 무시
                if abs(x2-x1) > 20 and abs(y2-y1) > 20:
                    # 기본 클래스로 추가 (나중에 수정 가능)
                    new_comp = {'class': 'Resistor', 'bbox': (x1,y1,x2,y2)}
                    # 중복 체크
                    is_duplicate = any(
                        abs(comp['bbox'][0] - x1) < 10 and abs(comp['bbox'][1] - y1) < 10
                        for comp in editable_comps
                    )
                    if not is_duplicate:
                        editable_comps.append(new_comp)
                        st.success(f"Added new component at ({x1},{y1})")
    
    # 컴포넌트 리스트 및 편집 섹션
    st.subheader("Component List & Editing")
    
    if not editable_comps:
        st.info("No components detected. Use 'Add New Component' mode to add components manually.")
        show_navigation(4, next_enabled=False)
        return
    
    # 컴포넌트 편집 테이블
    st.write(f"**Total Components: {len(editable_comps)}**")
    
    # 각 컴포넌트에 대한 편집 옵션
    components_to_delete = []
    
    for i, comp in enumerate(editable_comps):
        with st.expander(f"Component {i+1}: {comp['class']} {comp['bbox']}", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # 클래스 변경
                current_class = comp['class']
                class_idx = CLASS_OPTIONS.index(current_class) if current_class in CLASS_OPTIONS else 0
                new_class = st.selectbox(
                    "Component Type",
                    CLASS_OPTIONS,
                    index=class_idx,
                    key=f"class_{i}"
                )
                comp['class'] = new_class
            
            with col2:
                # 바운딩 박스 미세 조정
                x1, y1, x2, y2 = comp['bbox']
                
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    new_x1 = st.number_input("X1", value=x1, step=1, key=f"x1_{i}")
                    new_y1 = st.number_input("Y1", value=y1, step=1, key=f"y1_{i}")
                with sub_col2:
                    new_x2 = st.number_input("X2", value=x2, step=1, key=f"x2_{i}")
                    new_y2 = st.number_input("Y2", value=y2, step=1, key=f"y2_{i}")
                
                comp['bbox'] = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
            
            with col3:
                # 삭제 버튼
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    components_to_delete.append(i)
    
    # 삭제 처리 (역순으로 삭제하여 인덱스 문제 방지)
    for idx in sorted(components_to_delete, reverse=True):
        removed = editable_comps.pop(idx)
        st.success(f"Deleted {removed['class']} component")
        st.rerun()
    
    # 전체 작업 버튼들
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset to Auto-detected", key="reset_comps"):
            st.session_state.editable_comps = st.session_state.detected_comps.copy()
            st.success("Reset to original detection results")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear All", key="clear_all"):
            st.session_state.editable_comps = []
            st.success("Cleared all components")
            st.rerun()
    
    with col3:
        # 자동 재검출
        if st.button("🔍 Re-detect", key="redetect"):
            with st.spinner("Re-detecting components..."):
                detector = FasterRCNNDetector(model_path=MODEL_PATH)
                raw = detector.detect(warped)
                new_comps = [{'class':c,'bbox':b} for c,_,b in raw if c.lower()!='breadboard']
                st.session_state.detected_comps = new_comps
                st.session_state.editable_comps = new_comps.copy()
                st.success(f"Re-detected {len(new_comps)} components")
                st.rerun()
    
    # 최종 결과 시각화
    st.subheader("Final Component Layout")
    final_vis = warped.copy()
    for i, comp in enumerate(editable_comps):
        x1,y1,x2,y2 = comp['bbox']
        col = COLOR_MAP.get(comp['class'],'#6c757d')
        bgr_color = tuple(int(col.lstrip('#')[i:i+2], 16) for i in (4,2,0))
        cv2.rectangle(final_vis, (x1,y1), (x2,y2), bgr_color, 2)
        cv2.putText(final_vis, f"{i+1}:{comp['class']}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
    
    st.image(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB), 
             caption=f"Final Layout ({len(editable_comps)} components)", 
             use_container_width=False, width=DISPLAY_SIZE)
    
    # 최종 컴포넌트 목록을 세션에 저장
    st.session_state.final_comps = editable_comps
    
    st.success(f"✅ {len(editable_comps)} components ready for pin detection.")
    show_navigation(4, next_enabled=len(editable_comps) > 0)

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
            pin_results = []
            for i, comp in enumerate(st.session_state.final_comps):
                try:
                    cls = comp['class']
                    x1, y1, x2, y2 = comp['bbox']
                    pins = []
                    
                    # 바운딩 박스 유효성 검증
                    if x1 >= x2 or y1 >= y2:
                        pins = []
                    # 이미지 범위 검증
                    elif x1 < 0 or y1 < 0 or x2 > DISPLAY_SIZE or y2 > DISPLAY_SIZE:
                        pins = []
                    else:
                        # 클래스별 핀 검출
                        if cls == 'Resistor':
                            try:
                                detected = ResistorEndpointDetector().extract(warped, (x1, y1, x2, y2))
                                pins = list(detected) if detected and detected[0] is not None else []
                            except:
                                pins = []
                        elif cls == 'LED':
                            try:
                                result = LedEndpointDetector().extract(warped, (x1, y1, x2, y2), st.session_state.holes)
                                pins = result.get('endpoints', []) if result else []
                            except:
                                pins = []
                        elif cls == 'Diode':
                            try:
                                detected = DiodeEndpointDetector().extract(warped, (x1, y1, x2, y2))
                                pins = list(detected) if detected and detected[0] is not None else []
                            except:
                                pins = []
                        elif cls == 'IC':
                            try:
                                roi = warped[y1:y2, x1:x2]
                                if roi.size > 0:
                                    ics = ICChipPinDetector().detect(roi)
                                    pins = [(x1 + px, y1 + py) for px, py in ics[0]['pin_points']] if ics else []
                            except:
                                pins = []
                        elif cls == 'Line_area':
                            try:
                                roi = warped[y1:y2, x1:x2]
                                if roi.size > 0:
                                    wire_det = WireDetector()
                                    segs = wire_det.detect_wires(roi)
                                    eps, _ = wire_det.select_best_endpoints(segs)
                                    pins = [(x1 + pt[0], y1 + pt[1]) for pt in eps] if eps else []
                            except:
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
                except:
                    pin_results.append({
                        'class': comp.get('class', 'Unknown'), 
                        'bbox': comp.get('bbox', (0, 0, 0, 0)), 
                        'pins': []
                    })
        
        st.session_state.pin_results = pin_results
    
    pin_results = st.session_state.pin_results
    
    # Union-Find 함수 (main.py와 동일)
    parent = {net_id: net_id for net_id in set(st.session_state.hole_to_net.values())}
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    # Step 1: 전체 이미지에 모든 컴포넌트와 핀 표시
    st.subheader("🔍 Component Overview - Click to Edit Pins")
    
    # 전체 이미지에 컴포넌트와 핀 표시
    overview_img = warped.copy()
    
    # 컴포넌트 박스와 핀 그리기
    for i, comp in enumerate(pin_results):
        x1, y1, x2, y2 = comp['bbox']
        expected = 8 if comp['class'] == 'IC' else 2
        detected = len(comp['pins'])
        
        # 상태에 따른 박스 색상
        if detected == expected:
            color = (0, 255, 0)  # 초록 - 완료
        elif detected > 0:
            color = (0, 165, 255)  # 주황 - 부분 완료
        else:
            color = (0, 0, 255)  # 빨강 - 미완료
        
        # 컴포넌트 박스 그리기
        cv2.rectangle(overview_img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(overview_img, f"{i+1}: {comp['class']}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(overview_img, f"({detected}/{expected})", (x1, y2+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 핀 표시
        for px, py in comp['pins']:
            if st.session_state.hole_to_net:
                closest = min(st.session_state.hole_to_net.keys(), 
                            key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
                net_id = find(st.session_state.hole_to_net[closest])
                net_color = st.session_state.net_colors.get(net_id, (255, 255, 255))
                cv2.circle(overview_img, (int(px), int(py)), 6, net_color, -1)
                cv2.circle(overview_img, (int(px), int(py)), 6, (0, 0, 0), 2)
    
    st.image(cv2.cvtColor(overview_img, cv2.COLOR_BGR2RGB), 
             caption="Click on a component box below to edit its pins", 
             use_container_width=False, width=DISPLAY_SIZE)
    
    # Step 2: 컴포넌트 선택을 위한 버튼들
    st.subheader("📋 Select Component to Edit")
    
    # 상태별 카운트
    total = len(pin_results)
    completed = sum(1 for comp in pin_results if len(comp['pins']) == (8 if comp['class'] == 'IC' else 2))
    partial = sum(1 for comp in pin_results if 0 < len(comp['pins']) < (8 if comp['class'] == 'IC' else 2))
    missing = total - completed - partial
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("Completed", completed, delta=f"{completed}/{total}")
    with col3:
        st.metric("Partial", partial, delta="⚠️" if partial > 0 else "")
    with col4:
        st.metric("Missing", missing, delta="❌" if missing > 0 else "✅")
    
    # 컴포넌트 선택 버튼들 (3열로 배치)
    cols = st.columns(3)
    selected_comp_idx = None
    
    for i, comp in enumerate(pin_results):
        col_idx = i % 3
        expected = 8 if comp['class'] == 'IC' else 2
        detected = len(comp['pins'])
        
        # 상태 이모지
        if detected == expected:
            status = "✅"
        elif detected > 0:
            status = "⚠️"
        else:
            status = "❌"
        
        button_label = f"{status} {i+1}: {comp['class']}\n({detected}/{expected})"
        
        if cols[col_idx].button(button_label, key=f"select_comp_{i}", use_container_width=True):
            selected_comp_idx = i
    
    # Step 3: 선택된 컴포넌트의 핀 편집
    if selected_comp_idx is not None:
        st.session_state.selected_component = selected_comp_idx
    
    if 'selected_component' in st.session_state:
        comp_idx = st.session_state.selected_component
        comp = pin_results[comp_idx]
        expected = 8 if comp['class'] == 'IC' else 2
        
        st.subheader(f"✏️ Edit Pins: {comp['class']} #{comp_idx+1}")
        st.info(f"Expected: {expected} pins | Current: {len(comp['pins'])} pins")
        
        # 컴포넌트 ROI 추출 및 확대
        x1, y1, x2, y2 = comp['bbox']
        
        # 안전한 ROI 추출
        x1_safe = max(0, min(x1, DISPLAY_SIZE-1))
        y1_safe = max(0, min(y1, DISPLAY_SIZE-1))
        x2_safe = max(x1_safe+1, min(x2, DISPLAY_SIZE))
        y2_safe = max(y1_safe+1, min(y2, DISPLAY_SIZE))
        
        roi = warped[y1_safe:y2_safe, x1_safe:x2_safe]
        
        if roi.size > 0:
            # ROI 확대 (3배)
            scale_factor = 3.0
            roi_h, roi_w = roi.shape[:2]
            roi_enlarged = cv2.resize(roi, (int(roi_w * scale_factor), int(roi_h * scale_factor)))
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Click on the image to add pins:**")
                
                # 현재 핀들을 점으로 표시
                pin_objects = []
                for j, (px, py) in enumerate(comp['pins']):
                    # 상대 좌표로 변환
                    rel_x = (px - x1_safe) * scale_factor
                    rel_y = (py - y1_safe) * scale_factor
                    
                    pin_objects.append({
                        "type": "circle",
                        "left": rel_x - 8,
                        "top": rel_y - 8,
                        "width": 16,
                        "height": 16,
                        "fill": "red",
                        "stroke": "darkred",
                        "strokeWidth": 2
                    })
                
                # 핀 편집 캔버스
                canvas_result = st_canvas(
                    background_image=Image.fromarray(cv2.cvtColor(roi_enlarged, cv2.COLOR_BGR2RGB)),
                    width=roi_enlarged.shape[1],
                    height=roi_enlarged.shape[0],
                    drawing_mode="point",
                    initial_drawing={"objects": pin_objects},
                    key=f"pin_edit_{comp_idx}_{len(comp['pins'])}"  # 상태 변경시 캔버스 리셋
                )
                
                st.markdown("""
                **Instructions:**
                - Click to add a pin
                - Drag existing pins to move them
                - Delete pins by dragging them outside the image
                """)
            
            with col2:
                st.write("**Pin Management:**")
                
                # 캔버스에서 핀 정보 추출
                new_pins = []
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    for obj in canvas_result.json_data["objects"]:
                        if obj.get("type") == "circle":
                            # 캔버스 좌표를 원본 좌표로 변환
                            canvas_x = obj["left"] + obj["width"] / 2
                            canvas_y = obj["top"] + obj["height"] / 2
                            
                            # 스케일 조정
                            roi_x = canvas_x / scale_factor
                            roi_y = canvas_y / scale_factor
                            
                            # 절대 좌표
                            abs_x = roi_x + x1_safe
                            abs_y = roi_y + y1_safe
                            
                            # 범위 내 핀만 추가
                            if 0 <= abs_x < DISPLAY_SIZE and 0 <= abs_y < DISPLAY_SIZE:
                                new_pins.append((abs_x, abs_y))
                
                # 핀 개수 상태
                current_count = len(new_pins)
                if current_count == expected:
                    st.success(f"✅ Perfect! {expected} pins")
                elif current_count == 0:
                    st.info("No pins. Click on the image to add pins.")
                else:
                    st.warning(f"Need {expected} pins, have {current_count}")
                
                # 핀 좌표 표시 및 미세 조정
                if new_pins:
                    st.write("**Pin Coordinates:**")
                    adjusted_pins = []
                    for j, (px, py) in enumerate(new_pins):
                        sub_col1, sub_col2 = st.columns(2)
                        with sub_col1:
                            new_x = st.number_input(f"Pin {j+1} X", 
                                                  value=float(px), 
                                                  step=0.5,
                                                  key=f"adjust_x_{comp_idx}_{j}")
                        with sub_col2:
                            new_y = st.number_input(f"Pin {j+1} Y", 
                                                  value=float(py), 
                                                  step=0.5,
                                                  key=f"adjust_y_{comp_idx}_{j}")
                        adjusted_pins.append((new_x, new_y))
                    new_pins = adjusted_pins
                
                # 액션 버튼들
                if st.button("💾 Save Pins", key=f"save_{comp_idx}"):
                    comp['pins'] = new_pins
                    st.success(f"Saved {len(new_pins)} pins!")
                    st.rerun()
                
                if st.button("🔄 Auto Re-detect", key=f"redetect_{comp_idx}"):
                    # 자동 재검출 로직
                    try:
                        cls = comp['class']
                        bbox = (x1_safe, y1_safe, x2_safe, y2_safe)
                        auto_pins = []
                        
                        if cls == 'Resistor':
                            detected = ResistorEndpointDetector().extract(warped, bbox)
                            auto_pins = list(detected) if detected and detected[0] is not None else []
                        elif cls == 'LED':
                            result = LedEndpointDetector().extract(warped, bbox, st.session_state.holes)
                            auto_pins = result.get('endpoints', []) if result else []
                        elif cls == 'Diode':
                            detected = DiodeEndpointDetector().extract(warped, bbox)
                            auto_pins = list(detected) if detected and detected[0] is not None else []
                        
                        if auto_pins:
                            comp['pins'] = auto_pins
                            st.success(f"Auto-detected {len(auto_pins)} pins!")
                            st.rerun()
                        else:
                            st.warning("Auto-detection failed. Please add pins manually.")
                    except Exception as e:
                        st.error(f"Auto-detection error: {e}")
                
                if st.button("🗑️ Clear All", key=f"clear_{comp_idx}"):
                    comp['pins'] = []
                    st.success("Cleared all pins!")
                    st.rerun()
                
                if st.button("🔙 Back to Overview", key=f"back_{comp_idx}"):
                    if 'selected_component' in st.session_state:
                        del st.session_state.selected_component
                    st.rerun()
        else:
            st.error("Invalid component ROI. Please check component boundaries.")
    
    # 최종 상태 확인
    all_complete = all(
        len(comp['pins']) == (8 if comp['class'] == 'IC' else 2)
        for comp in pin_results
    )
    
    if all_complete:
        st.success("🎉 All components have the correct number of pins!")
    else:
        incomplete = [
            f"{i+1}: {comp['class']} ({len(comp['pins'])}/{8 if comp['class'] == 'IC' else 2})"
            for i, comp in enumerate(pin_results)
            if len(comp['pins']) != (8 if comp['class'] == 'IC' else 2)
        ]
        st.warning(f"⚠️ Incomplete: {', '.join(incomplete)}")
    
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
=======
    uploaded = st.file_uploader("Upload breadboard image", type=["jpg","png","jpeg"])
    if uploaded:
        data = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        disp_img, scale = resize_image(img, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
        st.session_state.img = img
        st.session_state.disp_img = disp_img
        st.session_state.scale = scale
        st.success("Image uploaded successfully.")
        if st.button("Next ▶️", key="next_1"):
            st.session_state.page += 1
    else:
        st.info("Please upload an image to proceed.")

# 2) 코너 조정
def page_2_corner_adjust():
    st.subheader("Step 2: Adjust 4 Corner Handles")
    img = st.session_state.img
    disp_img = st.session_state.disp_img
    scale = st.session_state.scale
    h, w = img.shape[:2]
    detector = FasterRCNNDetector(model_path=MODEL_PATH)
    dets = detector.detect(img)
    bb = next((box for cls,_,box in dets if cls.lower()=="breadboard"), None)
    if bb is None:
        st.error("Breadboard not detected.")
        return
    default_pts = [(bb[0],bb[1]),(bb[2],bb[1]),(bb[2],bb[3]),(bb[0],bb[3])]
    scaled_pts = [(int(x*scale), int(y*scale)) for x,y in default_pts]
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
    canvas = st_canvas(
        background_image=Image.fromarray(cv2.cvtColor(disp_img,cv2.COLOR_BGR2RGB)),
        width=disp_img.shape[1], height=disp_img.shape[0],
        drawing_mode="transform", initial_drawing={"objects":handles}, key="corner"
    )
    if canvas.json_data and canvas.json_data.get("objects"):
        src = [[(o["left"]+o["width"]/2)/scale, (o["top"]+o["height"]/2)/scale]
               for o in canvas.json_data["objects"]]
    else:
        src = np.float32(default_pts)
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32([[0,0],[w,0],[w,h],[0,h]]))
    warped = cv2.warpPerspective(img, M, (w,h))
    st.session_state.warped = warped
    st.session_state.warped_raw = warped.copy()
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_2"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_2"): st.session_state.page += 1

# 3) 변환 이미지
def page_3_transformed():
    st.subheader("Step 3: Transformed Image")
    st.image(cv2.cvtColor(st.session_state.warped, cv2.COLOR_BGR2RGB), use_container_width=True)
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_3"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_3"): st.session_state.page += 1

# 4) 컴포넌트 검출 및 편집
def page_4_component_edit():
    st.subheader("Step 4: Component Detection & Manual Edit")
    warped = st.session_state.warped
    disp_warp, scale = resize_image(warped, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
    disp_rgb = cv2.cvtColor(disp_warp, cv2.COLOR_BGR2RGB)
    detector = FasterRCNNDetector(model_path=MODEL_PATH)
    raw = detector.detect(warped)
    comps = [{'class':c,'bbox':b} for c,_,b in raw if c.lower()!='breadboard']
    handles = []
    COLOR_MAP = {'Resistor':'#e63946','LED':'#f4a261','Diode':'#457b9d','IC':'#9d4edd',
                 'Line_area':'#2a9d8f','Capacitor':'#6c757d'}
    for comp in comps:
        x1,y1,x2,y2 = comp['bbox']
        col = COLOR_MAP.get(comp['class'],'#6c757d')
        handles.append({
            "type":"rect","left":x1*scale,"top":y1*scale,
            "width":(x2-x1)*scale,"height":(y2-y1)*scale,
            "stroke":col,"fill":f"{col}33","cornerColor":col,"cornerSize":6
        })
    canvas = st_canvas(background_image=Image.fromarray(disp_rgb),
                       width=disp_warp.shape[1], height=disp_warp.shape[0],
                       drawing_mode="transform", initial_drawing={"objects":handles}, key="comp")
    if canvas.json_data and canvas.json_data.get("objects"):
        updated = []
        for idx, o in enumerate(canvas.json_data["objects"]):
            l,t = o['left']/scale, o['top']/scale
            w_box,h_box = o['width']/scale, o['height']/scale
            updated.append({'class': comps[idx]['class'],
                            'bbox':(int(l),int(t),int(l+w_box),int(t+h_box))})
    else:
        updated = comps
    st.session_state.final_comps = updated
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_4"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_4"): st.session_state.page += 1

# 5) 구멍 검출 및 넷 클러스터링
# --- Page 5: Hole Detection & Net Clustering ---
# --- Page 5: Hole Detection & Net Clustering ---
# --- Page 5: Hole Detection & Net Clustering ---
def page_5_template_holes():
    st.subheader("Step 5: Hole Detection & Net Clustering")
    warped_raw = st.session_state.warped_raw

    # 화면 표시용 축소 이미지 및 스케일 계산
    disp_warp, scale = resize_image(warped_raw, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)

    # HoleDetector 초기화 & 구멍 검출
    hd = HoleDetector(
        template_csv_path=os.path.join(BASE_DIR, "detector", "template_holes_complete.csv"),
        template_image_path=os.path.join(BASE_DIR, "detector", "breadboard18.jpg"),
        max_nn_dist=20.0
    )
    holes = hd.detect_holes(warped_raw)
    st.write(f"Detected holes: {len(holes)} points")

    # 넷 클러스터링 (윈도우 없이)
    nets, row_nets = hd.get_board_nets(holes, base_img=warped_raw, show=False)
    st.write(f"Detected nets: {len(nets)} clusters")

    # hole_to_net 맵 생성 (row_nets 기준)
    hole_to_net = {}
    for row_idx, clusters in row_nets:
        for entry in clusters:
            net_id = entry['net_id']
            for x, y in entry['pts']:
                hole_to_net[(int(round(x)), int(round(y)))] = net_id
    st.session_state.hole_to_net = hole_to_net

    # net_colors 생성
    rng = np.random.default_rng(1234)
    net_ids = sorted(set(hole_to_net.values()))
    net_colors = {nid: tuple(int(c) for c in rng.integers(0, 256, 3)) for nid in net_ids}
    st.session_state.net_colors = net_colors

    # 시각화를 위한 오버레이 (홀별 넷 색상 표시)
    vis = warped_raw.copy()
    for (x, y), net_id in hole_to_net.items():
        color = net_colors[net_id]
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)

    # 축소 후 표시
    disp_vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
    st.image(cv2.cvtColor(disp_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    # 세션 변수 저장
    st.session_state.holes = holes
    st.session_state.row_nets = row_nets  

    # navigation
    cols = st.columns([1, 1, 1])
    if cols[0].button("◀️ Previous", key="prev_5"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_5"): st.session_state.page += 1
>>>>>>> 43c99fd46a94b88ad6ea9a12de5345dc93c72d7d

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

<<<<<<< HEAD
if __name__ == "__main__":
    main()
=======
# --- Page 6: Pin Detection & Net Visualization ---
def page_6_pin_detection():
    st.subheader("Step 6: Pin Detection & Net Visualization")
    warped = st.session_state.warped_raw

    # --- 초기화: pin_results ---
    if 'pin_results' not in st.session_state:
        pin_results = []
        for comp in st.session_state.final_comps:
            cls = comp['class']
            x1, y1, x2, y2 = comp['bbox']
            # 자동 핀 검출
            if cls == 'Resistor':
                raw = ResistorEndpointDetector().extract(warped, comp['bbox'])
                pins = raw or []
            elif cls == 'LED':
                result = LedEndpointDetector().extract(warped, comp['bbox'], st.session_state.holes)
                pins = result.get('endpoints', []) if result else []
            elif cls == 'Diode':
                pins = DiodeEndpointDetector().extract(warped, comp['bbox']) or []
            elif cls == 'IC':
                ics = ICChipPinDetector().detect(warped)
                pins = [(px + x1, py + y1) for px, py in ics[0]['pin_points']] if ics else []
            elif cls == 'Line_area':
                roi = warped[y1:y2, x1:x2]
                segs = WireDetector().detect_wires(roi)
                eps, _ = WireDetector().select_best_endpoints(segs)
                pins = [(x1 + pt[0], y1 + pt[1]) for pt in eps] if eps else []
            else:
                pins = []
            pin_results.append({'class': cls, 'bbox': (x1, y1, x2, y2), 'pins': pins})
        st.session_state.pin_results = pin_results

    # hole_to_net & net_colors 로드
    hole_to_net = st.session_state.hole_to_net
    net_colors = st.session_state.net_colors

    # 1) Visualization Overlay
    disp_vis, disp_scale = resize_image(warped.copy(), MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
    for comp in st.session_state.pin_results:
        for px, py in comp['pins']:
            closest = min(hole_to_net.keys(), key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
            net_id = hole_to_net[closest]
            color = net_colors[net_id]
            cx, cy = int(px * disp_scale), int(py * disp_scale)
            cv2.circle(disp_vis, (cx, cy), 6, color, -1)
            cv2.putText(disp_vis, str(net_id), (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    st.image(cv2.cvtColor(disp_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    # 2) Component 선택 및 핀 수정 UI
    options = [f"{i+1}: {c['class']}" for i, c in enumerate(st.session_state.pin_results)]
    sel = st.selectbox("Select component to adjust pins", list(range(len(st.session_state.pin_results))), format_func=lambda i: options[i])
    comp = st.session_state.pin_results[sel]
    x1, y1, x2, y2 = comp['bbox']
    expected = 8 if comp['class'] == 'IC' else 2
    st.write(f"Adjust pins for **{comp['class']}** (expected **{expected}** points)")
    roi = warped[y1:y2, x1:x2]
    disp_roi, s = resize_image(roi, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
    canvas = st_canvas(
        background_image=Image.fromarray(cv2.cvtColor(disp_roi, cv2.COLOR_BGR2RGB)),
        width=disp_roi.shape[1], height=disp_roi.shape[0], drawing_mode="point", key=f"pin_canvas_{sel}"
    )
    if canvas.json_data and canvas.json_data.get("objects"):
        pts = []
        for o in canvas.json_data["objects"]:
            cx = o['left'] + o['width']/2
            cy = o['top'] + o['height']/2
            pts.append((cx/s + x1, cy/s + y1))
        if len(pts) == expected:
            comp['pins'] = pts
            st.success(f"Updated pins: {pts}")
        else:
            st.warning(f"Click exactly {expected} points. Current: {len(pts)}")

    # navigation
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_6"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_6"): st.session_state.page += 1

# 7) 값 입력
def page_7_value_input():
    st.subheader("Step 7: Component Values Input")
    vals = {}
    for pr in st.session_state.pin_results:
        if pr['class']=='Resistor':
            vals[tuple(pr['bbox'])] = st.number_input(
                f"Resistance (Ω) for bbox {pr['bbox']}", value=10.0, key=str(pr['bbox']))
        else:
            vals[tuple(pr['bbox'])] = 0.0
    st.session_state.comp_values = vals
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_7"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_7"): st.session_state.page += 1

# 8) 핀 수동 수정
def page_8_fix_pins():
    st.subheader("Step 8: Fix Pins (Manual)")
    fixed = []
    for i, pr in enumerate(st.session_state.pin_results):
        coords = []
        for j, p in enumerate(pr['pins']):
            x = st.number_input(f"Pin {j+1} X for {pr['class']}", value=float(p[0]), key=f"x_{i}_{j}")
            y = st.number_input(f"Pin {j+1} Y for {pr['class']}", value=float(p[1]), key=f"y_{i}_{j}")
            coords.append((x,y))
        fixed.append({'class':pr['class'],'bbox':pr['bbox'],'pins':coords})
    st.session_state.fixed_pins = fixed
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_8"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_8"): st.session_state.page += 1

# 9) 전원 선택
def page_9_power_selection():
    st.subheader("Step 9: Power Terminal Selection")
    disp = st.session_state.disp_img
    canvas = st_canvas(
        background_image=Image.fromarray(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)),
        width=disp.shape[1], height=disp.shape[0], drawing_mode="point", key="power_sel"
    )
    pts = [(o['left'], o['top']) for o in canvas.json_data.get('objects',[])] if canvas.json_data else []
    if len(pts) >= 2:
        st.session_state.power_pairs = pts[:2]
    st.write("Power Pairs:", st.session_state.power_pairs if 'power_pairs' in st.session_state else [])
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_9"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_9"): st.session_state.page += 1

# 10) 회로 생성
def page_10_generate_circuit():
    st.subheader("Step 10: Generate Circuit")
    try:
        generate_circuit(
            components=st.session_state.fixed_pins,
            holes=st.session_state.holes,
            nets=st.session_state.nets,
            values=st.session_state.comp_values,
            power_pairs=st.session_state.power_pairs,
            base_img=st.session_state.warped_raw
        )
        img_path = os.path.join(BASE_DIR, "circuit.jpg")
        spice_path = os.path.join(BASE_DIR, "circuit.spice")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.session_state.circuit_img = img
        if os.path.exists(spice_path):
            with open(spice_path, 'rb') as f:
                st.download_button("Download SPICE File", f, file_name="circuit.spice")
            st.session_state.spice_file = spice_path
    except Exception as e:
        st.error(f"Circuit generation error: {e}")
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_10"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_10"): st.session_state.page += 1

# 11) 오류 검사
def page_11_error_check():
    st.subheader("Step 11: Error Checking")
    try:
        ec = ErrorChecker(st.session_state.spice_file)
        errors = ec.run_all_checks()
        df = pd.DataFrame(errors)
        st.table(df)
        st.session_state.errors = errors
    except Exception as e:
        st.error(f"Error checking failed: {e}")
    cols = st.columns([1,1,1])
    if cols[0].button("◀️ Previous", key="prev_11"): st.session_state.page -= 1
    if cols[2].button("Next ▶️", key="next_11"): st.session_state.page += 1

# 12) 최종 결과 요약
def page_12_summary():
    st.subheader("Step 12: Summary")
    st.write("**Components:**", st.session_state.final_comps)
    st.write("**Pins:**", st.session_state.fixed_pins)
    st.write("**Values:**", st.session_state.comp_values)
    st.write("**Power Pairs:**", st.session_state.power_pairs)
    st.write("**Errors:**", st.session_state.errors)
    if st.button("Restart ▶️", key="restart"):  
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state.page = 1

# 메인 페이지 네비게이션
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
    page_5_template_holes()
elif page == 6:
    page_6_pin_detection()
elif page == 7:
    page_7_value_input()
elif page == 8:
    page_8_fix_pins()
elif page == 9:
    page_9_power_selection()
elif page == 10:
    page_10_generate_circuit()
elif page == 11:
    page_11_error_check()
elif page == 12:
    page_12_summary()
else:
    st.error("Invalid step. Restarting...")
>>>>>>> 43c99fd46a94b88ad6ea9a12de5345dc93c72d7d
