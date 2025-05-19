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
# streamlit_app_part2.py의 page_4_component_edit 함수 수정
# streamlit_app_part2.py의 page_4_component_edit 함수 수정

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
    
    # 편집 모드 상태 초기화
    if 'edit_mode_enabled' not in st.session_state:
        st.session_state.edit_mode_enabled = False
    
    # 캔버스 상태 초기화 (중요!)
    if 'canvas_objects' not in st.session_state:
        st.session_state.canvas_objects = []
    
    editable_comps = st.session_state.editable_comps
    
    # 컴포넌트 클래스 옵션
    CLASS_OPTIONS = ['Resistor', 'LED', 'Diode', 'IC', 'Line_area', 'Capacitor']
    
    # 편집 모드 토글
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Detected {len(editable_comps)} components**")
    with col2:
        edit_mode = st.toggle("Edit Mode", key="edit_mode_toggle", value=st.session_state.edit_mode_enabled)
        st.session_state.edit_mode_enabled = edit_mode
    
    # 편집 모드에 따른 처리
    if edit_mode:
        # 편집 모드 - transform으로 위치 수정
        st.write("**🛠️ Edit Mode: Drag to move, Click to edit properties**")
        
        # editable_comps에서 핸들 생성
        handles = []
        for i, comp in enumerate(editable_comps):
            x1, y1, x2, y2 = comp['bbox']
            col = COLOR_MAP.get(comp['class'], '#6c757d')
            handles.append({
                "type": "rect",
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "stroke": col,
                "fill": f"{col}33",
                "cornerColor": col,
                "cornerSize": 6,
                "id": f"comp_{i}"  # 고유 ID 추가
            })
        
        # 캔버스로 위치 편집 및 클릭 감지
        canvas_result = st_canvas(
            background_image=Image.fromarray(disp_rgb),
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            drawing_mode="transform",
            initial_drawing={"objects": handles},
            key="comp_edit_canvas"
        )
        
        # 클릭된 컴포넌트 감지
        clicked_component = None
        if canvas_result.json_data:
            # 마우스 이벤트 또는 선택된 객체 확인
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                # 가장 최근에 수정된 객체를 클릭된 것으로 간주
                # 또는 선택 상태가 변경된 객체 찾기
                for i, obj in enumerate(objects):
                    if i < len(editable_comps) and obj.get("id") == f"comp_{i}":
                        # 위치가 변경된 경우 업데이트
                        l, t = obj['left'], obj['top']
                        w_box, h_box = obj['width'], obj['height']
                        new_bbox = (int(l), int(t), int(l + w_box), int(t + h_box))
                        if new_bbox != editable_comps[i]['bbox']:
                            editable_comps[i]['bbox'] = new_bbox
        
        # 컴포넌트 클릭 감지 (별도 캔버스 사용)
        st.write("**Click on a component to edit its properties:**")
        
        # 클릭 감지용 투명 캔버스
        click_canvas = st_canvas(
            background_image=Image.fromarray(disp_rgb),
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            drawing_mode="point",
            stroke_width=1,
            stroke_color="rgba(255,0,0,0.1)",
            fill_color="rgba(255,0,0,0.1)",
            key="click_detection_canvas"
        )
        
        # 클릭된 위치로 컴포넌트 찾기
        if click_canvas.json_data and click_canvas.json_data.get("objects"):
            last_click = click_canvas.json_data["objects"][-1]
            click_x = last_click["left"] + last_click["width"] / 2
            click_y = last_click["top"] + last_click["height"] / 2
            
            # 클릭한 위치가 어떤 컴포넌트에 속하는지 확인
            for i, comp in enumerate(editable_comps):
                x1, y1, x2, y2 = comp['bbox']
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    clicked_component = i
                    break
        
        # 클릭된 컴포넌트의 편집 UI 표시
        if clicked_component is not None:
            comp = editable_comps[clicked_component]
            
            # 팝업 스타일 UI
            with st.container():
                st.markdown("---")
                st.markdown(f"### 🔧 Editing Component {clicked_component + 1}")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    # 클래스 변경
                    st.write("**Component Type:**")
                    current_class = comp['class']
                    class_idx = CLASS_OPTIONS.index(current_class) if current_class in CLASS_OPTIONS else 0
                    new_class = st.selectbox(
                        "Select Type",
                        CLASS_OPTIONS,
                        index=class_idx,
                        key=f"popup_class_{clicked_component}"
                    )
                    
                    if st.button("💾 Update Class", key=f"update_class_{clicked_component}"):
                        comp['class'] = new_class
                        st.session_state.editable_comps = editable_comps.copy()
                        st.success(f"Updated to {new_class}")
                        st.rerun()
                
                with col2:
                    # 좌표 미세 조정
                    st.write("**Position:**")
                    x1, y1, x2, y2 = comp['bbox']
                    
                    new_x1 = st.number_input("X1", value=x1, step=1, key=f"popup_x1_{clicked_component}")
                    new_y1 = st.number_input("Y1", value=y1, step=1, key=f"popup_y1_{clicked_component}")
                    new_x2 = st.number_input("X2", value=x2, step=1, key=f"popup_x2_{clicked_component}")
                    new_y2 = st.number_input("Y2", value=y2, step=1, key=f"popup_y2_{clicked_component}")
                    
                    if st.button("📐 Update Position", key=f"update_pos_{clicked_component}"):
                        comp['bbox'] = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
                        st.session_state.editable_comps = editable_comps.copy()
                        st.success("Position updated")
                        st.rerun()
                
                with col3:
                    # 삭제 및 기타 작업
                    st.write("**Actions:**")
                    
                    if st.button("🗑️ Delete", key=f"popup_delete_{clicked_component}", type="secondary"):
                        editable_comps.pop(clicked_component)
                        st.session_state.editable_comps = editable_comps.copy()
                        st.success("Component deleted")
                        st.rerun()
                    
                    if st.button("📋 Duplicate", key=f"popup_duplicate_{clicked_component}"):
                        # 컴포넌트 복제 (약간 오프셋)
                        x1, y1, x2, y2 = comp['bbox']
                        new_comp = {
                            'class': comp['class'],
                            'bbox': (x1 + 20, y1 + 20, x2 + 20, y2 + 20)
                        }
                        editable_comps.append(new_comp)
                        st.session_state.editable_comps = editable_comps.copy()
                        st.success("Component duplicated")
                        st.rerun()
                    
                    if st.button("❌ Close", key=f"popup_close_{clicked_component}"):
                        st.rerun()
                
                # 현재 선택된 컴포넌트 하이라이트 표시
                st.markdown(f"""
                <div style='padding: 10px; background-color: {COLOR_MAP.get(comp['class'], '#6c757d')}20; 
                           border-left: 4px solid {COLOR_MAP.get(comp['class'], '#6c757d')}; border-radius: 5px;'>
                    <strong>Selected:</strong> {comp['class']} at ({comp['bbox'][0]}, {comp['bbox'][1]}) - ({comp['bbox'][2]}, {comp['bbox'][3]})
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
        
        # 세션 상태 업데이트 (위치 변경 반영)
        if st.session_state.editable_comps != editable_comps:
            st.session_state.editable_comps = editable_comps.copy()
        
        st.info("💡 Drag boxes to move, click on components to edit properties. Turn off Edit Mode when done.")
    
    else:
        # 보기 모드 - 현재 상태만 표시
        st.write("**👁️ View Mode: Review detected components**")
        
        # 현재 컴포넌트 상태 표시
        vis_img = warped.copy()
        for i, comp in enumerate(editable_comps):
            x1, y1, x2, y2 = comp['bbox']
            col = COLOR_MAP.get(comp['class'], '#6c757d')
            bgr_color = tuple(int(col.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), bgr_color, 2)
            cv2.putText(vis_img, f"{i+1}:{comp['class']}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), 
                caption=f"Current Layout ({len(editable_comps)} components)", 
                use_container_width=False, width=DISPLAY_SIZE)
        
        st.info("💡 Enable Edit Mode to modify component positions.")
    
    # 컴포넌트 리스트 및 개별 편집
    st.subheader("Component List & Properties")
    
    if not editable_comps:
        st.info("No components detected.")
        show_navigation(4, next_enabled=False)
        return
    
    # 컴포넌트 편집을 위한 expander
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
                if new_class != comp['class']:
                    comp['class'] = new_class
                    # 클래스 변경 시 즉시 세션 상태 업데이트
                    st.session_state.editable_comps = editable_comps.copy()
                    st.rerun()
            
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
                
                new_bbox = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
                if new_bbox != comp['bbox']:
                    comp['bbox'] = new_bbox
                    # 좌표 변경 시 즉시 세션 상태 업데이트
                    st.session_state.editable_comps = editable_comps.copy()
            
            with col3:
                # 삭제 버튼
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    components_to_delete.append(i)
    
    # 삭제 처리 (역순으로 삭제하여 인덱스 문제 방지)
    if components_to_delete:
        for idx in sorted(components_to_delete, reverse=True):
            editable_comps.pop(idx)
        st.session_state.editable_comps = editable_comps.copy()
        st.success(f"Deleted {len(components_to_delete)} component(s)")
        st.rerun()
    
    # 전체 작업 버튼들
    st.subheader("Batch Operations")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Reset to Auto-detected", key="reset_comps"):
            st.session_state.editable_comps = st.session_state.detected_comps.copy()
            st.session_state.edit_mode_enabled = False
            st.success("Reset to original detection results")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear All", key="clear_all"):
            st.session_state.editable_comps = []
            st.session_state.edit_mode_enabled = False
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
                st.session_state.edit_mode_enabled = False
                st.success(f"Re-detected {len(new_comps)} components")
                st.rerun()
    
    with col4:
        # 새 컴포넌트 추가 모드
        if st.button("➕ Add Component", key="add_mode"):
            # 새 컴포넌트 추가를 위한 임시 모드
            st.session_state.add_component_mode = True
            st.rerun()
    
    # 새 컴포넌트 추가 모드
    if st.session_state.get('add_component_mode', False):
        st.subheader("➕ Add New Component")
        st.write("**Draw a rectangle to add a new component:**")
        
        # 기존 컴포넌트가 표시된 이미지
        vis_img = warped.copy()
        for comp in editable_comps:
            x1, y1, x2, y2 = comp['bbox']
            col = COLOR_MAP.get(comp['class'], '#6c757d')
            bgr_color = tuple(int(col.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), bgr_color, 2)
        
        # 새 컴포넌트 그리기 캔버스
        canvas_add = st_canvas(
            background_image=Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)),
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            drawing_mode="rect",
            stroke_width=2,
            stroke_color="#ff0000",
            fill_color="rgba(255,0,0,0.1)",
            key="add_component_canvas"
        )
        
        # 새로 그린 박스 처리
        if canvas_add.json_data and canvas_add.json_data.get("objects"):
            for obj in canvas_add.json_data["objects"]:
                x1 = int(obj["left"])
                y1 = int(obj["top"])
                x2 = int(obj["left"] + obj["width"])
                y2 = int(obj["top"] + obj["height"])
                
                # 유효한 크기 확인
                if abs(x2-x1) > 20 and abs(y2-y1) > 20:
                    # 중복 확인
                    is_duplicate = any(
                        abs(comp['bbox'][0] - x1) < 20 and abs(comp['bbox'][1] - y1) < 20
                        for comp in editable_comps
                    )
                    
                    if not is_duplicate:
                        # 새 컴포넌트 클래스 선택
                        new_class = st.selectbox(
                            "Select class for new component:",
                            CLASS_OPTIONS,
                            key="new_comp_class"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Add Component", key="confirm_add"):
                                new_comp = {'class': new_class, 'bbox': (x1, y1, x2, y2)}
                                editable_comps.append(new_comp)
                                st.session_state.editable_comps = editable_comps.copy()
                                st.session_state.add_component_mode = False
                                st.success(f"Added new {new_class} component")
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ Cancel", key="cancel_add"):
                                st.session_state.add_component_mode = False
                                st.rerun()
        
        # 추가 모드 종료 버튼
        if st.button("🔙 Exit Add Mode", key="exit_add"):
            st.session_state.add_component_mode = False
            st.rerun()
    
    # 최종 결과 저장
    st.session_state.final_comps = editable_comps
    
    # 요약 정보
    st.markdown("---")
    st.success(f"✅ {len(editable_comps)} components ready for pin detection.")
    
    # 컴포넌트별 개수 표시
    if editable_comps:
        comp_counts = {}
        for comp in editable_comps:
            comp_type = comp['class']
            comp_counts[comp_type] = comp_counts.get(comp_type, 0) + 1
        
        st.write("**Component Summary:**")
        summary_text = ", ".join([f"{cls}: {count}" for cls, count in comp_counts.items()])
        st.write(summary_text)
    
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


# streamlit_app_part2.py의 간소화된 page_6_pin_detection 함수

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
                cls = comp['class']
                x1, y1, x2, y2 = comp['bbox']
                pins = []
                
                # 바운딩 박스 유효성 검증
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > DISPLAY_SIZE or y2 > DISPLAY_SIZE:
                    pins = []
                else:
                    # 클래스별 핀 검출
                    try:
                        if cls == 'Resistor':
                            detected = ResistorEndpointDetector().extract(warped, (x1, y1, x2, y2))
                            pins = list(detected) if detected and detected[0] is not None else []
                        elif cls == 'LED':
                            result = LedEndpointDetector().extract(warped, (x1, y1, x2, y2), st.session_state.holes)
                            pins = result.get('endpoints', []) if result else []
                        elif cls == 'Diode':
                            detected = DiodeEndpointDetector().extract(warped, (x1, y1, x2, y2))
                            pins = list(detected) if detected and detected[0] is not None else []
                        elif cls == 'IC':
                            roi = warped[y1:y2, x1:x2]
                            if roi.size > 0:
                                ics = ICChipPinDetector().detect(roi)
                                pins = [(x1 + px, y1 + py) for px, py in ics[0]['pin_points']] if ics else []
                        elif cls == 'Line_area':
                            roi = warped[y1:y2, x1:x2]
                            if roi.size > 0:
                                wire_det = WireDetector(kernel_size=4)
                                wire_det.configure_white_thresholds(warped)
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
        
        st.session_state.pin_results = pin_results
    
    pin_results = st.session_state.pin_results
    
    # Union-Find 함수
    parent = {net_id: net_id for net_id in set(st.session_state.hole_to_net.values())}
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    # 전체 이미지에 컴포넌트와 핀 표시
    st.subheader("🔍 Component Overview")
    
    overview_img = warped.copy()
    for i, comp in enumerate(pin_results):
        x1, y1, x2, y2 = comp['bbox']
        expected = 8 if comp['class'] == 'IC' else 2
        detected = len(comp['pins'])
        
        # 상태에 따른 박스 색상
        if detected == expected:
            color = (0, 255, 0)  # 초록
        elif detected > 0:
            color = (0, 165, 255)  # 주황
        else:
            color = (0, 0, 255)  # 빨강
        
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
    
    # 이미지 표시
    st.image(cv2.cvtColor(overview_img, cv2.COLOR_BGR2RGB), 
             caption="Component Overview - Select a component below to edit pins", 
             use_container_width=False, width=DISPLAY_SIZE)
    
    # 상태 요약
    total = len(pin_results)
    completed = sum(1 for comp in pin_results if len(comp['pins']) == (8 if comp['class'] == 'IC' else 2))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Components", total)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("Remaining", total - completed)
    
    # 컴포넌트 선택
    st.subheader("📋 Select Component to Edit")
    
    # 버튼으로 컴포넌트 선택
    selected_comp_idx = None
    cols = st.columns(3)
    
    for i, comp in enumerate(pin_results):
        col_idx = i % 3
        expected = 8 if comp['class'] == 'IC' else 2
        detected = len(comp['pins'])
        
        status = "✅" if detected == expected else ("⚠️" if detected > 0 else "❌")
        button_label = f"{status} {i+1}: {comp['class']}\n({detected}/{expected})"
        
        if cols[col_idx].button(button_label, key=f"select_comp_{i}", use_container_width=True):
            selected_comp_idx = i
    
    # 선택된 컴포넌트 편집
    if selected_comp_idx is not None:
        st.session_state.selected_component = selected_comp_idx
    
    if 'selected_component' in st.session_state:
        comp_idx = st.session_state.selected_component
        comp = pin_results[comp_idx]
        expected = 8 if comp['class'] == 'IC' else 2
        
        st.markdown("---")
        st.subheader(f"✏️ Edit Pins: {comp['class']} #{comp_idx+1}")
        
        # 컴포넌트 ROI 추출
        x1, y1, x2, y2 = comp['bbox']
        x1_safe = max(0, min(x1, DISPLAY_SIZE-1))
        y1_safe = max(0, min(y1, DISPLAY_SIZE-1))
        x2_safe = max(x1_safe+1, min(x2, DISPLAY_SIZE))
        y2_safe = max(y1_safe+1, min(y2, DISPLAY_SIZE))
        
        try:
            roi = warped[y1_safe:y2_safe, x1_safe:x2_safe]
            
            if roi.size > 0:
                # ROI 확대
                scale_factor = 4.0
                roi_h, roi_w = roi.shape[:2]
                roi_enlarged = cv2.resize(roi, (int(roi_w * scale_factor), int(roi_h * scale_factor)))
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Enhanced ROI - Click to add/edit pins:**")
                    
                    # 현재 핀들을 캔버스에 표시
                    pin_objects = []
                    for j, (px, py) in enumerate(comp['pins']):
                        rel_x = (px - x1_safe) * scale_factor
                        rel_y = (py - y1_safe) * scale_factor
                        pin_objects.append({
                            "type": "circle",
                            "left": rel_x - 10,
                            "top": rel_y - 10,
                            "width": 20,
                            "height": 20,
                            "fill": "red",
                            "stroke": "darkred",
                            "strokeWidth": 3
                        })
                    
                    # 확대된 이미지를 background로 사용하는 캔버스
                    canvas_result = st_canvas(
                        background_image=Image.fromarray(cv2.cvtColor(roi_enlarged, cv2.COLOR_BGR2RGB)),
                        width=roi_enlarged.shape[1],
                        height=roi_enlarged.shape[0],
                        drawing_mode="circle",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        fill_color="#FF0000",
                        initial_drawing={"objects": pin_objects},
                        key=f"pin_edit_{comp_idx}"
                    )
                
                with col2:
                    st.write("**Pin Management:**")
                    
                    # 캔버스에서 핀 추출
                    new_pins = []
                    if canvas_result.json_data and canvas_result.json_data.get("objects"):
                        for obj in canvas_result.json_data["objects"]:
                            canvas_x = obj["left"] + obj["width"] / 2
                            canvas_y = obj["top"] + obj["height"] / 2
                            roi_x = canvas_x / scale_factor
                            roi_y = canvas_y / scale_factor
                            abs_x = roi_x + x1_safe
                            abs_y = roi_y + y1_safe
                            
                            if 0 <= abs_x < DISPLAY_SIZE and 0 <= abs_y < DISPLAY_SIZE:
                                new_pins.append((abs_x, abs_y))
                    
                    # 상태 표시
                    current_count = len(new_pins)
                    if current_count == expected:
                        st.success(f"✅ Perfect! {expected} pins")
                    else:
                        st.info(f"Current: {current_count}/{expected} pins")
                    
                    # 액션 버튼들
                    if st.button("💾 Save Pins", key=f"save_{comp_idx}", type="primary"):
                        comp['pins'] = new_pins
                        st.success(f"Saved {len(new_pins)} pins!")
                        st.rerun()
                    
                    if st.button("🔄 Auto Re-detect", key=f"redetect_{comp_idx}"):
                        # 자동 재검출
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
                                st.warning("Auto-detection failed.")
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
                st.error("Invalid component ROI.")
                
        except Exception as e:
            st.error(f"Error processing ROI: {e}")
    
    # 최종 상태 확인
    all_complete = all(
        len(comp['pins']) == (8 if comp['class'] == 'IC' else 2)
        for comp in pin_results
    )
    
    st.markdown("---")
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
