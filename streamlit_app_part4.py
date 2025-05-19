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