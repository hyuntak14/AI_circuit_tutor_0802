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
TOTAL_PAGES = 10

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
        show_navigation(9, next_enabled=False)
        return
    
    # 입력 데이터 검증
    if not st.session_state.fixed_pins:
        st.error("❌ No components detected. Please complete previous steps.")
        show_navigation(9, next_enabled=False)
        return
        
    if not st.session_state.hole_to_net:
        st.error("❌ No hole-to-net mapping available. Please complete hole detection.")
        show_navigation(9, next_enabled=False)
        return
        
    if not st.session_state.power_points or len(st.session_state.power_points) < 2:
        st.error("❌ Please select at least 2 power terminals.")
        show_navigation(9, next_enabled=False)
        return
    
    with st.spinner("⚡ Generating circuit diagram and SPICE file..."):
        try:
            # nearest_net 함수 정의 (오류 처리 포함)
            def find_nearest_net(pt):
                hole_to_net = st.session_state.hole_to_net
                if not hole_to_net:
                    raise ValueError("hole_to_net is empty")
                
                closest = min(hole_to_net.keys(), key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2)
                return hole_to_net[closest]
            
            # 전원 쌍 변환 및 단자 찾기 (main.py의 로직 참조)
            all_endpoints = [pt for comp in st.session_state.fixed_pins for pt in comp['pins']]
            
            if not all_endpoints:
                st.error("❌ No component endpoints found. Please check pin detection.")
                show_navigation(9, next_enabled=False)
                return
            
            power_pairs = []
            voltage = st.session_state.voltage
            
            # 클릭한 위치에서 가장 가까운 실제 엔드포인트 찾기
            for plus_pt, minus_pt in [(st.session_state.power_points[0], st.session_state.power_points[1])]:
                closest_plus = min(all_endpoints, key=lambda p: (p[0]-plus_pt[0])**2 + (p[1]-plus_pt[1])**2)
                closest_minus = min(all_endpoints, key=lambda p: (p[0]-minus_pt[0])**2 + (p[1]-minus_pt[1])**2)
                
                net_plus = find_nearest_net(closest_plus)
                net_minus = find_nearest_net(closest_minus)
                
                # schemdraw용 그리드 좌표 변환 (640x640 기준) - division by zero 방지
                img_w = DISPLAY_SIZE
                comp_count = len([c for c in st.session_state.fixed_pins if c['class'] != 'Line_area'])
                
                # division by zero 방지
                if comp_count == 0:
                    comp_count = 1  # 최소값 설정
                    
                grid_width = comp_count * 2 + 2
                
                # 추가 안전장치
                if grid_width == 0:
                    grid_width = 4  # 기본값 설정
                    
                x_plus_grid = closest_plus[0] / img_w * grid_width
                x_minus_grid = closest_minus[0] / img_w * grid_width
                
                power_pairs.append((net_plus, x_plus_grid, net_minus, x_minus_grid))
            
            # 와이어 연결 처리
            wires = []
            for comp in st.session_state.fixed_pins:
                if comp['class'] == 'Line_area' and len(comp['pins']) == 2:
                    try:
                        net1 = find_nearest_net(comp['pins'][0])
                        net2 = find_nearest_net(comp['pins'][1])
                        if net1 != net2:
                            wires.append((net1, net2))
                    except Exception as e:
                        st.warning(f"⚠️ Wire connection error for {comp['class']}: {e}")
                        continue
            
            # fixed_pins에 value 정보 추가
            comps_with_values = []
            for comp in st.session_state.fixed_pins:
                comp_with_value = comp.copy()
                # bbox를 키로 사용하여 comp_values에서 값 찾기
                bbox_key = tuple(comp['bbox'])
                comp_with_value['value'] = st.session_state.comp_values.get(bbox_key, 0.0)
                comps_with_values.append(comp_with_value)
            
            # 회로 생성 - value가 포함된 컴포넌트 전달
            mapped, hole_to_net_updated = generate_circuit(
                all_comps=comps_with_values,
                holes=st.session_state.holes,
                wires=wires,
                voltage=voltage,
                output_spice=os.path.join(BASE_DIR, "circuit.spice"),
                output_img=os.path.join(BASE_DIR, "circuit.jpg"),
                hole_to_net=st.session_state.hole_to_net,
                power_pairs=power_pairs
            )
            
            st.session_state.circuit_components = mapped
            st.session_state.power_pairs = power_pairs  # power_pairs 저장
            st.success("✅ Circuit generated successfully!")
            
        except ZeroDivisionError as e:
            st.error(f"❌ Division by zero error: {str(e)}")
            st.error("This usually happens when:")
            st.error("- No components are detected")
            st.error("- Grid width calculation results in zero")
            st.error("- Invalid coordinate calculations")
            show_navigation(9, next_enabled=False)
            return
        except ValueError as e:
            st.error(f"❌ Value error: {str(e)}")
            st.error("Please check if all required data is properly initialized.")
            show_navigation(9, next_enabled=False)
            return
        except TypeError as e:
            st.error(f"❌ Type error in generate_circuit call: {str(e)}")
            st.info("This might be due to incorrect parameter names. Please check the generate_circuit function signature.")
            
            # generate_circuit 함수의 시그니처 출력을 위한 디버깅 정보
            try:
                import inspect
                sig = inspect.signature(generate_circuit)
                st.code(f"generate_circuit signature: {sig}")
            except:
                pass
            
            show_navigation(9, next_enabled=False)
            return
        except Exception as e:
            st.error(f"❌ Circuit generation failed: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            show_navigation(9, next_enabled=False)
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
            else:
                st.warning("⚠️ Circuit image generated but cannot be loaded")
        else:
            st.warning("⚠️ Circuit image not found")
    
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
        else:
            st.warning("⚠️ SPICE file not generated")
    
    show_navigation(9, next_enabled=True)

# 11) 오류 검사
def page_11_error_checking():
    st.subheader("Step 11: Circuit Error Checking")
    
    if 'spice_file' not in st.session_state or not os.path.exists(st.session_state.spice_file):
        st.error("❌ No SPICE file available for error checking.")
        show_navigation(10, next_enabled=False)
        return
    
    if 'circuit_components' not in st.session_state:
        st.error("❌ No circuit components available for error checking.")
        show_navigation(10, next_enabled=False)
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
            show_navigation(10, next_enabled=True)  # 오류가 있어도 다음 단계로 진행 가능
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
    
    show_navigation(10, next_enabled=True)

# 12) 최종 요약
def page_12_summary():
    st.subheader("Step 12: Project Summary")
    # 프로젝트 완료 메시지
    st.balloons()
    st.success("🎉 Breadboard to Schematic conversion completed!")

    # 🔍 Similar Circuit Comparison (text only)
    st.markdown("### 🔍 Similar Circuit Comparison")
    import glob, re, os
    import networkx as nx
    from checker.Circuit_comparer import CircuitComparer

    # Reference GraphML files directory
    graphml_dir = os.path.join(BASE_DIR, "checker")
    ref_files = glob.glob(os.path.join(graphml_dir, "circuit*.graphml"))
    if ref_files:
        # Load current circuit graph
        current_file = os.path.join(BASE_DIR, "circuit.graphml")
        try:
            G_curr = nx.read_graphml(current_file)
        except Exception:
            G_curr = None

        best_score = -1.0
        best_file = None
        if G_curr is not None:
            for f in ref_files:
                try:
                    G_ref = nx.read_graphml(f)
                    comparer = CircuitComparer(G_curr, G_ref, debug=False)
                    score = comparer.compute_similarity()
                    if score > best_score:
                        best_score = score
                        best_file = f
                except Exception:
                    continue
        if best_file is not None and best_score >= 0:
            # Extract topic number from filename
            m = re.search(r"circuit(\d+)_", os.path.basename(best_file))
            num = int(m.group(1)) if m else None
            topic_map = {
                1: "병렬회로",
                2: "직렬회로",
                3: "키르히호프",
                4: "키르히호프2법칙",
                5: "중첩의 원리",
                6: "오실로스코프 실습1",
                7: "오실로스코프 실습2",
                8: "반파정류회로",
                9: "반파정류회로2",
                10: "비반전 증폭기"
            }
            topic = topic_map.get(num, "알 수 없는 주제")
            st.write(
                f"**The generated circuit is most similar to the '{topic}' topic**"
                f" (file: {os.path.basename(best_file)}), similarity score: {best_score:.2f}."
            )
        else:
            st.info("ℹ️ No valid circuit comparisons found.")
    else:
        st.info("ℹ️ No reference .graphml files found for comparison.")

    # 재시작 및 이전 버튼
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Start New Project", key="restart", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'page':
                    del st.session_state[key]
            st.session_state.page = 1
            st.rerun()
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