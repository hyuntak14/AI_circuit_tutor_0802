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
        drawing_mode="circle", key="power_sel"
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
