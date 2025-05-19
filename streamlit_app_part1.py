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
