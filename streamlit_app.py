import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from detector.fasterrcnn_detector import FasterRCNNDetector

# 절대경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fasterrcnn.pt")

# 캔버스 최대 크기
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 600

def main():
    st.title("📸 Breadboard to Schematic")

    uploaded = st.file_uploader("Upload breadboard image", type=["jpg","png","jpeg"])
    if not uploaded:
        st.info("이미지를 업로드해 주세요.")
        return

    # 원본 로드
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 축소된 디스플레이용 크기 계산
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    disp_img = cv2.resize(img, (disp_w, disp_h))

    st.subheader("1. Original Image (for corner adjust)")
    st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), use_container_width=False)

    # Faster R-CNN으로 breadboard bbox 검출
    detector = FasterRCNNDetector(model_path=MODEL_PATH)
    dets = detector.detect(img)
    bb = next((box for cls,_,box in dets if cls.lower()=="breadboard"), None)
    if bb is None:
        st.error("빵판을 검출하지 못했습니다.")
        return
    x1,y1,x2,y2 = bb

    # 원본 좌표 기준 4개 꼭짓점
    default_pts = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    # 디스플레이용 스케일링
    scaled_pts = [(int(x*scale), int(y*scale)) for x,y in default_pts]

    # “핸들” 크기
    # “핸들” 크기
    HANDLE_SIZE = 16

    # initial_drawing: 4개의 작은 사각형(handle) 객체만 생성
    handles = []
    for (cx, cy) in scaled_pts:
        handles.append({
            "type": "rect",
            "left": cx - HANDLE_SIZE//2,
            "top": cy - HANDLE_SIZE//2,
            "width": HANDLE_SIZE,
            "height": HANDLE_SIZE,
            "stroke": "red",                # ← strokeColor 대신 stroke
            "strokeWidth": 2,
            "fill": "rgba(255,0,0,0.3)",    # ← fillColor 대신 fill
            "cornerColor": "red",
            "cornerSize": 6,
            "transparentCorners": False,
        })

    st.subheader("2. Adjust 4 Corner Handles")
    st.write("파란 네모를 드래그하여 각 꼭짓점을 맞춰주세요.")

    canvas_res = st_canvas(
        background_image=Image.fromarray(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)),
        width=disp_w, height=disp_h,
        drawing_mode="transform",    # 기존 객체(사각형)만 움직일 수 있음
        initial_drawing={"objects": handles},
        key="corner_adjust"
    )

    # 이동된 handle들의 중심을 원본 좌표로 복원
    if canvas_res.json_data and canvas_res.json_data.get("objects"):
        src = []
        for obj in canvas_res.json_data["objects"]:
            left = obj["left"]
            top  = obj["top"]
            w_obj = obj["width"]
            h_obj = obj["height"]
            # 사각형 중앙이 실제 꼭짓점
            cx_disp = left + w_obj/2
            cy_disp = top  + h_obj/2
            # 원본 이미지 좌표로 역변환
            src.append([cx_disp/scale, cy_disp/scale])
        src = np.float32(src)
    else:
        # 조정 없으면 기본값 그대로
        src = np.float32(default_pts)

    # Perspective Warp
    dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    warped_raw = warped.copy()
    st.subheader("3. Transformed Image")
    st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)


    
    # 4) Component Detection & Manual Edit
    st.subheader("4. Component Detection & Manual Edit")

    # ➤ 클래스 리스트 & 색상 맵
    CLASS_LIST    = ['Resistor','LED','Diode','IC','Line_area','Capacitor']
    CLASS_OPTIONS = ['Unassigned'] + CLASS_LIST
    COLOR_MAP = {
        'Resistor':  '#e63946',
        'LED':       '#f4a261',
        'Diode':     '#457b9d',
        'IC':        '#9d4edd',
        'Line_area': '#2a9d8f',
        'Capacitor': '#6c757d'
    }
    DEFAULT_COLOR = '#6c757d'

    # ➤ 모드 선택
    mode = st.radio("🛠️ Mode", ["Adjust","Add New"], horizontal=True)
    drawing_mode = "transform" if mode=="Adjust" else "rect"
    st.write(f"현재 모드: **{mode}**")

    # ➤ 디스플레이용 warped 이미지
    disp_warp = cv2.resize(warped, (disp_w, disp_h))
    disp_rgb  = cv2.cvtColor(disp_warp, cv2.COLOR_BGR2RGB)

    # ➤ 최초 검출된 컴포넌트
    detector = FasterRCNNDetector(model_path=MODEL_PATH)
    raw = detector.detect(warped)
    comps = [{'class':c,'bbox':b} for c,_,b in raw if c.lower()!='breadboard']

    # ➤ initial drawing 생성
    handles = []
    for comp in comps:
        cls = comp['class']
        col = COLOR_MAP.get(cls, DEFAULT_COLOR)
        x1,y1,x2,y2 = comp['bbox']
        handles.append({
            "type": "rect",
            "left":   x1*scale,
            "top":    y1*scale,
            "width":  (x2-x1)*scale,
            "height": (y2-y1)*scale,
            "stroke": col,
            "strokeWidth": 2,
            "fill":   f"{col}33",
            "cornerColor": col,
            "cornerSize": 6,
            "transparentCorners": False,
        })

    # ➤ 캔버스 옵션 구성
    canvas_args = dict(
        background_image=Image.fromarray(disp_rgb),
        width=disp_w, height=disp_h,
        drawing_mode=drawing_mode,
        initial_drawing={"objects": handles},
        key="comp_edit"
    )
    if mode=="Add New":
        # 새로 그리는 박스의 기본 스타일
        canvas_args.update({
            "stroke_width": 2,
            "stroke_color": DEFAULT_COLOR,
            "fill_color":   "rgba(0,0,0,0)"
        })

    canvas_comp = st_canvas(**canvas_args)

    # ➤ updated 리스트 생성
    updated = []
    if canvas_comp.json_data and canvas_comp.json_data.get("objects"):
        for idx, obj in enumerate(canvas_comp.json_data["objects"]):
            left, top = obj["left"]/scale, obj["top"]/scale
            w_box, h_box = obj["width"]/scale, obj["height"]/scale
            x1n, y1n = int(left),       int(top)
            x2n, y2n = int(left+w_box), int(top+h_box)
            # 기존 인식 객체면 원래 클래스, 아니면 None
            orig_cls = comps[idx]['class'] if idx < len(comps) else None
            updated.append({'bbox': (x1n,y1n,x2n,y2n), 'class': orig_cls})
    else:
        updated = [{'bbox':c['bbox'], 'class':c['class']} for c in comps]

    # ➤ 삭제/클래스 변경 UI
    st.subheader("5. Review & Edit Components")
    final_comps = []
    cols = st.columns([1,3,3])
    for i, comp in enumerate(updated):
        do_del = cols[0].checkbox("Delete", key=f"del_{i}")
        cls_sel = cols[1].selectbox(
            "Class",
            CLASS_OPTIONS,
            index=CLASS_OPTIONS.index(comp['class']) if comp['class'] in CLASS_OPTIONS else 0,
            key=f"class_{i}"
        )
        coords = f"{comp['bbox'][0]}, {comp['bbox'][1]} - {comp['bbox'][2]}, {comp['bbox'][3]}"
        cols[2].write(coords)
        if not do_del:
            final_comps.append({
                'bbox': comp['bbox'],
                'class': cls_sel if cls_sel!='Unassigned' else None
            })

    # ➤ 최종 박스 시각화
    viz = disp_warp.copy()
    for comp in final_comps:
        x1,y1,x2,y2 = [int(c*scale) for c in comp['bbox']]
        col = COLOR_MAP.get(comp['class'], DEFAULT_COLOR)
        bgr = tuple(int(col.lstrip('#')[i:i+2], 16) for i in (4,2,0))
        cv2.rectangle(viz, (x1,y1), (x2,y2), bgr, 2)
    st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption="Final Boxes", use_container_width=False)

    # → final_comps를 다음 단계에 넘겨주세요.

        # 3) Perspective Warp 까지 끝난 뒤 바로 추가
        # --- after `warped_raw = warped.copy()` ---

    st.subheader("4. Template Holes Adjustment & Fallback")

    from detector.hole_detector import HoleDetector
    hd = HoleDetector(
        template_csv_path=os.path.join(BASE_DIR, "detector", "template_holes_complete.csv"),
        template_image_path=os.path.join(BASE_DIR, "detector", "breadboard18.jpg"),
        max_nn_dist=20.0
    )

    # (1) load raw template pts
    tpl_raw = np.loadtxt(hd.template_csv_path, delimiter=',')  # shape (N,2)

    # (2) scale template → warped_raw coords
    tpl_img = cv2.imread(hd.template_image_path)
    th, tw = tpl_img.shape[:2]
    H, W = warped_raw.shape[:2]
    scale_model_x, scale_model_y = W/tw, H/th
    tpl_scaled = [(x*scale_model_x, y*scale_model_y) for x,y in tpl_raw]

    # (3) display scaling: warped_raw → disp_warp
    #    이미 disp_warp = resize(warped_raw, (disp_w, disp_h)) 로 만들어 두셨습니다.
    disp_tpl = [(x*scale, y*scale) for x,y in tpl_scaled]

    # (4) build handles for tpl_scaled
    HANDLE = 8
    tpl_handles = []
    for cx, cy in disp_tpl:
        tpl_handles.append({
            "type": "rect",
            "left": cx - HANDLE/2,
            "top":  cy - HANDLE/2,
            "width":  HANDLE,
            "height": HANDLE,
            "stroke": "#ffba08",
            "strokeWidth": 2,
            "fill":   "rgba(255,186,8,0.3)",
            "cornerColor": "#ffba08",
            "cornerSize": 6,
            "transparentCorners": False,
        })

    st.write("템플릿 기반 구멍 포인트를 직접 조정할 수 있습니다.")
    canvas_tpl = st_canvas(
        background_image=Image.fromarray(disp_rgb),
        width=disp_w, height=disp_h,
        drawing_mode="transform",
        initial_drawing={"objects": tpl_handles},
        key="tpl_adjust"
    )

    # (5) recover user-adjusted template pts → warp coords
    if canvas_tpl.json_data and canvas_tpl.json_data.get("objects"):
        user_tpl = []
        for obj in canvas_tpl.json_data["objects"]:
            cx = obj["left"] + obj["width"]/2
            cy = obj["top"]  + obj["height"]/2
            user_tpl.append( (cx/scale, cy/scale) )
    else:
        user_tpl = tpl_scaled  # no adjustment

    # --- now do hole detection with fallback ---

    st.subheader("5. Hole Detection & Net Clustering")

    # try main.py 방식
    try:
        holes = hd.detect_holes(warped_raw)
        st.success(f"✅ detect_holes succeeded: {len(holes)} points")
    except ValueError as e:
        st.warning(f"detect_holes failed ({e})\n→ 템플릿 포인트 사용({len(user_tpl)}개)")
        holes = user_tpl

    # cluster nets
    try:
        nets, row_nets = hd.get_board_nets(holes, base_img=warped_raw, show=False)
        st.success(f"✅ Nets: {len(nets)} clusters")
    except ValueError as e:
        st.warning(f"Net clustering failed ({e}) → 빈 리스트")
        nets, row_nets = [], []

    # visualize hole clusters
    vis = warped_raw.copy()
    rng = np.random.default_rng(0)
    for _, clusters in row_nets:
        for entry in clusters:
            col = tuple(int(c) for c in rng.integers(0,256,3))
            for x,y in entry['pts']:
                cv2.circle(vis, (int(x),int(y)), 3, col, -1)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
             caption="Hole Clusters", use_container_width=True)

    # prepare hole→net map
    hole_to_net = {
        (int(round(x)), int(round(y))): entry['net_id']
        for _, clusters in row_nets
        for entry in clusters
        for x,y in entry['pts']
    }


if __name__ == "__main__":
    main()
