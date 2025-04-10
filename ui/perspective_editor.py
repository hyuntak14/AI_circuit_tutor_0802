import cv2
import numpy as np
from itertools import combinations
points = []
selected_point = -1
clone = None
scale = 1.0
display_size = (800, 800)  # 최대 표시 크기

def get_scaled_size(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    return int(w * scale), int(h * scale), scale

def click_event(event, x, y, flags, param):
    global points, selected_point, scale

    real_x = int(x / scale)
    real_y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(points):
            if abs(p[0] - real_x) < 15 and abs(p[1] - real_y) < 15:
                selected_point = i
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if selected_point != -1:
            points[selected_point] = (real_x, real_y)
            redraw_image()

    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = -1

def redraw_image():
    global clone, points, scale

    temp = clone.copy()
    for i, p in enumerate(points):
        cv2.circle(temp, p, 150, (255, 0, 0), -1)
        cv2.putText(temp, str(i), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    resized_w, resized_h, scale = get_scaled_size(temp, *display_size)
    display = cv2.resize(temp, (resized_w, resized_h))
    cv2.imshow("Select Points", display)

import cv2
import numpy as np
from itertools import combinations
import cv2
import numpy as np

def find_board_corners_auto(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]

    # 기준 꼭짓점 (YOLO 박스 기준)
    ref_box = np.array([[0, 0], [x2 - x1, 0], [x2 - x1, y2 - y1], [0, y2 - y1]])

    # 1. Grayscale → CLAHE → Blur → Edge
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # 2. Contour 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_quad = None

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > max_area:
            # 중심점 비교 → YOLO 박스에서 크게 벗어난 사각형 제거
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if abs(cx - (x2 - x1) / 2) < (x2 - x1) * 0.3 and abs(cy - (y2 - y1) / 2) < (y2 - y1) * 0.3:
                    best_quad = approx
                    max_area = area

    # 3. 복원 (원본 좌표 기준으로 이동)
    if best_quad is not None:
        best_quad = best_quad.reshape(4, 2)
        best_quad[:, 0] += x1
        best_quad[:, 1] += y1
        return best_quad.tolist()

    # 못 찾으면 YOLO 박스 그대로 사용
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def select_and_transform(image, bbox):
    global points, clone, selected_point, scale
    selected_point = -1

    x1, y1, x2, y2 = bbox

    # ✅ YOLO 박스를 기준으로 단순 네 꼭짓점 설정
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    clone = image.copy()
    redraw_image()
    cv2.setMouseCallback("Select Points", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 27:  # Enter or Esc
            break

    cv2.destroyAllWindows()

    if len(points) == 4:
        pts_src = np.float32(points)
        pts_dst = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])  # 정사각형으로 보정
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, matrix, (640, 640))
        return warped, (x1, y1)

    return image, (x1, y1)
