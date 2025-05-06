import cv2
import numpy as np
from hole_detector import HoleDetector
from sklearn.neighbors import NearestNeighbors

# === 사용자 설정 ===
IMAGE_PATH          = 'breadboard4.jpg'             # 입력 이미지
TEMPLATE_CSV        = "template_holes_complete.csv"  # 템플릿 CSV
TEMPLATE_IMAGE_PATH = "breadboard18.jpg"            # 템플릿 원본 이미지 경로 (스케일링)
MAX_NN_DIST         = 20.0                           # 매칭 허용 픽셀 거리
DISPLAY_SCALE       = 0.5                            # 화면 출력 크기 비율

# 1) 템플릿 좌표 불러오기
template = np.loadtxt(TEMPLATE_CSV, delimiter=",", dtype=np.float32)
idx_temp = np.lexsort((template[:,0], template[:,1]))
template_sorted = template[idx_temp]

# 2) 이미지 로드 및 HoleDetector
def load_and_detect_holes(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    centers = HoleDetector().detect_holes(img)
    arr = np.array(centers, dtype=np.float32)
    idx = np.lexsort((arr[:,0], arr[:,1]))
    return arr[idx], img

holes_arr, img = load_and_detect_holes(IMAGE_PATH)

# 3) 템플릿 좌표 스케일링
tmpl_img = cv2.imread(TEMPLATE_IMAGE_PATH)
if tmpl_img is not None:
    t_h, t_w = tmpl_img.shape[:2]
    i_h, i_w = img.shape[:2]
    scale_x = i_w / t_w
    scale_y = i_h / t_h
else:
    print(f"Warning: cannot load template image '{TEMPLATE_IMAGE_PATH}'. Using 1:1 scale.")
    scale_x = scale_y = 1.0
scaled_pts = np.column_stack((template_sorted[:,0] * scale_x,
                              template_sorted[:,1] * scale_y))

# 4) 초기 매칭: Nearest Neighbor
nbrs = NearestNeighbors(n_neighbors=1).fit(holes_arr)
dists, inds = nbrs.kneighbors(scaled_pts)
mask = dists.flatten() < MAX_NN_DIST
src_pts = scaled_pts[mask].reshape(-1,1,2)
dst_pts = holes_arr[inds.flatten()[mask]].reshape(-1,1,2)

# 5) Affine transform estimation with RANSAC
if len(src_pts) < 3:
    raise RuntimeError("Too few matched points for affine estimation")
affine, inliers = cv2.estimateAffinePartial2D(
    src_pts, dst_pts,
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0
)
# Affine 변환 적용
fitted_pts = cv2.transform(scaled_pts.reshape(-1,1,2), affine).reshape(-1,2)

# 6) 결과 시각화: 검출된 구멍(빨강), 보정된 템플릿(초록)
out = img.copy()
# 빨강: 검출된 구멍 위치
for x, y in holes_arr:
    cv2.drawMarker(out, (int(x), int(y)), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
# 초록: 보정된 템플릿 위치
for x, y in fitted_pts:
    cv2.circle(out, (int(x), int(y)), 2, (0,255,0), -1)

cv2.putText(out, 'Red: detected holes | Green: fitted template', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# 7) 작은 크기로 출력
disp = cv2.resize(out, (0,0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
cv2.imshow("Aligned Template", disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
