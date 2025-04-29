import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from hole_detector import HoleDetector

class HoleTemplateEditor:
    def __init__(self, image_path, save_csv_path='template_holes.csv', threshold=10):
        # 이미지 로드
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.save_csv_path = save_csv_path
        # 1차 구멍 검출: HoleDetector 사용
        detector = HoleDetector()
        holes = detector.detect_holes(img_bgr)
        self.points = np.array(holes, dtype=np.float32)
        # 인터랙티브 상태 변수
        self.selected_idx = None
        self.dragging = False
        self.threshold = threshold  # 픽셀 단위 선택 반경
        # 영역 삭제용 변수
        self.region_selecting = False
        self.region_start = None
        self.rect_patch = None

        # 시각화 설정
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.imshow(self.image)
        self.scatter = self.ax.scatter(self.points[:,0], self.points[:,1], s=30, c='lime', picker=True)
        self.ax.set_title(
            "Left-click: add/select/drag | Middle-drag: select delete region | 'r': delete nearest | 's': save | 'q': quit"
        )

        # 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        x, y = event.xdata, event.ydata
        # Middle-click: 영역 선택 시작
        if event.button == 2:
            self.region_selecting = True
            self.region_start = (x, y)
            self.rect_patch = Rectangle((x, y), 0, 0, fill=False, color='red', linewidth=1)
            self.ax.add_patch(self.rect_patch)
            return
        # Left-click: 선택/드래그 또는 추가
        if event.button == 1:
            if self.points.size > 0:
                dists = np.hypot(self.points[:,0] - x, self.points[:,1] - y)
                idx = np.argmin(dists)
                if dists[idx] < self.threshold:
                    self.selected_idx = idx
                    self.dragging = True
                    return
            # 새 점 추가
            self.points = np.vstack([self.points, [x, y]]) if self.points.size else np.array([[x, y]])
            self.update()
        # Right-click: nearest 삭제
        elif event.button == 3:
            self._delete_nearest(x, y)

    def on_motion(self, event):
        # 영역 선택 중일 때 사각형 크기 업데이트
        if self.region_selecting and self.rect_patch and event.inaxes == self.ax and event.xdata is not None:
            x0, y0 = self.region_start
            x1, y1 = event.xdata, event.ydata
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            self.rect_patch.set_xy((xmin, ymin))
            self.rect_patch.set_width(xmax - xmin)
            self.rect_patch.set_height(ymax - ymin)
            self.fig.canvas.draw_idle()
            return
        # 점 드래그
        if not self.dragging or self.selected_idx is None: return
        if event.inaxes != self.ax or event.xdata is None: return
        self.points[self.selected_idx] = [event.xdata, event.ydata]
        self.update()

    def on_release(self, event):
        # 영역 선택 종료: 해당 영역 내 점 삭제
        if self.region_selecting and event.button == 2 and event.inaxes == self.ax and event.xdata is not None:
            x0, y0 = self.region_start
            x1, y1 = event.xdata, event.ydata
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            # 영역 내 점 필터링
            mask = ~((self.points[:,0] >= xmin) & (self.points[:,0] <= xmax) &
                     (self.points[:,1] >= ymin) & (self.points[:,1] <= ymax))
            self.points = self.points[mask]
            # 패치 제거
            if self.rect_patch:
                self.rect_patch.remove()
            self.region_selecting = False
            self.region_start = None
            self.rect_patch = None
            self.update()
            return
        # 점 드래그 종료
        if self.dragging and event.button == 1:
            self.dragging = False
            self.selected_idx = None

    def on_key(self, event):
        if event.key == 's':
            np.savetxt(self.save_csv_path, self.points, delimiter=',', fmt='%d')
            print(f"Saved {len(self.points)} points to {self.save_csv_path}")
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'r':
            if event.inaxes == self.ax and event.xdata is not None:
                self._delete_nearest(event.xdata, event.ydata)

    def _delete_nearest(self, x, y):
        if self.points.size == 0: return
        dists = np.hypot(self.points[:,0] - x, self.points[:,1] - y)
        idx = np.argmin(dists)
        if dists[idx] < self.threshold:
            self.points = np.delete(self.points, idx, axis=0)
            self.update()

    def update(self):
        self.scatter.set_offsets(self.points)
        self.fig.canvas.draw_idle()

# === 직접 코드에서 경로 설정 ===
if __name__ == '__main__':
    IMAGE_PATH = 'breadboard18.jpg'    # 브레드보드 이미지
    OUTPUT_CSV = 'template_holes.csv'     # 저장될 CSV
    HoleTemplateEditor(IMAGE_PATH, save_csv_path=OUTPUT_CSV)
