import cv2
import numpy as np
import matplotlib.pyplot as plt
from hole_detection_test2 import detect_breadboard_holes_no_illumination

class HoleTemplateEditor:
    def __init__(self, image_path, save_csv_path='template_holes.csv', threshold=10):
        # 이미지 로드
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.save_csv_path = save_csv_path
        # 초기 구멍 검출
        holes, _ = detect_breadboard_holes_no_illumination(image_path, debug=False)
        self.points = np.array(holes, dtype=np.float32)
        # 인터랙티브 상태 변수
        self.selected_idx = None
        self.dragging = False
        self.threshold = threshold  # 픽셀 단위 선택 반경

        # 시각화 설정
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.imshow(self.image)
        self.scatter = self.ax.scatter(self.points[:,0], self.points[:,1], s=30, c='lime', picker=True)
        self.ax.set_title(
            "Left-click: add/select/drag | Right-click: delete | 's': save | 'q': quit"
        )

        # 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        # 우클릭: 삭제
        if event.button == 'r':
            if self.points.size == 0: return
            dists = np.hypot(self.points[:,0] - x, self.points[:,1] - y)
            idx = np.argmin(dists)
            if dists[idx] < self.threshold:
                self.points = np.delete(self.points, idx, axis=0)
                self.update()
            return
        # 좌클릭: 선택/드래그 또는 추가
        if self.points.size > 0:
            dists = np.hypot(self.points[:,0] - x, self.points[:,1] - y)
            idx = np.argmin(dists)
            if dists[idx] < self.threshold:
                self.selected_idx = idx
                self.dragging = True
                return
        # 새 점 추가
        self.points = np.vstack([self.points, [x, y]]) if self.points.size else np.array([[x,y]])
        self.update()

    def on_motion(self, event):
        if not self.dragging or self.selected_idx is None: return
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        self.points[self.selected_idx] = [x, y]
        self.update()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_idx = None

    def on_key(self, event):
        if event.key == 's':
            # CSV 저장
            np.savetxt(self.save_csv_path, self.points, delimiter=',', fmt='%d')
            print(f"Saved {len(self.points)} points to {self.save_csv_path}")
        elif event.key == 'q':
            plt.close(self.fig)

    def update(self):
        self.scatter.set_offsets(self.points)
        self.fig.canvas.draw_idle()


# === 직접 코드에서 경로 설정 ===
if __name__ == '__main__':
    # TODO: 여기에 이미지 파일 경로와 출력 CSV 경로를 직접 지정하세요
    IMAGE_PATH   = 'breadboard18.jpg'       # 브레드보드 이미지 파일 경로
    OUTPUT_CSV   = 'template_holes.csv'        # 저장할 CSV 파일 경로

    HoleTemplateEditor(IMAGE_PATH, save_csv_path=OUTPUT_CSV)
