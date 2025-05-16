import cv2
import numpy as np
import math

class ICChipPinDetector:
    """
    IC 칩(검은색 직사각형) 검출 및 핀(8개) 위치 추출 클래스.
    Pin-1 클릭 방식으로 핀 번호 매핑을 지원합니다.
    """
    def __init__(self,
                 gray_thresh: int = 60,
                 min_area: int = 1000,
                 visualize: bool = False):
        # 그레이스케일 임계값 및 최소 면적 설정
        self.gray_thresh = gray_thresh
        self.min_area = min_area
        self.visualize = visualize

    def detect(self, image: np.ndarray):
        """
        이미지에서 IC 칩을 검출하고, Pin-1 클릭을 통해 핀 순서를 매핑하여 반환합니다.
        반환: [{'box': np.ndarray(shape=(4,2)), 'pin_points': List[(x,y)]}, ...]
        """
        detections = []
        # 1) 그레이스케일 변환 및 임계처리
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, self.gray_thresh, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # 박스 근사 및 ROI 추출
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(int)
            x, y, w, h = cv2.boundingRect(box)
            roi = image[y:y+h, x:x+w]

            # 기존 핀 포인트 추출 로직 (사용자기존 구현 부분)
            pin_points = self._extract_pin_points_from_roi(roi)
            if not pin_points or len(pin_points) != 8:
                continue

            # Pin-1 클릭 UI 표시
            disp = roi.copy()
            for px, py in pin_points:
                cv2.circle(disp, (px, py), 4, (0, 255, 0), -1)
            click_pt = []
            win = 'Click Pin-1'
            def on_click(event, cx, cy, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    click_pt.append((cx, cy))
                    cv2.destroyWindow(win)
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, on_click)
            cv2.imshow(win, disp)
            cv2.waitKey(0)

            # 클릭 위치에 따라 핀 순서 매핑
            if click_pt:
                # 클릭 좌표 전역 변환
                cx, cy = click_pt[0]
                click_global = (x + cx, y + cy)
                # 전역 핀 좌표
                pins_global = [(x + px, y + py) for px, py in pin_points]

                # 칩 중심 계산
                Cx = x + w / 2
                Cy = y + h / 2
                # 각도 계산
                angles = []
                for gx, gy in pins_global:
                    dx = gx - Cx
                    dy = Cy - gy
                    ang = math.degrees(math.atan2(dy, dx)) % 360
                    angles.append(ang)

                # 클릭한 핀 인덱스 찾기
                dists = [((gx - click_global[0])**2 + (gy - click_global[1])**2, idx)
                         for idx, (gx, gy) in enumerate(pins_global)]
                _, start_idx = min(dists, key=lambda t: t[0])
                start_ang = angles[start_idx]

                # 오름차순 정렬하여 순서 매핑
                delta = [(((ang - start_ang) % 360), pins_global[i]) for i, ang in enumerate(angles)]
                ordered_global = [pt for _, pt in sorted(delta, key=lambda t: t[0])]
                # ROI 상대좌표로 변환
                pin_points = [(int(gx - x), int(gy - y)) for gx, gy in ordered_global]

            # 결과 저장
            detections.append({'box': box, 'pin_points': pin_points})

            # 시각화 옵션
            if self.visualize:
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                for (px, py) in pin_points:
                    cv2.circle(image, (x + px, y + py), 4, (0, 0, 255), -1)

        return detections

    def _extract_pin_points_from_roi(self, roi: np.ndarray):
        """
        ROI에서 8개 핀 위치를 검출하는 내부 메소드.
        Canny edge + contour 중심 검출 방식.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pin_points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 100:
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                pin_points.append((cx, cy))

        # 개수가 정확히 8개일 때만 반환
        if len(pin_points) != 8:
            return []

        return pin_points


    def draw(self, image: np.ndarray, detections):
        """
        검출 결과 시각화
        """
        for det in detections:
            box = det['box']
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            for pt in det['pin_points']:
                cv2.circle(image, pt, 6, (0, 255, 0), -1)
