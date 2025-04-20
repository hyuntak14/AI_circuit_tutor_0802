import cv2
import numpy as np
import networkx as nx
from detector.location_detector import HoleDetector

class WireDetector:
    def __init__(self, kernel_size=5):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 80]),
            'red1': ([0, 80, 80], [20, 255, 255]),
            'red2': ([150, 80, 80], [180, 255, 255])
        }
        self.hole_detector = HoleDetector()
        self.white_block = None
        self.white_c = None
        self.full_white_mask = None  # 전체 마스크 저장

    def configure_white_thresholds(self, full_image):
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        self.white_block, self.white_c = self.hole_detector.find_best_threshold_params(
            gray_eq, block_sizes=[11, 15, 19], c_values=[0, 5, 10]
        )
        if self.white_block is None:
            raise RuntimeError("전체 이미지 기준 흰 선 threshold 파라미터를 찾지 못했습니다.")

        hsv = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, self.kernel)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, self.kernel)

        thresh = cv2.adaptiveThreshold(
            gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, self.white_block, self.white_c
        )
        comb = cv2.bitwise_or(hsv_mask, thresh)
        comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, self.kernel)
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, self.kernel)
        self.full_white_mask = comb  # 저장

    def extract_white_wire_mask(self, image, bbox):
        x1, y1, x2, y2 = bbox
        mask_crop = self.full_white_mask[y1:y2, x1:x2].copy()
        # 핀 제거
        roi = image[y1:y2, x1:x2]
        mask_crop = self.remove_holes(mask_crop, roi)
        return mask_crop

    def remove_holes(self, mask, image):
        from detector.pin_detector import PinDetector
        _, pins = PinDetector().detect_pins(image)
        hole_mask = np.zeros_like(mask)
        for x, y in pins:
            cv2.circle(hole_mask, (x, y), 8, 255, -1)
        return cv2.bitwise_and(mask, cv2.bitwise_not(hole_mask))

    def skeletonize(self, img):
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        skel = np.zeros(img_bin.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            open_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img_bin, open_img)
            skel = cv2.bitwise_or(skel, temp)
            img_bin = cv2.erode(img_bin, element)
            if cv2.countNonZero(img_bin) == 0:
                break
        return skel

    def skeleton_to_graph(self, skel):
        G = nx.Graph()
        h, w = skel.shape
        for (y, x) in np.argwhere(skel > 0):
            G.add_node((x, y))
        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for (y, x) in np.argwhere(skel > 0):
            for dy, dx in offsets:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_] > 0:
                    G.add_edge((x, y), (nx_, ny))
        return G

    def find_wire_endpoints_graph(self, skel):
        G = self.skeleton_to_graph(skel)
        return [node for node, deg in G.degree() if deg == 1]

    def detect_wires(self, image):
        if self.full_white_mask is None:
            self.configure_white_thresholds(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        wire_segments = {}
        for color_key, out_key in [('black', 'black'), ('red', 'red')]:
            if color_key == 'black':
                lo, hi = self.color_ranges['black']
                mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            else:
                m1 = cv2.inRange(hsv, *map(np.array, self.color_ranges['red1']))
                m2 = cv2.inRange(hsv, *map(np.array, self.color_ranges['red2']))
                mask = cv2.bitwise_or(m1, m2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            skel = self.skeletonize(mask)
            ep = self.find_wire_endpoints_graph(skel)
            wire_segments[out_key] = {'mask': mask, 'skeleton': skel, 'endpoints': ep}
        return wire_segments

    def select_best_endpoints(self, wire_segments):
        best_channel, max_count = None, 0
        for color, seg in wire_segments.items():
            cnt = cv2.countNonZero(seg['skeleton'])
            if cnt > max_count:
                best_channel, max_count = color, cnt
        if best_channel:
            return wire_segments[best_channel]['endpoints'], best_channel
        return [], None
    





    def process_line_area_wires(self, warped_img, all_components, scale_factor=4):
        line_area_comps = [c for c in all_components if c[0].lower() == 'line_area']
        if not line_area_comps:
            print("Line_area 객체가 없습니다.")
            return

        composite_list = []
        for cls_name, conf, bbox in line_area_comps:
            x1, y1, x2, y2 = bbox
            roi = warped_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            h, w = roi.shape[:2]
            roi_original_scaled = cv2.resize(roi, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LINEAR)
            wire_segments = self.detect_wires(roi)

            channel_images = []
            for color in ['black', 'white', 'red']:
                seg = wire_segments.get(color)
                if seg is None:
                    continue
                mask_color = cv2.cvtColor(seg['mask'], cv2.COLOR_GRAY2BGR)
                skel_color = cv2.cvtColor(seg['skeleton'], cv2.COLOR_GRAY2BGR)
                for pt in seg['endpoints']:
                    cv2.circle(skel_color, pt, 2, (0,0,255), -1)
                mask_scaled = cv2.resize(mask_color, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LINEAR)
                skel_scaled = cv2.resize(skel_color, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack((mask_scaled, skel_scaled))
                channel_images.append(combined)
            if channel_images:
                channels_composite = np.hstack(channel_images)
                w1 = roi_original_scaled.shape[1]
                w2 = channels_composite.shape[1]
                if w1 < w2:
                    pad = w2 - w1
                    roi_original_scaled = cv2.copyMakeBorder(roi_original_scaled, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
                elif w2 < w1:
                    pad = w1 - w2
                    channels_composite = cv2.copyMakeBorder(channels_composite, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
                composite = np.vstack((roi_original_scaled, channels_composite))
            else:
                composite = roi_original_scaled
            composite_list.append(composite)

        if composite_list:
            max_width = max(img.shape[1] for img in composite_list)
            padded_list = []
            for img in composite_list:
                h_img, w_img = img.shape[:2]
                if w_img < max_width:
                    pad = max_width - w_img
                    img = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
                padded_list.append(img)
            final_composite = np.vstack(padded_list)
            window_name = "Line_area 전선 검출 결과"
            cv2.imshow(window_name, final_composite)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
