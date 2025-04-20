import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ======================================================
# HoleDetector 클래스: 브레드보드 구멍(홀) 검출 및 격자화 관련 기능
# ======================================================
class HoleDetector:
    def __init__(self):
        pass

    @staticmethod
    def filter_valid_holes(contours, draw=False, image=None, return_stats=False):
        hole_centers = []
        circularities = []
        aspect_ratios = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 5 or area > 150:  # 면적 임계값 (이미지 해상도에 맞게 조정)
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.1:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            hole_centers.append((cx, cy))
            circularities.append(circularity)
            aspect_ratios.append(aspect_ratio)
            if draw and image is not None:
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
                cv2.drawContours(image, [approx], -1, (255, 0, 0), 2)
        if return_stats:
            return hole_centers, circularities, aspect_ratios
        else:
            return hole_centers

    @staticmethod
    def find_best_threshold_params(gray_img, block_sizes, c_values):
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        best_score = -np.inf
        best_params = (None, None)
        for block_size in block_sizes:
            for c_val in c_values:
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, block_size, c_val
                )
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hole_centers, circularities, aspect_ratios = HoleDetector.filter_valid_holes(contours, return_stats=True)
                valid = len(hole_centers)
                if valid == 0:
                    continue
                mean_circ = np.mean(circularities)
                mean_ar = np.mean(aspect_ratios)
                noise_ratio = 1 - (valid / len(contours)) if len(contours) > 0 else 1
                score = (
                    0.5 * (1 - abs(mean_circ - 0.785)) +
                    0.3 * (1 - abs(mean_ar - 1.0)) -
                    0.2 * noise_ratio
                )
                if score > best_score:
                    best_score = score
                    best_params = (block_size, c_val)
                    print(f"New best: blockSize={block_size}, C={c_val}, valid={valid}, score={score:.3f}")
                else:
                    print(f"blockSize={block_size}, C={c_val}, valid={valid}, score={score:.3f}")
        return best_params

    def detect_holes(self, image, debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        best_block, best_c = self.find_best_threshold_params(gray, [11, 15, 19], [0, 5, 10, 15])
        if best_block is None:
            raise RuntimeError("No valid threshold parameters found.")
        print(f"[Auto Selected] Best threshold: blockSize={best_block}, C={best_c}")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, best_block, best_c
        )
        detected_image = image.copy()
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hole_centers = self.filter_valid_holes(contours, draw=True, image=detected_image)
        if debug:
            block_sizes = [11, 15, 19]
            c_values = [0, 5, 10, 15]
            plt.figure(figsize=(len(c_values) * 3, len(block_sizes) * 3))
            for row_idx, block_size in enumerate(block_sizes):
                for col_idx, c_val in enumerate(c_values):
                    temp_thresh = cv2.adaptiveThreshold(
                        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY_INV, block_size, c_val
                    )
                    idx = row_idx * len(c_values) + col_idx + 1
                    plt.subplot(len(block_sizes), len(c_values), idx)
                    plt.imshow(temp_thresh, cmap='gray')
                    plt.title(f"B={block_size}, C={c_val}")
                    plt.axis('off')
            plt.suptitle("Threshold Results by blockSize and C")
            plt.tight_layout()
            plt.show()
        return hole_centers, detected_image

    @staticmethod
    def cluster_1d_points(coords, threshold=10):
        coords = np.sort(coords)
        clusters = []
        current_cluster = [coords[0]]
        for i in range(1, len(coords)):
            if coords[i] - coords[i - 1] < threshold:
                current_cluster.append(coords[i])
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [coords[i]]
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        return np.array(clusters)

    @staticmethod
    def find_cluster_index(value, clusters):
        return int(np.argmin(np.abs(clusters - value)))

    def organize_holes_into_grid(self, holes):
        holes = np.array(holes)
        x_coords = holes[:, 0]
        y_coords = holes[:, 1]
        x_clusters = self.cluster_1d_points(x_coords)
        y_clusters = self.cluster_1d_points(y_coords)
        grid = {}
        for hole in holes:
            col_idx = self.find_cluster_index(hole[0], x_clusters)
            row_idx = self.find_cluster_index(hole[1], y_clusters)
            grid[(row_idx, col_idx)] = hole
        return grid

    def get_merged_row_groups(self, holes, y_threshold=15):
        sorted_holes = sorted(holes, key=lambda p: p[1])
        groups = []
        for pt in sorted_holes:
            if not groups:
                groups.append([pt])
            else:
                current_group = groups[-1]
                median_y = np.median([p[1] for p in current_group])
                if abs(pt[1] - median_y) < y_threshold:
                    current_group.append(pt)
                else:
                    groups.append([pt])
        row_groups = []
        for group in groups:
            sorted_group = sorted(group, key=lambda p: p[0])
            median_y = np.median([p[1] for p in sorted_group])
            row_groups.append((sorted_group, median_y))
        row_groups.sort(key=lambda x: x[1])
        numbered_groups = [(i, group, median_y) for i, (group, median_y) in enumerate(row_groups)]
        return numbered_groups

    def interpolate_incomplete_rows(self, row_groups, min_required=20, tolerance=10):
        # 목표 구멍 수를 행 번호에 따라 설정하는 예시 함수
        def get_target_holes(row_number):
            if row_number < 5:
                return 50
            elif 5 <= row_number <= 64:
                remainder = (row_number - 5) % 6
                return 40 if remainder == 5 else 50
            else:
                return 40

        well_detected_rows = [grp for grp in row_groups if abs(len(grp[1]) - get_target_holes(grp[0])) <= tolerance]
        new_groups = []
        for row_number, group, median_y in row_groups:
            target = get_target_holes(row_number)
            if len(group) >= min_required and len(group) < target:
                candidate_rows = [grp for grp in well_detected_rows if get_target_holes(grp[0]) == target]
                if not candidate_rows:
                    candidate_rows = well_detected_rows
                if candidate_rows:
                    dists = [abs(median_y - comp[2]) for comp in candidate_rows]
                    closest_idx = np.argmin(dists)
                    ref_group = candidate_rows[closest_idx][1]
                    ref_x_positions = sorted([pt[0] for pt in ref_group])
                    if len(ref_x_positions) == target:
                        ideal_x_positions = ref_x_positions
                    else:
                        x_min, x_max = ref_x_positions[0], ref_x_positions[-1]
                        ideal_x_positions = np.linspace(x_min, x_max, target)
                else:
                    sorted_group = sorted(group, key=lambda p: p[0])
                    x_min, x_max = sorted_group[0][0], sorted_group[-1][0]
                    ideal_x_positions = np.linspace(x_min, x_max, target)
                new_group = [(float(x), median_y) for x in ideal_x_positions]
                new_groups.append((row_number, new_group, median_y, True))
            else:
                new_groups.append((row_number, group, median_y, False))
        return new_groups

    def draw_processed_holes(self, detected_img, processed_rows):
        proc_image = detected_img.copy()
        for row_number, pts, median_y, is_interp in processed_rows:
            for (x, y) in pts:
                if is_interp:
                    cv2.circle(proc_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(proc_image, (int(x), int(y)), 2, (255, 0, 255), -1)
            if pts:
                x0, y0 = pts[0]
                cv2.putText(proc_image, f"Row {row_number}", (int(x0) - 30, int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return proc_image

# ======================================================
# ComponentPinDetector 클래스: 전기 소자(및 전선) 핀 위치 검출
# ======================================================
class ComponentPinDetector:
    def __init__(self):
        pass

    def detect_component_pins(self, image, component_type, component_roi):
        """
        주어진 소자 타입과 ROI(바운딩 박스) 내에서 소자의 핀 위치를 검출합니다.
        반환값은 전체 이미지 좌표로 변환된 핀 위치 리스트입니다.
        """
        roi = image[component_roi[1]:component_roi[3], component_roi[0]:component_roi[2]]
        if component_type == 'Resistor':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
            pin_positions = self.extract_resistor_leg_endpoints(lines, roi.shape)
        elif component_type == 'LED':
            pin_positions = self.extract_led_pins(roi)
        elif component_type == 'Capacitor':
            pin_positions = self.extract_capacitor_pins(roi)
        elif component_type == 'Diode':
            pin_positions = self.extract_diode_pins(roi)
        elif component_type == 'IC':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            pin_positions = self.extract_ic_pins(contours, roi.shape)
        else:
            pin_positions = []
        # ROI 기준 좌표를 전체 이미지 좌표로 변환
        global_pin_positions = [(x + component_roi[0], y + component_roi[1]) for (x, y) in pin_positions]
        return global_pin_positions

    def extract_resistor_leg_endpoints(self, lines, roi_shape):
        if lines is None:
            return []
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append((x1, y1))
            points.append((x2, y2))
        if not points:
            return []
        points = np.array(points)
        left_idx = np.argmin(points[:, 0])
        right_idx = np.argmax(points[:, 0])
        left_point = tuple(points[left_idx])
        right_point = tuple(points[right_idx])
        return [left_point, right_point]

    def extract_led_pins(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        if not centroids:
            return []
        centroids = np.array(centroids)
        left_idx = np.argmin(centroids[:, 0])
        right_idx = np.argmax(centroids[:, 0])
        return [tuple(centroids[left_idx]), tuple(centroids[right_idx])]

    def extract_capacitor_pins(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        left_pin = (x, y + h // 2)
        right_pin = (x + w, y + h // 2)
        return [left_pin, right_pin]

    def extract_diode_pins(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        if not centroids:
            return []
        centroids = np.array(centroids)
        left_idx = np.argmin(centroids[:, 0])
        right_idx = np.argmax(centroids[:, 0])
        return [tuple(centroids[left_idx]), tuple(centroids[right_idx])]

    def extract_ic_pins(self, contours, roi_shape):
        pins = []
        h, w = roi_shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 50:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if cx < w * 0.3 or cx > w * 0.7:
                        pins.append((cx, cy))
        if not pins:
            pins = [(0, h // 2), (w, h // 2)]
        return pins

# ======================================================
# (테스트용) 메인 함수: 이미지 파일에서 구멍 검출과 소자 핀 위치 검출 예시
# ======================================================
if __name__ == '__main__':
    image_path = "breadboard7.jpg"  # 실제 이미지 경로로 수정하세요.
    image = cv2.imread(image_path)
    if image is None:
        print("이미지 로드 실패")
    else:
        # 구멍 검출 테스트
        hole_detector = HoleDetector()
        holes, detected_img = hole_detector.detect_holes(image, debug=True)
        grid = hole_detector.organize_holes_into_grid(holes)
        merged_groups = hole_detector.get_merged_row_groups(holes, y_threshold=15)
        interpolated_groups = hole_detector.interpolate_incomplete_rows(merged_groups, min_required=20, tolerance=5)
        proc_img = hole_detector.draw_processed_holes(detected_img, interpolated_groups)
        cv2.imshow("Hole Detection", proc_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 소자 핀 검출 테스트 (예시로 수동 라벨링된 bounding box 사용)
        comp_pin_detector = ComponentPinDetector()
        # 예시 bounding box 및 소자 타입 (실제 수동 라벨링 결과 사용)
        component_roi = (100, 100, 200, 200)
        comp_type = "Resistor"
        pins = comp_pin_detector.detect_component_pins(image, comp_type, component_roi)
        print(f"Detected pins for {comp_type}: {pins}")
