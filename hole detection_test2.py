import cv2
import numpy as np
import matplotlib.pyplot as plt

###########################################
# ① 구멍 검출 관련 함수들
###########################################
def filter_valid_holes(contours, draw=False, image=None, return_stats=False):
    hole_centers = []
    circularities = []
    aspect_ratios = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 5 or area > 150:
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
            hole_centers, circularities, aspect_ratios = filter_valid_holes(contours, return_stats=True)
            valid = len(hole_centers)
            if valid == 0:
                continue
            mean_circ = np.mean(circularities)
            mean_ar = np.mean(aspect_ratios)
            noise_ratio = 1 - (valid / len(contours)) if contours else 1
            score = (
                0.5 * (1 - abs(mean_circ - 0.785)) +
                0.3 * (1 - abs(mean_ar - 1.0)) -
                0.2 * noise_ratio
            )
            if score > best_score:
                best_score = score
                best_params = (block_size, c_val)
                print(f"✅ New best → B={block_size}, C={c_val}, valid={valid}, score={score:.3f}")
            else:
                print(f"⏺ B={block_size}, C={c_val}, valid={valid}, score={score:.3f}")
    return best_params

def detect_breadboard_holes_no_illumination(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지 파일을 찾지 못했습니다.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    best_block, best_c = find_best_threshold_params(gray, [11, 15, 19], [0, 5, 10, 15])
    if best_block is None:
        raise RuntimeError("적절한 threshold 조합을 찾지 못했습니다.")
    print(f"[자동선택] Best Threshold Params → blockSize={best_block}, C={best_c}")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, best_block, best_c
    )
    detected_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_centers = filter_valid_holes(contours, draw=True, image=detected_image)
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

###########################################
# ② 행 그룹화 및 병합 (get_merged_row_groups)
###########################################
def get_merged_row_groups(holes, y_threshold=15):
    """
    검출된 구멍들을 y 좌표 기준으로 정렬하고, 인접한 구멍의 중앙 y 차이가 y_threshold 미만이면 하나의 행으로 병합.
    위쪽(작은 y)부터 번호를 부여한 리스트를 반환.
    반환 형식: [(row_number, [ (x, y), ... ], median_y), ...]
    """
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

###########################################
# ③ 최적의 y_threshold 값 탐색 (목표: 67행, 각 행 목표 구멍 수: 행별 패턴 적용)
###########################################
def get_target_holes(row_number):
    # 행 번호에 따라 목표 구멍 개수가 다름
    if row_number < 5:
        return 50
    elif 5 <= row_number <= 64:
        remainder = (row_number - 5) % 6
        return 40 if remainder == 5 else 50
    else:
        return 40

def find_optimal_y_threshold(holes, target_rows=67, target_holes=50, y_threshold_range=(5, 30), step=0.5):
    best_score = float('inf')
    best_threshold = None
    best_groups = None

    for th in np.arange(y_threshold_range[0], y_threshold_range[1] + step, step):
        groups = get_merged_row_groups(holes, y_threshold=th)
        num_rows = len(groups)
        total_deviation = 0
        # 각 그룹마다 목표 구멍 수와 실제 구멍 수의 차이를 합산
        for i, (_, group, _) in enumerate(groups):
            total_deviation += abs(len(group) - target_holes)
        avg_deviation = total_deviation / num_rows if num_rows != 0 else float('inf')
        score = abs(num_rows - target_rows) + avg_deviation
        if score < best_score:
            best_score = score
            best_threshold = th
            best_groups = groups
            print(f"y_threshold={th:.2f} -> rows={num_rows}, avg_deviation={avg_deviation:.2f}, score={score:.2f}")
    return best_threshold, best_groups, best_score

###########################################
# ④ 불완전한 행에 대한 보간 함수 (잘 검출된 행의 x축 좌표 기준 보간)
#     - 각 행에 대해 목표 구멍 수는 get_target_holes(row_number)로 결정.
#     - 보간 기준: 불완전한 행(구멍 수가 min_required 이상, 목표 미만)에 대해,
#       현재 행과 목표 값이 같은 '잘 검출된 행' (즉, 실제 구멍 수와 목표 구멍 수의 차이가 허용 오차 이내인 행) 중
#       y축 중앙값 차이가 가장 작은 행을 기준으로 x 좌표 분포를 가져와 보간.
###########################################
def interpolate_incomplete_rows_ref(row_groups, min_required=20, tolerance=5):
    # 잘 검출된 행: 실제 구멍 수와 목표 구멍 수 차이가 tolerance 이내인 행들.
    well_detected_rows = [grp for grp in row_groups if abs(len(grp[1]) - get_target_holes(grp[0])) <= tolerance]
    
    new_groups = []
    for row_number, group, median_y in row_groups:
        target = get_target_holes(row_number)
        if len(group) >= min_required and len(group) < target:
            # 현재 행과 같은 목표를 가진 well detected 행들 후보 추출
            candidate_rows = [grp for grp in well_detected_rows if get_target_holes(grp[0]) == target]
            if not candidate_rows:
                candidate_rows = well_detected_rows
            if candidate_rows:
                # y축 중앙값 차이가 가장 작은 후보 행 선택
                dists = [abs(median_y - comp[2]) for comp in candidate_rows]
                closest_idx = np.argmin(dists)
                ref_group = candidate_rows[closest_idx][1]
                ref_x_positions = sorted([pt[0] for pt in ref_group])
                # 만약 후보 행이 이미 목표 구멍 수를 만족한다면 그대로 사용
                if len(ref_x_positions) == target:
                    ideal_x_positions = ref_x_positions
                else:
                    # 그렇지 않으면 후보 행의 x 좌표 범위로 보간
                    x_min, x_max = ref_x_positions[0], ref_x_positions[-1]
                    ideal_x_positions = np.linspace(x_min, x_max, target)
            else:
                # 후보 행이 없으면 현재 행의 x 범위를 이용
                group_sorted = sorted(group, key=lambda p: p[0])
                x_min, x_max = group_sorted[0][0], group_sorted[-1][0]
                ideal_x_positions = np.linspace(x_min, x_max, target)
            # 보간된 행: 중앙 y 값은 현재 행의 median_y, 보간 플래그 True
            new_group = [(float(x), median_y) for x in ideal_x_positions]
            new_groups.append((row_number, new_group, median_y, True))
        else:
            new_groups.append((row_number, group, median_y, False))
    return new_groups

###########################################
# ⑤ 결과 시각화 함수들 (보간된 행은 점 크기를 키우고 색상 변경)
###########################################
def draw_processed_holes(detected_img, processed_rows):
    proc_image = detected_img.copy()
    for row_number, pts, median_y, is_interp in processed_rows:
        for (x, y) in pts:
            if is_interp:
                cv2.circle(proc_image, (int(x), int(y)), 5, (0, 0, 255), -1)  # 보간: 빨간색, 크기 5
            else:
                cv2.circle(proc_image, (int(x), int(y)), 2, (255, 0, 255), -1)
        if pts:
            x0, y0 = pts[0]
            cv2.putText(proc_image, f"Row {row_number}", (int(x0)-30, int(y0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return proc_image

def plot_hole_scatter(ax, holes):
    if not holes:
        return
    xs, ys = zip(*holes)
    ax.scatter(xs, ys, s=10, c='blue')
    ax.invert_yaxis()
    ax.set_title("Scatter Plot of Detected Holes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

def plot_clustered_rows(ax, holes):
    if not holes:
        return
    ys_arr = np.array([[y] for _, y in holes])
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=1.2, min_samples=3).fit(ys_arr)
    labels = clustering.labels_
    for label in set(labels):
        color = 'gray' if label == -1 else plt.cm.tab20(label % 10)
        xs = [x for (x, y), l in zip(holes, labels) if l == label]
        ys = [y for (x, y), l in zip(holes, labels) if l == label]
        ax.scatter(xs, ys, s=10, c=[color],
                   label=f'Row {label}' if label != -1 else 'Noise')
    ax.invert_yaxis()
    ax.set_title("Clustered Rows of Holes (DBSCAN)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(fontsize=8, loc='upper right')

###########################################
# ⑥ 메인 실행: 구멍 검출, 행 그룹화, 최적 파라미터 탐색, 보간 및 시각화
###########################################
if __name__ == '__main__':
    image_file = "breadboard10.jpg"
    holes, detected_image = detect_breadboard_holes_no_illumination(image_file, debug=True)
    print(f"검출된 구멍 개수: {len(holes)}")
    
    # 최적의 y_threshold 찾기 (목표: 67행)
    target_rows = 67
    target_holes = None  # 각 행은 행 번호에 따라 다르게 결정됨
    best_threshold, best_groups, best_score = find_optimal_y_threshold(
        holes, target_rows=target_rows, target_holes=50,  # 여기서 50은 완전 행의 기준으로 사용
        y_threshold_range=(5, 30), step=0.5
    )
    print(f"최적의 y_threshold: {best_threshold:.2f} (score={best_score:.2f})")
    
    print(f"병합된 행 그룹 수: {len(best_groups)}")
    for row_number, group, median_y in best_groups:
        print(f"Row {row_number}: 구멍 개수={len(group)}, median_y={median_y}")
    
    # 불완전한 행(구멍 20개 이상, 목표 미만)에 대해 보간 적용 (목표는 행 번호에 따라 get_target_holes로 결정)
    interpolated_groups = interpolate_incomplete_rows_ref(best_groups, min_required=20, tolerance=5)
    
    proc_image = draw_processed_holes(detected_image, interpolated_groups)
    cv2.imshow("Processed Rows with Interpolation", cv2.cvtColor(proc_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 추가: 보간된 구멍과 원래 검출된 구멍을 함께 scatter plot으로 시각화
    original_points = []
    interpolated_points = []
    for row_number, pts, median_y, is_interp in interpolated_groups:
        if is_interp:
            interpolated_points.extend(pts)
        else:
            original_points.extend(pts)
    
    plt.figure(figsize=(8,6))
    if original_points:
        xs_orig, ys_orig = zip(*original_points)
        plt.scatter(xs_orig, ys_orig, s=10, c='blue', label='Detected Holes')
    if interpolated_points:
        xs_interp, ys_interp = zip(*interpolated_points)
        plt.scatter(xs_interp, ys_interp, s=20, c='red', label='Interpolated Holes')
    plt.gca().invert_yaxis()
    plt.title("Scatter Plot of Detected and Interpolated Holes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()
