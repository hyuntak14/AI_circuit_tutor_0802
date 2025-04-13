import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN




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


###########################################
# ① 핀홀(구멍) 검출 (조명 보정 포함)
###########################################
def detect_breadboard_holes_no_illumination(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 조명 균일화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 최적 파라미터 탐색
    best_block, best_c = find_best_threshold_params(gray, [11, 15, 19], [0, 5, 10, 15])
    if best_block is None:
        raise RuntimeError("적절한 threshold 조합을 찾지 못했습니다.")
    print(f"[자동선택] Best Threshold Params → blockSize={best_block}, C={best_c}")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, best_block, best_c
    )

    # 구멍 검출
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
            hole_centers, circularities, aspect_ratios = filter_valid_holes(
                contours, return_stats=True
            )

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



###########################################
# ② 행 그룹화 및 순차적 라벨링
###########################################

from sklearn.cluster import DBSCAN
import numpy as np

def find_eps_for_target_rows(holes, target_rows=68, min_samples=30, eps_range=(1, 20), step=0.1):
    ys = np.array([[y] for _, y in holes])

    for eps in np.arange(eps_range[0], eps_range[1], step):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(ys)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == target_rows:
            print(f"✅ Found eps={eps:.2f} → {n_clusters} rows")
            return eps
    print("❌ Couldn't find suitable eps for target rows")
    return None

def find_best_eps_for_target_rows(holes, target_rows=68, min_samples=10, eps_range=(1, 20), step=0.1):
    ys = np.array([[y] for _, y in holes])
    best_eps = None
    min_diff = float('inf')
    
    for eps in np.arange(eps_range[0], eps_range[1], step):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(ys)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        diff = abs(n_clusters - target_rows)
        if diff < min_diff:
            min_diff = diff
            best_eps = eps
            # 만약 완벽히 일치한다면 바로 반환할 수도 있습니다.
            if diff == 0:
                break
                
    if best_eps is None:
        print("❌ eps 값을 찾지 못했습니다.")
    else:
        print(f"✅ Best eps found: {best_eps:.2f} (n_clusters ≈ {target_rows - min_diff} ~ {target_rows + min_diff})")
    return best_eps


# 2. eps로 클러스터링 수행
def group_holes_by_row(holes, eps, min_samples=4):
    if not holes:
        return {}
    ys = np.array([[y] for _, y in holes])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(ys)
    row_groups = {}
    for label, (x, y) in zip(clustering.labels_, holes):
        if label == -1:
            continue
        row_groups.setdefault(label, []).append((x, y))
    return row_groups

def get_sorted_row_groups_with_fixed_count(holes, target_rows=67, min_samples=4):
    eps = find_best_eps_for_target_rows(holes, target_rows=67, min_samples=4, eps_range=(80, 90), step=0.1)
    if eps is None:
        raise ValueError("적절한 eps를 찾지 못했습니다.")
    groups = group_holes_by_row(holes, eps, min_samples)
    
    sorted_groups = []
    for label, pts in groups.items():
        median_y = np.median([p[1] for p in pts])
        sorted_pts = sorted(pts, key=lambda p: p[0])
        sorted_groups.append((label, sorted_pts, median_y))
    sorted_groups = sorted(sorted_groups, key=lambda t: t[2])
    renumbered_groups = [(i, pts, median_y) for i, (_, pts, median_y) in enumerate(sorted_groups)]
    return renumbered_groups


def get_sorted_row_groups(holes):
    groups = group_holes_by_row(holes)
    sorted_groups = []
    for label, pts in groups.items():
        median_y = np.median([p[1] for p in pts])
        sorted_pts = sorted(pts, key=lambda p: p[0])
        sorted_groups.append((label, sorted_pts, median_y))
    sorted_groups = sorted(sorted_groups, key=lambda t: t[2])
    renumbered_groups = [(i, pts, median_y) for i, (_, pts, median_y) in enumerate(sorted_groups)]
    return renumbered_groups

###########################################
# ⑤ 후처리 결과를 원본 이미지에 오버레이하는 함수
###########################################
def draw_processed_holes(detected_img, processed_rows):
    proc_image = detected_img.copy()
    for label, x_coords, median_y in processed_rows:
        for x in x_coords:
            cv2.circle(proc_image, (int(x), int(median_y)), 2, (255, 0, 255), -1)
    return proc_image

###########################################
# Noise 포함/제외 시각화 함수들
###########################################
def plot_clustered_rows(ax, holes):
    if not holes:
        return
    ys_arr = np.array([[y] for _, y in holes])
    clustering = DBSCAN(eps=1.2, min_samples=3).fit(ys_arr)
    labels = clustering.labels_
    for label in set(labels):
        color = 'gray' if label == -1 else plt.cm.tab20(label % 10)
        xs = [x for (x, y), l in zip(holes, labels) if l == label]
        ys = [y for (x, y), l in zip(holes, labels) if l == label]
        ax.scatter(xs, ys, s=10, c=[color], label=f'Row {label}' if label != -1 else 'Noise')
    ax.invert_yaxis()
    ax.set_title("Clustered Rows of Holes (DBSCAN)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(fontsize=8, loc='upper right')

def plot_clustered_rows_without_noise(ax, holes):
    if not holes:
        return
    ys_arr = np.array([[y] for _, y in holes])
    clustering = DBSCAN(eps=1, min_samples=10).fit(ys_arr)
    labels = clustering.labels_
    for label in sorted(set(labels)):
        if label == -1:
            continue
        color = plt.cm.tab20(label % 10)
        xs = [x for (x, y), l in zip(holes, labels) if l == label]
        ys = [y for (x, y), l in zip(holes, labels) if l == label]
        ax.scatter(xs, ys, s=10, c=[color], label=f'Row {label}')
    ax.invert_yaxis()
    ax.set_title("Clustered Rows (Noise Removed)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(fontsize=8, loc='upper right')




###########################################
# 실행 메인
###########################################
if __name__ == '__main__':
    image_file = "breadboard7.jpg"
    holes, detected_image = detect_breadboard_holes_no_illumination(image_file, debug=True)
    print(f"검출된 구멍 개수: {len(holes)}")
    # 2. eps 자동 탐색 (68개의 row가 나오게)
    eps = find_best_eps_for_target_rows(holes, target_rows=68, min_samples=4, eps_range=(1, 20), step=0.1)
    # 이 줄 수정!
    sorted_rows = get_sorted_row_groups_with_fixed_count(holes, target_rows=68, min_samples=4)

    print(f"검출된 행 수: {len(sorted_rows)}")

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

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(detected_image)
    axes[0].set_title("Detected Holes")
    axes[0].axis("off")
    plot_hole_scatter(axes[1], holes)
    plot_clustered_rows(axes[2], holes)
    plot_clustered_rows_without_noise(axes[3], holes)
    plt.tight_layout()
    plt.show()
