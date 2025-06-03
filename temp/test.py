
import cv2
import numpy as np
import os
from skimage import color, feature, measure

def standard_hough(img, edges):
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    out = img.copy()
    if lines is not None:
        for rho, theta in lines[:,0]:
            a = np.cos(theta); b = np.sin(theta)
            x0 = a*rho; y0 = b*rho
            x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*(a))
            cv2.line(out, (x1,y1), (x2,y2), (0,0,255), 2)
    return out

def probabilistic_hough(img, edges):
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=30, maxLineGap=10)
    out = img.copy()
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:,0]:
            cv2.line(out, (x1,y1), (x2,y2), (0,255,0), 2)
    return out

def lsd_detector(img_gray, img_color):
    lsd = cv2.createLineSegmentDetector(0)
    # Unpack detect return values: lines, width, prec, nfa
    lines, _, _, _ = lsd.detect(img_gray)
    out = img_color.copy()
    if lines is not None:
        out = lsd.drawSegments(out, lines)
    return out

def fast_line_detector(img_gray, img_color):
    try:
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(img_gray)
        out = img_color.copy()
        if lines is not None:
            out = fld.drawSegments(out, lines)
        return out
    except AttributeError:
        return img_color.copy()

def ransac_lines(img_gray, img_color):
    # Use Canny to get edge coords
    edges = feature.canny(img_gray, sigma=2)
    coords = np.column_stack(np.nonzero(edges))
    out = img_color.copy()
    if len(coords) < 2:
        return out
    # Detect multiple lines by iterative RANSAC
    remaining = coords.copy()
    for _ in range(3):  # try detecting up to 3 lines
        model_robust, inliers = measure.ransac(remaining, measure.LineModelND, min_samples=2,
                                               residual_threshold=1, max_trials=1000)
        if inliers is None or np.sum(inliers) < 2:
            break
        p0, p1 = model_robust.params
        # Convert line model (point p0 and direction p1) to endpoints
        h, w = img_gray.shape
        # Compute two far points along line within image bounds
        if abs(p1[0]) > 1e-6:
            t0 = (0 - p0[0]) / p1[0]
            y0 = p0[1] + t0 * p1[1]
            t1 = (w - p0[0]) / p1[0]
            y1 = p0[1] + t1 * p1[1]
            cv2.line(out, (0, int(np.clip(y0, 0, h))), (w, int(np.clip(y1, 0, h))), (255,0,0), 2)
        else:
            # vertical line
            y0 = 0
            y1 = h
            x0 = p0[0]
            x1 = p0[0]
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
        # Remove inliers and repeat
        remaining = remaining[~inliers]
        if len(remaining) < 2:
            break
    return out

def add_label(img, text):
    labeled = img.copy()
    cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return labeled

def process_image(file):
    img = cv2.imread(file)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 800 / max(h, w) if max(h, w) > 800 else 1
    img_color = cv2.resize(img, (int(w*scale), int(h*scale)))
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)

    # Apply each method
    std_h = add_label(standard_hough(img_color, edges), "Standard Hough")
    prob_h = add_label(probabilistic_hough(img_color, edges), "Probabilistic Hough")
    lsd = add_label(lsd_detector(img_gray, img_color), "LSD")
    fld = add_label(fast_line_detector(img_gray, img_color), "Fast Line Detector")
    # RANSAC may fail, label accordingly
    try:
        rans = add_label(ransac_lines(img_gray, img_color), "RANSAC")
    except:
        rans = add_label(img_color, "RANSAC unavailable")

    # Stack results: Two rows, three columns
    top_row = np.hstack((std_h, prob_h, lsd))
    bottom_row = np.hstack((fld, rans, img_color))
    # If widths mismatch, resize bottom_row
    h1, w1 = top_row.shape[:2]
    h2, w2 = bottom_row.shape[:2]
    if w2 != w1:
        bottom_row = cv2.resize(bottom_row, (w1, int(h2 * (w1 / w2))))
    combined = np.vstack((top_row, bottom_row))
    return combined

def main():
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [f for f in os.listdir('.') if 'led' in f.lower() and f.lower().endswith(image_extensions)]
    if not files:
        print("No 'led' images found.")
        return
    for file in files:
        result = process_image(file)
        if result is None:
            continue
        window_name = f"Line Detection - {file}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, result)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

if __name__ == "__main__":
    main()
