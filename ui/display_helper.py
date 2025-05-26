# display_helper.py - 공통 디스플레이 유틸리티
import cv2
import numpy as np

DISPLAY_SIZE = (1200, 1200)

def resize_for_display(image, target_size=DISPLAY_SIZE):
    """이미지를 지정된 크기로 리사이즈 (비율 유지)"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 비율 유지하면서 리사이즈
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 중앙에 배치하기 위해 패딩 추가
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    padded = cv2.copyMakeBorder(resized, pad_h, target_h - new_h - pad_h, 
                               pad_w, target_w - new_w - pad_w, 
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded, scale, (pad_w, pad_h)

def show_image(window_name, image, wait_key=True):
    """이미지를 1200x1200 크기로 표시"""
    display_img, _, _ = resize_for_display(image)
    cv2.imshow(window_name, display_img)
    
    if wait_key:
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        return key
    return None

def coord_display_to_original(display_x, display_y, original_image):
    """디스플레이 좌표를 원본 좌표로 변환"""
    _, scale, (pad_w, pad_h) = resize_for_display(original_image)
    real_x = int((display_x - pad_w) / scale)
    real_y = int((display_y - pad_h) / scale)
    return max(0, min(real_x, original_image.shape[1]-1)), max(0, min(real_y, original_image.shape[0]-1))

def coord_original_to_display(orig_x, orig_y, original_image):
    """원본 좌표를 디스플레이 좌표로 변환"""
    _, scale, (pad_w, pad_h) = resize_for_display(original_image)
    disp_x = int(orig_x * scale + pad_w)
    disp_y = int(orig_y * scale + pad_h)
    return disp_x, disp_y