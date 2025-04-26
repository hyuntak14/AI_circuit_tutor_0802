import cv2
import numpy as np
import matplotlib.pyplot as plt
from new_led_detector import ImprovedLedEndpointDetector
from hole_detector import HoleDetector  # 기존 코드에서 임포트한다고 가정


def test_led_detector_cropped(img_path):
    """이미 잘려진 LED 이미지에 대한 테스트 함수"""
    print(f"테스트 이미지: {img_path}")
    
    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {img_path}")
        return
    
    # 이미지 전체가 이미 LED 영역이므로 bbox는 이미지 전체가 됨
    h, w = img.shape[:2]
    led_bbox = [0, 0, w, h]
    
    # 이미지 복사본
    img_display = img.copy()
    img_result = img.copy()
    
    # bbox 시각화
    cv2.rectangle(img_display, 
                  (led_bbox[0], led_bbox[1]), 
                  (led_bbox[2], led_bbox[3]), 
                  (0, 255, 0), 2)
    
    # 브레드보드 구멍 생성 (테스트용)
    try:
        hole_detector = HoleDetector()
        holes = generate_holes_around_led(w, h)
        print(f"생성된 테스트용 구멍 수: {len(holes)}")
    except Exception as e:
        print(f"구멍 생성 중 오류 발생: {e}")
        holes = generate_holes_around_led(w, h)
    
    # 구멍 시각화
    for hx, hy in holes:
        cv2.circle(img_display, (hx, hy), 3, (255, 0, 0), -1)
    
    # 개선된 LED 핀 검출기 초기화
    led_detector = ImprovedLedEndpointDetector(
        wire_thickness_range=(1, 4),
        leg_min_length=8,         # 작은 이미지에 맞게 조정
        orientation_weight=0.6,
        visualize=False
    )
    
    # LED 핀 검출 실행
    try:
        result = led_detector.extract(img, led_bbox, holes)
        if result:
            print("LED 핀 검출 성공!")
            print(f"검출된 엔드포인트: {result['endpoints']}")
            print(f"매핑된 구멍 인덱스: {result['holes']}")
            
            # 결과 시각화
            led_detector.draw(img_result, result, holes)
        else:
            print("LED 핀 검출 실패")
    except Exception as e:
        print(f"LED 핀 검출 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # 디버그 이미지 수집
    if hasattr(led_detector, 'get_debug_images'):
        debug_images = led_detector.get_debug_images()
    else:
        attrs = ['color_lines_mask', 'inpainted_image', 'gray', 'enhanced', 'binary', 'filtered', 'skeleton']
        debug_images = {attr: getattr(led_detector, attr) for attr in attrs if getattr(led_detector, attr, None) is not None}
    
    # 결과 및 디버그 이미지 시각화
    plt.figure(figsize=(15, 10))
    
    # 원본
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('원본 LED 이미지')
    plt.axis('off')
    
    # 구멍 위치
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title('가상의 구멍 위치')
    plt.axis('off')
    
    # 검출 결과
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title('LED 핀 검출 결과')
    plt.axis('off')
    
    # 디버그 이미지
    plot_idx = 4
    for name, debug_img in debug_images.items():
        if plot_idx > 6:
            break
        plt.subplot(2, 3, plot_idx)
        if debug_img.ndim == 3:
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(debug_img, cmap='gray')
        plt.title(name)
        plt.axis('off')
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('led_detection_results.png')
    plt.show()
    
    cv2.destroyAllWindows()
    return result, holes, debug_images


def generate_holes_around_led(width, height, num_rows=3, num_cols=5):
    """LED 주변에 가상의 브레드보드 구멍 생성"""
    holes = []
    bottom_margin = height // 4
    left_margin = width // 6
    right_margin = width // 6
    step_w = (width - left_margin - right_margin) // (num_cols - 1)
    step_h = bottom_margin // (num_rows - 1)
    # 상단
    for c in range(num_cols):
        x = left_margin + c * step_w
        y = height // 8
        holes.append((x, y))
    # 하단
    for r in range(num_rows):
        for c in range(num_cols):
            x = left_margin + c * step_w
            y = height - bottom_margin + r * step_h
            holes.append((x, y))
    return holes

if __name__ == "__main__":
    test_led_detector_cropped('led1.jpg')
