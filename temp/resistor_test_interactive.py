import os
import cv2
from Improved_resistor_det import ImprovedResistorEndpointDetector

# --- 전역 변수 설정 ---
image = None
detector = ImprovedResistorEndpointDetector()
control_window = "Controls"
result_window = "Result"

def on_trackbar_change(_):
    """ 트랙바 값이 변경될 때마다 호출되는 콜백 함수 """
    if image is None:
        return

    # 트랙바에서 현재 값 읽기
    p = detector.params
    p['fixed_threshold'] = cv2.getTrackbarPos("Threshold", control_window)
    p['min_resistor_area'] = cv2.getTrackbarPos("Min Area", control_window)
    p['max_resistor_area'] = cv2.getTrackbarPos("Max Area", control_window)
    p['min_aspect_ratio'] = cv2.getTrackbarPos("Min Aspect Ratio", control_window) / 10.0
    
    p['use_opening'] = cv2.getTrackbarPos("Use Opening", control_window) == 1
    p['use_closing'] = cv2.getTrackbarPos("Use Closing", control_window) == 1
    
    method_idx = cv2.getTrackbarPos("Method", control_window)
    methods = ['contour', 'pca', 'neighbor']
    p['endpoint_method'] = methods[method_idx]
    
    # 디텍터 파라미터 업데이트
    detector.params = p
    
    # 엔드포인트 다시 추출
    h, w = image.shape[:2]
    endpoints = detector.extract(image.copy(), (0, 0, w, h))
    
    # 결과 시각화
    vis_image = image.copy()
    if endpoints:
        detector.draw(vis_image, endpoints)
    else:
        cv2.putText(vis_image, "Detection Failed", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.imshow(result_window, vis_image)


def setup_controls():
    """ 파라미터 조정을 위한 트랙바 생성 """
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 400, 350)
    
    p = detector.params
    cv2.createTrackbar("Threshold", control_window, p['fixed_threshold'], 255, on_trackbar_change)
    cv2.createTrackbar("Min Area", control_window, p['min_resistor_area'], 5000, on_trackbar_change)
    cv2.createTrackbar("Max Area", control_window, p['max_resistor_area'], 20000, on_trackbar_change)
    cv2.createTrackbar("Min Aspect Ratio", control_window, int(p['min_aspect_ratio'] * 10), 100, on_trackbar_change)

    # 체크박스처럼 사용할 트랙바 (0 또는 1)
    cv2.createTrackbar("Use Opening", control_window, int(p['use_opening']), 1, on_trackbar_change)
    cv2.createTrackbar("Use Closing", control_window, int(p['use_closing']), 1, on_trackbar_change)

    # 감지 방법 선택 트랙바 (0: contour, 1: pca, 2: neighbor)
    cv2.createTrackbar("Method", control_window, 0, 2, on_trackbar_change)


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    current_dir = os.getcwd()
    files = [f for f in os.listdir(current_dir) if "resistor" in f.lower() and f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not files:
        print("현재 폴더에 'resistor'가 포함된 이미지 파일이 없습니다.")
    else:
        setup_controls() # 제어 창 한 번만 생성
        
        for file_name in files:
            print(f"\n이미지 로드: {file_name}")
            print("파라미터를 조정한 후, 아무 키나 눌러 다음 이미지로 넘어가세요. 'q'를 누르면 종료됩니다.")
            
            image = cv2.imread(file_name)
            if image is None:
                print(f"이미지 로드 실패: {file_name}")
                continue
            
            # 초기 결과 표시
            on_trackbar_change(0)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()