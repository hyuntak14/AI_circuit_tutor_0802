import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def detect_led_lead_endpoints(image_path):
    """
    LED 리드의 끝점 좌표를 검출하는 함수
    
    Args:
        image_path (str): 이미지 파일 경로
    
    Returns:
        list: [(x1, y1), (x2, y2)] 형태의 끝점 좌표 리스트
    """
    
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 원본 이미지 복사 (결과 시각화용)
    result_image = image.copy()
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거 및 전처리
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. 어두운 구멍 검출 (브레드보드 구멍)
    # 임계값을 사용해 어두운 영역 찾기
    _, thresh_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 모폴로지 연산으로 구멍 모양 개선
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    holes_mask = cv2.morphologyEx(thresh_dark, cv2.MORPH_OPEN, kernel_circle)
    holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, kernel_circle)
    
    # 4. 구멍 검출 (원형 검출)
    circles = cv2.HoughCircles(
        holes_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=3,
        maxRadius=15
    )
    
    hole_centers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        hole_centers = [(x, y) for x, y in circles[:, :2]]
    
    # 5. 엣지 검출로 리드 찾기
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 6. 허프 라인 변환으로 직선 검출
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=20,
        maxLineGap=10
    )
    
    # 7. 리드 후보 직선 필터링
    lead_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 직선의 길이 계산
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # 직선의 각도 계산 (수직에 가까운 것 선택)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # 필터링 조건: 길이가 충분하고, 수직에 가까운 직선
            if length > 15 and (angle > 70 and angle < 110):
                lead_lines.append(((x1, y1), (x2, y2)))
    
    # 8. 대안 방법: 밝기 기반 리드 검출
    if len(lead_lines) < 2:
        # 금속 리드는 반사로 인해 밝을 수 있음
        _, thresh_bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 세로 방향 구조 요소로 리드 모양 강화
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        vertical_mask = cv2.morphologyEx(thresh_bright, cv2.MORPH_OPEN, kernel_vertical)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 윤곽선을 직선으로 근사
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 직선에 가까운 윤곽선만 선택
            if len(approx) >= 2:
                area = cv2.contourArea(contour)
                rect = cv2.boundingRect(contour)
                aspect_ratio = rect[3] / rect[2] if rect[2] > 0 else 0
                
                # 세로로 긴 모양이고 적당한 크기인 것 선택
                if aspect_ratio > 2 and area > 50 and area < 1000:
                    # 윤곽선의 시작점과 끝점 찾기
                    top_point = tuple(contour[contour[:, :, 1].argmin()][0])
                    bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
                    lead_lines.append((top_point, bottom_point))
    
    # 9. 리드와 구멍의 교차점 찾기
    endpoints = []
    
    if len(lead_lines) >= 2 and len(hole_centers) >= 2:
        # 각 리드라인에 대해 가장 가까운 구멍 찾기
        for line in lead_lines[:2]:  # 처음 2개 리드만 처리
            line_points = np.array([line[0], line[1]])
            hole_array = np.array(hole_centers)
            
            # 라인의 각 끝점에서 가장 가까운 구멍 찾기
            distances = cdist(line_points, hole_array)
            min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
            
            closest_hole = hole_centers[min_dist_idx[1]]
            endpoints.append(closest_hole)
    
    # 10. 대안: 리드라인의 끝점을 직접 사용
    if len(endpoints) < 2 and len(lead_lines) >= 2:
        for line in lead_lines[:2]:
            # 아래쪽 끝점 선택 (일반적으로 브레드보드는 아래쪽)
            y1, y2 = line[0][1], line[1][1]
            if y1 > y2:
                endpoints.append(line[0])
            else:
                endpoints.append(line[1])
    
    # 11. 최종 결과가 없다면 이미지 중심부에서 추정
    if len(endpoints) < 2:
        height, width = gray.shape
        # 이미지를 세로로 나누어 좌우에서 각각 1개씩 찾기
        left_region = gray[:, :width//2]
        right_region = gray[:, width//2:]
        
        # 각 영역에서 가장 어두운 점 찾기 (구멍 위치 추정)
        left_min_loc = np.unravel_index(np.argmin(left_region), left_region.shape)
        right_min_loc = np.unravel_index(np.argmin(right_region), right_region.shape)
        
        endpoints = [
            (left_min_loc[1], left_min_loc[0]),
            (right_min_loc[1] + width//2, right_min_loc[0])
        ]
    
    # 12. 결과 시각화
    for i, point in enumerate(endpoints[:2]):
        cv2.circle(result_image, point, 5, (0, 0, 255), -1)  # 빨간색 점
        cv2.putText(result_image, f'P{i+1}', 
                   (point[0]+10, point[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 디버깅용: 검출된 라인들도 표시
    for line in lead_lines[:2]:
        cv2.line(result_image, line[0], line[1], (0, 255, 0), 2)
    
    # 검출된 구멍들도 표시
    for center in hole_centers:
        cv2.circle(result_image, center, 3, (255, 0, 0), 2)
    
    return endpoints[:2], result_image

def main():
    """메인 함수"""
    try:
        # 이미지 파일 경로 (실제 파일명으로 변경해주세요)
        image_path = "led2.jpg"
        
        # LED 리드 끝점 검출
        endpoints, result_image = detect_led_lead_endpoints(image_path)
        
        # 결과 출력
        print("검출된 LED 리드 끝점 좌표:")
        print(endpoints)
        
        # 결과 이미지 표시
        plt.figure(figsize=(12, 8))
        
        # 원본 이미지
        original = cv2.imread(image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("원본 이미지")
        plt.axis('off')
        
        # 결과 이미지
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("검출된 끝점")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 결과 저장
        cv2.imwrite("led_detection_result.png", result_image)
        print("결과 이미지가 'led_detection_result.png'로 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("이미지 파일 경로를 확인해주세요.")

if __name__ == "__main__":
    main()