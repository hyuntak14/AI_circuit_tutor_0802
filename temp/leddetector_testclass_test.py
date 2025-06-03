import os
import cv2
from led_detector_testclass import LedEndpointDetector

detector = LedEndpointDetector(visualize=False)

# 현재 폴더의 "led"가 포함된 이미지 목록
image_files = [f for f in os.listdir('.') if 'led' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print("검출할 파일 목록:", image_files)

for file in image_files:
    image = cv2.imread(file)
    if image is None:
        print(f"[경고] {file}은(는) 읽을 수 없는 파일입니다.")
        continue

    h, w = image.shape[:2]
    bbox = (0, 0, w, h)
    holes = []

    # 1) _preprocess를 통해 중간 결과 시각화
    gray, gamma, bw = detector._preprocess(image)

    # 중간 결과 시각화
    cv2.imshow("Original", image)
    cv2.imshow("Gray", gray)
    cv2.imshow("Gamma Corrected", gamma)
    cv2.imshow("Binarized (bw)", bw)

    # 2) extract 수행
    result = detector.extract(image, bbox, holes)

    if result is not None:
        print(f"[{file}] 검출된 양끝점:", result['endpoints'])
        # 최종 결과 이미지 시각화
        detector.draw(image, result, holes)
        cv2.imshow("Final Result", image)
        cv2.imwrite(f"output_{file}", image)
    else:
        print(f"[{file}] 양끝점을 검출할 수 없습니다.")

    # 사용자 확인
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("검출 및 시각화 완료.")
