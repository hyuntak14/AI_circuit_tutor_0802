import cv2
import matplotlib.pyplot as plt
import sys

# detector 패키지가 있는 breadboard_project 상위 폴더를 추가
sys.path.append(r'd:\Hyuntak\연구실\AR 회로 튜터\breadboard_project')

from wire_detector import WireDetector

def test_wire_detector(img_path):
    # 전체 이미지 로드
    full_img = cv2.imread(img_path)
    if full_img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

    cropped = full_img.copy()
    detector = WireDetector(kernel_size=5)
    # configure_white_thresholds 우회
    detector.configure_white_thresholds = lambda img: setattr(
        detector, 'full_white_mask',
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    )
    detector.full_white_mask = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

    # 전체 영역 bbox
    h, w = full_img.shape[:2]
    bbox = (0, 0, w, h)
    white_mask = detector.extract_white_wire_mask(full_img, bbox)

    # black/red wire mask 및 skeleton 계산
    wire_segments = detector.detect_wires(cropped)
    endpoints, channel = detector.select_best_endpoints(wire_segments)

    # -- 시각화: 1~4 --
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title('Cropped Image')
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('White Wire Mask')
    plt.imshow(white_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Black Wire Mask')
    plt.imshow(wire_segments['black']['mask'], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title(f"Skeleton & Endpoints ({channel})")
    skel = wire_segments[channel]['skeleton']
    plt.imshow(skel, cmap='gray')
    for (x, y) in endpoints:
        plt.scatter(x, y, s=50, c='red', marker='o')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # -- 최종 검출 결과: 원본에 오버레이 --
    overlay = full_img.copy()
    for (x, y) in endpoints:
        cv2.circle(overlay, (x, y), 8, (0, 0, 255), 2)
    cv2.putText(overlay, f"Channel: {channel}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # OpenCV 윈도우로 보기
    cv2.imshow("Final Detection Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import os
import glob

def test_all_wire_files():
    """
    현재 디렉토리에서 파일명에 'wire'가 포함된 모든 이미지 파일을 찾아서 테스트
    """
    # 지원하는 이미지 확장자들
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # wire가 포함된 모든 이미지 파일 찾기
    wire_files = []
    for ext in image_extensions:
        # 대소문자 구분없이 찾기
        pattern = f"*wire*{ext}"
        wire_files.extend(glob.glob(pattern, recursive=False))
        
        # 대문자 확장자도 찾기
        pattern_upper = f"*wire*{ext.upper()}"
        wire_files.extend(glob.glob(pattern_upper, recursive=False))
        
        # WIRE 대문자도 찾기
        pattern_wire_upper = f"*WIRE*{ext}"
        wire_files.extend(glob.glob(pattern_wire_upper, recursive=False))
        
        pattern_wire_upper2 = f"*WIRE*{ext.upper()}"
        wire_files.extend(glob.glob(pattern_wire_upper2, recursive=False))
    
    # 중복 제거
    wire_files = list(set(wire_files))
    
    if not wire_files:
        print("파일명에 'wire'가 포함된 이미지 파일을 찾을 수 없습니다.")
        print("현재 디렉토리의 파일들:")
        for f in os.listdir('.'):
            if any(f.lower().endswith(ext[1:]) for ext in image_extensions):
                print(f"  - {f}")
        return
    
    print(f"찾은 wire 파일들: {len(wire_files)}개")
    for f in wire_files:
        print(f"  - {f}")
    print()
    
    # 각 파일에 대해 테스트 실행
    for i, img_path in enumerate(sorted(wire_files), 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(wire_files)}] 처리 중: {img_path}")
        print(f"{'='*50}")
        
        try:
            test_wire_detector(img_path)
            print(f"✓ {img_path} 처리 완료")
        except Exception as e:
            print(f"✗ {img_path} 처리 실패: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print(f"전체 {len(wire_files)}개 파일 처리 완료")
    print(f"{'='*50}")


def test_specific_wire_files(file_pattern="wire*.jpg"):
    """
    특정 패턴의 wire 파일들만 테스트
    
    Args:
        file_pattern (str): 파일 패턴 (예: "wire*.jpg", "*wire*.*")
    """
    wire_files = glob.glob(file_pattern)
    
    if not wire_files:
        print(f"패턴 '{file_pattern}'에 맞는 파일을 찾을 수 없습니다.")
        return
    
    print(f"패턴 '{file_pattern}'로 찾은 파일들: {len(wire_files)}개")
    for f in sorted(wire_files):
        print(f"  - {f}")
    print()
    
    for i, img_path in enumerate(sorted(wire_files), 1):
        print(f"\n[{i}/{len(wire_files)}] 처리 중: {img_path}")
        try:
            test_wire_detector(img_path)
            print(f"✓ {img_path} 처리 완료")
        except Exception as e:
            print(f"✗ {img_path} 처리 실패: {str(e)}")


if __name__ == '__main__':
    # 방법 1: 모든 wire 파일 자동 검색 및 테스트
    test_all_wire_files()
    
    # 방법 2: 특정 패턴만 테스트하고 싶은 경우 (주석 해제)
    # test_specific_wire_files("wire*.jpg")
    # test_specific_wire_files("*wire*.*")  # 모든 확장자
    
    # 방법 3: 기존처럼 특정 파일 하나만 테스트
    # img_path = "wire4.jpg"
    # test_wire_detector(img_path)
