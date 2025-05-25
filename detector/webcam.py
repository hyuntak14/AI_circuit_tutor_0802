# webcam_capture.py
import cv2
import os

class WebcamCapture:
    def __init__(self, output_dir='captured_images'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def capture_image(self, filename='capture.jpg'):
        """
        웹캠을 켜서 사진을 찍고 저장합니다.
        :param filename: 저장할 파일 이름 (기본값: capture.jpg)
        :return: 저장된 이미지의 경로
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("웹캠을 열 수 없습니다.")

        print("웹캠을 켭니다. 스페이스바로 사진을 찍습니다. ESC로 종료합니다.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            cv2.imshow('Webcam - Press SPACE to capture', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("캡처 종료.")
                break
            elif key == 32:  # SPACE
                save_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(save_path, frame)
                print(f"이미지가 저장되었습니다: {save_path}")
                cap.release()
                cv2.destroyAllWindows()
                return save_path

        cap.release()
        cv2.destroyAllWindows()
        return None
