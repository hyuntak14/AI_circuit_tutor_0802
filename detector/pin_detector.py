import cv2
import numpy as np

class PinDetector:
    def __init__(self):
        pass

    def detect_pins(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
            param1=50, param2=15, minRadius=2, maxRadius=6
        )

        pins = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                pins.append((x, y))
                #cv2.circle(image, (x, y), r, (0, 0, 255), 2)   #핀홀 이미지 시각화
                #cv2.circle(image, (x, y), 2, (255, 0, 0), 3)

        return image, pins
