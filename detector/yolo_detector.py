from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect_breadboard(self, image_path):
        results = self.model(image_path)
        r = results[0]
        image = r.orig_img.copy()
        breadboard_boxes = []

        for box in r.boxes:
            cls_id = int(box.cls[0])
            if self.class_names[cls_id].lower() == 'breadboard':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                breadboard_boxes.append((x1, y1, x2, y2))

        return image, breadboard_boxes, self.class_names

    def detect_components(self, image):
        results = self.model(image)[0]
        components = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            components.append((self.class_names[cls_id], conf, (x1, y1, x2, y2)))
        return components
