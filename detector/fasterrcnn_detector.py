# detector/fasterrcnn_detector.py
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2

class FasterRCNNDetector:
    def __init__(self, model_path=None, num_classes=8):  # 8 classes + background
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.class_names = ['background', 'Breadboard', 'Capacitor', 'Diode', 'IC', 'LED', 'Line_area', 'Resistor']

    def detect(self, image):
        image_tensor = F.to_tensor(image).to(self.device)
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        components = []
        for label, box, score in zip(predictions['labels'], predictions['boxes'], predictions['scores']):
            if score < 0.5:
                continue
            cls_name = self.class_names[label]
            x1, y1, x2, y2 = box.int().tolist()
            components.append((cls_name, float(score), (x1, y1, x2, y2)))
        return components
