# keypoint_detector.py (또는 main.py 내 KeypointDetector 정의 부분)
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 직접 mse 함수를 정의합니다.
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class KeypointDetector:
    def __init__(self, model_paths):
        """
        model_paths: 각 소자별 키포인트 모델의 절대 경로가 담긴 딕셔너리
                     예: {
                           'Capacitor': r"...\densenet_Capacitor_keypoint_model.h5",
                           'Diode':     r"...\densenet_Diode_keypoint_model.h5",
                           ...
                         }
        """
        self.model_paths = model_paths
        self.models = {}  # 소자별로 로드한 모델 캐싱

    def get_model(self, component_type):
        if component_type not in self.models:
            if component_type not in self.model_paths:
                raise ValueError(f"모델 경로가 제공되지 않았습니다: {component_type}")
            model_path = self.model_paths[component_type]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            # custom_objects로 직접 정의한 mse 함수를 등록합니다.
            self.models[component_type] = load_model(model_path, custom_objects={'mse': mse})
        return self.models[component_type]

    def detect_keypoint(self, image, bbox, component_type):
        """
        image: 원본 이미지 (BGR numpy array)
        bbox: (x1, y1, x2, y2) 형태의 소자 바운딩 박스
        component_type: 소자 종류 (예: 'Diode', 'IC' 등)
        returns: 원본 이미지 좌표계 상의 여러 (x, y) 좌표 리스트
                 - IC: 8개의 포인트, 그 외: 2개의 포인트 반환
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        input_size = (224, 224)
        roi_resized = cv2.resize(roi, input_size)
        roi_preprocessed = roi_resized.astype("float32") / 255.0
        roi_input = np.expand_dims(roi_preprocessed, axis=0)
        model = self.get_model(component_type)
        pred = model.predict(roi_input)
        keypoint_norm = pred[0]  # 예측 결과: 정규화된 좌표 벡터
        
        # 소자 종류에 따른 기대 keypoint 수 (IC: 8, 나머지: 2)
        num_points = 8 if component_type.upper() == 'IC' else 2
        expected_length = 2 * num_points  # 각 포인트마다 x, y 2개씩
        
        if len(keypoint_norm) < expected_length:
            raise ValueError(f"모델의 출력 차원이 예상({expected_length})보다 작습니다: {len(keypoint_norm)} (확실하지 않음)")
        
        keypoints = []
        for i in range(num_points):
            norm_x = keypoint_norm[2*i]
            norm_y = keypoint_norm[2*i+1]
            pt_x = norm_x * (x2 - x1) + x1
            pt_y = norm_y * (y2 - y1) + y1
            keypoints.append((int(pt_x), int(pt_y)))
        return keypoints
