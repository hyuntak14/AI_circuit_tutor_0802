import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 설정
MODEL_DIR = r'D:\Hyuntak\cropped_for_labelme\code'
INPUT_SIZE = (224, 224)

# 라벨 정보 불러오는 함수
def load_ground_truth_and_images(root_dir, class_name, input_size=(224, 224)):
    import json, cv2
    from glob import glob

    folder = os.path.join(root_dir, class_name)
    images, keypoints = [], []

    for json_path in glob(os.path.join(folder, '*.json')):
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_path = os.path.join(folder, data['imagePath'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, input_size)
        img = img.astype(np.float32) / 255.0

        w, h = data['imageWidth'], data['imageHeight']
        pts = []
        for shape in data['shapes']:
            x, y = shape['points'][0]
            pts.extend([x / w, y / h])

        images.append(img)
        keypoints.append(pts)

    return np.array(images), np.array(keypoints)

# 커스텀 정확도
def custom_accuracy_np(y_true, y_pred, threshold=0.1):
    correct = np.abs(y_true - y_pred) <= threshold
    return np.mean(correct)

# 모델 평가
def evaluate_model(model_path, class_name):
    model = tf.keras.models.load_model(model_path, compile=False)
    X, y_true = load_ground_truth_and_images(os.path.dirname(MODEL_DIR), class_name)
    if len(X) == 0:
        return None
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    acc = custom_accuracy_np(y_true, y_pred)
    return mae, mse, r2, acc

# 모든 모델 비교
def compare_models():
    from prettytable import PrettyTable
    table = PrettyTable(['Model', 'Class', 'MAE', 'MSE', 'R2', 'Custom Acc'])

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith('.h5'):
            model_path = os.path.join(MODEL_DIR, filename)
            parts = filename.replace('.h5', '').split('_')
            backbone = parts[0]
            class_name = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]

            result = evaluate_model(model_path, class_name)
            if result is None:
                continue

            mae, mse, r2, acc = result
            table.add_row([backbone, class_name, round(mae, 4), round(mse, 4), round(r2, 4), round(acc, 4)])

    print("\n📊 [모델별 성능 비교표]")
    print(table)

if __name__ == '__main__':
    compare_models()
