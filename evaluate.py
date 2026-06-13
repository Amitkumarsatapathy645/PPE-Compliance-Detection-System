from ultralytics import YOLO
import json
import os

DATA_YAML  = './PPE-0.3-1/data.yaml'
TEST_IMAGE = './PPE-0.3-1/test/images'


def evaluate_yolo_model(weights_path, model_name):
    print(f"\nEvaluating {model_name}...")
    
    model = YOLO(weights_path)
    metrics = model.val(data=DATA_YAML, split='test')
    
    results = {
        'model':      model_name,
        'mAP50':      round(float(metrics.box.map50), 4),
        'mAP50_95':   round(float(metrics.box.map),   4),
        'Precision':  round(float(metrics.box.p.mean()), 4),
        'Recall':     round(float(metrics.box.r.mean()), 4),
    }
    
    results['F1'] = round(
        2 * results['Precision'] * results['Recall'] /
        (results['Precision'] + results['Recall'] + 1e-8),
        4
    )
    
    print(results)
    return results


def measure_speed(weights_path, model_name, image_path):
    import time
    import numpy as np
    import cv2
    
    model = YOLO(weights_path)
    img = cv2.imread(image_path)
    
    # Warmup
    for _ in range(5):
        model.predict(img, verbose=False)
    
    times = []
    for _ in range(30):
        start = time.perf_counter()
        model.predict(img, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_ms = round(np.mean(times), 2)
    fps    = round(1000 / avg_ms, 1)
    
    print(f"{model_name}: {fps} FPS")
    return avg_ms, fps


if __name__ == "__main__":
    
    test_imgs = os.listdir(TEST_IMAGE)
    one_img   = os.path.join(TEST_IMAGE, test_imgs[0])
    
    all_results = []
    
    r = evaluate_yolo_model(
        r"C:\Users\amitk\OneDrive\Desktop\runs\detect\runs\ppe_detector\weights\best.pt",
        'YOLOv8n'
    )
    ms, fps = measure_speed(
        r"C:\Users\amitk\OneDrive\Desktop\runs\detect\runs\ppe_detector\weights\best.pt",
        'YOLOv8n', one_img
    )
    r['FPS'] = fps
    all_results.append(r)
    
    
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nSaved results.json")