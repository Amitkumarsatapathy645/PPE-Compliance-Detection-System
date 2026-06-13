from ultralytics import YOLO
import torch

print(f"GPU available: {torch.cuda.is_available()}")

def train_model():
    # Load model
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='./PPE-0.3-1/data.yaml',
        epochs=10,
        imgsz=640,
        batch=4,
        workers=2,
        cache=False,
        name='ppe_detector',
        project='runs',
        patience=5,
        device=0,
        pretrained=True,
        augment=True,
        seed=42,
        verbose=True
    )

    print("Training complete!")
    return results


if __name__ == "__main__":
    train_model()