from ultralytics import YOLO

model = YOLO(r"C:\Users\amitk\OneDrive\Desktop\runs\detect\runs\ppe_detector\weights\best.pt")

model.predict(
    source='PPE-0.3-1/test/images',
    save=True,
    conf=0.25
)