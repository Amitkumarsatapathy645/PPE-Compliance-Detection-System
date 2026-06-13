from ultralytics import YOLO
import cv2
import datetime
import os
model = YOLO('C:\\Users\\amitk\\OneDrive\\Desktop\\runs\\detect\\runs\\ppe_detector-2\\weights\\best (4).pt')
os.makedirs("violations", exist_ok=True)
cap = cv2.VideoCapture(0)
violation_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated = results[0].plot()
    names = results[0].names
    ids = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
    labels = [names[int(i)] for i in ids]

    alerts = []
    if "Without Helmet" in labels:
        alerts.append("No Helmet")
    if "Without Gloves" in labels:
        alerts.append("No Gloves")
    if "Without Mask" in labels:
        alerts.append("No Mask")

    if alerts:
        violation_count += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alert_text = " | ".join(alerts)

        cv2.putText(annotated, f"ALERT: {alert_text}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        print(f"ALERT: {alert_text}")

        with open("violations_log.txt", "a") as f:
            f.write(f"{timestamp} - {alert_text}\n")
        cv2.imwrite(f"violations/{timestamp}.jpg", annotated)

    cv2.putText(annotated, f"Total Violations: {violation_count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    cv2.imshow("PPE Detection System", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()