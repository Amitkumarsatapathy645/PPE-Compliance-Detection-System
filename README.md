# 🦺 PPE Compliance Detection System using YOLOv8

A real-time Personal Protective Equipment (PPE) detection system built using **YOLOv8** and **OpenCV** to monitor workplace safety and detect PPE violations across multiple object classes.

## 🚀 Features

* Real-time PPE detection using webcam/video input
* Detection across **16 object classes**
* Image and video inference support
* Performance evaluation and model comparison
* Violation logging system
* Data augmentation and exploratory data analysis (EDA)
* Accuracy and speed tradeoff analysis

---

## 🛠️ Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy
* Matplotlib
* Pandas
* Scikit-learn

---

## 📂 Project Structure

```text
PPE-Compliance-Detection-System
│
├── train_yolov8.py
├── predict.py
├── realtime.py
├── evaluate.py
├── comparison.py
├── eda.py
├── results.json
├── requirements.txt
├── accuracy_comparison.png
├── augmentation_examples.png
├── eda_results.png
├── single_model_performance.png
├── speed_comparison.png
├── tradeoff.png
└── README.md
```

---

## 📊 Model Performance

| Metric       | Score |
| ------------ | ----- |
| Precision    | 73.2% |
| Recall       | 51.1% |
| mAP@0.5      | 56.2% |
| mAP@0.5:0.95 | 35.9% |

---

## 📈 Performance Analysis

The model was evaluated using multiple metrics and comparison graphs:

* Accuracy comparison
* Speed comparison
* Precision vs Recall tradeoff
* Single-model performance analysis
* Data augmentation examples

---

## 🎯 Use Cases

* Construction site safety monitoring
* Industrial worker compliance
* Manufacturing plants
* Smart surveillance systems
* Workplace hazard prevention

---

## ▶️ Run Inference

### Real-Time Detection

```bash
python realtime.py
```

### Image Prediction

```bash
python predict.py
```

### Model Evaluation

```bash
python evaluate.py
```

---

## 📌 Future Improvements

* Deploy using Streamlit
* Add email/SMS alerts for violations
* Dashboard for analytics and reporting
* Edge deployment with TensorRT
* Cloud-based monitoring system

---

## 👨‍💻 Author

**Amit Kumar Satapathy**

* LinkedIn: https://www.linkedin.com/in/amit-kumar-satapathy-59547722a
* GitHub: https://github.com/Amitkumarsatapathy645
