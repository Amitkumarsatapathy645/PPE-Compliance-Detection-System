import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results.json') as f:
    data = json.load(f)

models = [d['model'] for d in data]
map50  = [d['mAP50'] for d in data]
prec   = [d['Precision'] for d in data]
rec    = [d['Recall'] for d in data]
f1     = [d['F1'] for d in data]
fps    = [d['FPS'] for d in data]

# ----------- PRINT TABLE (VERY IMPORTANT) -----------
print("\nFINAL COMPARISON")
print("-"*70)
print(f"{'Model':<15} {'mAP50':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPS':>7}")
print("-"*70)

for d in data:
    print(f"{d['model']:<15} {d['mAP50']:>7.3f} {d['Precision']:>7.3f} "
          f"{d['Recall']:>7.3f} {d['F1']:>7.3f} {d['FPS']:>7.1f}")

print("-"*70)


# ----------- PLOT 1: ACCURACY COMPARISON -----------
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - width, map50, width, label='mAP@0.5')
plt.bar(x, prec, width, label='Precision')
plt.bar(x + width, rec, width, label='Recall')

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Model Accuracy Comparison")
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.savefig("accuracy_comparison.png", dpi=150)
plt.show()


# ----------- PLOT 2: SPEED COMPARISON -----------
plt.figure(figsize=(8, 5))
bars = plt.bar(models, fps)

plt.ylabel("FPS")
plt.title("Inference Speed Comparison")

for bar, val in zip(bars, fps):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{val:.1f}",
             ha='center')

plt.grid(axis='y', alpha=0.3)
plt.savefig("speed_comparison.png", dpi=150)
plt.show()


# ----------- PLOT 3: SPEED vs ACCURACY -----------
plt.figure(figsize=(8, 6))

for d in data:
    plt.scatter(d['FPS'], d['mAP50'], s=200)
    plt.text(d['FPS']+0.5, d['mAP50'], d['model'])

plt.xlabel("FPS (Speed)")
plt.ylabel("mAP@0.5 (Accuracy)")
plt.title("Speed vs Accuracy Trade-off")

plt.axhline(0.9, linestyle='--', alpha=0.5)
plt.axvline(30, linestyle='--', alpha=0.5)

plt.grid(alpha=0.3)
plt.savefig("tradeoff.png", dpi=150)
plt.show()