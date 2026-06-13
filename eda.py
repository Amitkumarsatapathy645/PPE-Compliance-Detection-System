import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Dataset Path ───────────────────────────────────────
DATASET_PATH = './PPE-0.3-1'

# ── Class Names (from data.yaml) ───────────────────────
CLASS_NAMES = [
    'Ear Protectors', 'Glasses', 'Gloves', 'Helmet', 'Mask',
    'Person', 'Safety_shoes', 'Shoes', 'Vest',
    'Without Ear Protectors', 'Without Glass', 'Without Glove',
    'Without Helmet', 'Without Mask', 'Without Shoes', 'Without Vest'
]


def run_eda():

    train_images = DATASET_PATH + '/train/images'
    train_labels = DATASET_PATH + '/train/labels'
    val_images   = DATASET_PATH + '/valid/images'
    test_images  = DATASET_PATH + '/test/images'

    # ── Count Images ────────────────────────────────────
    train_count = len(os.listdir(train_images))
    val_count   = len(os.listdir(val_images))
    test_count  = len(os.listdir(test_images))

    print("="*40)
    print("DATASET SUMMARY")
    print("="*40)
    print(f"Training images:   {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images:       {test_count}")
    print(f"Total:             {train_count+val_count+test_count}")

    # ── Count ALL Classes ───────────────────────────────
    class_counts = [0] * len(CLASS_NAMES)

    for lbl_file in Path(train_labels).glob('*.txt'):
        with open(lbl_file) as f:
            for line in f:
                cls = int(line.strip().split()[0])
                class_counts[cls] += 1

    print("\nCLASS DISTRIBUTION:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name}: {class_counts[i]}")

    # ── Plot 1: Class Distribution ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Dataset EDA - PPE Detection (16 Classes)',
                 fontsize=14, fontweight='bold')

    axes[0].barh(CLASS_NAMES, class_counts, color='skyblue')
    axes[0].set_title('Class Distribution')
    axes[0].set_xlabel('Count')

    # ── Plot 2: Dataset Split ───────────────────────────
    axes[1].pie([train_count, val_count, test_count],
                labels=['Train', 'Validation', 'Test'],
                autopct='%1.1f%%')
    axes[1].set_title('Dataset Split')

    # ── Plot 3: Sample Image ────────────────────────────
    sample_imgs = list(Path(train_images).glob('*.jpg'))[:4]
    if not sample_imgs:
        sample_imgs = list(Path(train_images).glob('*.png'))[:4]

    axes[2].axis('off')
    axes[2].set_title('Sample Image Preview')

    if sample_imgs:
        img = cv2.imread(str(sample_imgs[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[2].imshow(img)

    plt.tight_layout()
    plt.savefig('eda_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nSaved: eda_results.png")

    # ── Augmentation Preview ────────────────────────────
    if sample_imgs:
        show_augmentations(str(sample_imgs[0]))


def show_augmentations(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Data Augmentation Examples',
                 fontsize=13, fontweight='bold')

    augmented = [
        ('Original', img),
        ('Horizontal Flip', cv2.flip(img, 1)),
        ('Brightness +', cv2.convertScaleAbs(img, alpha=1.4, beta=30)),
        ('Blur', cv2.GaussianBlur(img, (7, 7), 0)),
        ('Darker', cv2.convertScaleAbs(img, alpha=0.6, beta=0)),
        ('Rotated 15°', cv2.warpAffine(
            img,
            cv2.getRotationMatrix2D(
                (img.shape[1]//2, img.shape[0]//2),
                15, 1.0),
            (img.shape[1], img.shape[0])
        )),
    ]

    for i, (title, aug_img) in enumerate(augmented):
        ax = axes[i//3][i % 3]
        ax.imshow(aug_img)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Saved: augmentation_examples.png")


if __name__ == "__main__":
    run_eda()