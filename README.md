# 🚦Traffic Sign Detection with YOLOv8 under Few-Shot and Low-Data Regimes

This project investigates the ability of the YOLOv8 object detection model to learn robust representations of traffic signs under limited data conditions. We assess how performance evolves as we increase the size of the training dataset, starting from a *few-shot* setup with as little as 4 examples per class.

---

##  Project Overview

Traffic sign detection is a critical task in autonomous driving systems, requiring high-speed and high-accuracy recognition of small, fine-grained objects. Traditional models like Faster R-CNN or SSD perform well in accuracy but often fall short on real-time constraints. YOLOv8, the latest model in the YOLO family, is designed for real-time detection with competitive accuracy.

While YOLOv8 has achieved state-of-the-art results on general object detection benchmarks, its performance under **low-data regimes**—particularly for small, domain-specific datasets like traffic signs—has not been thoroughly explored. This project focuses precisely on that, evaluating YOLOv8's performance as we scale the dataset size.

We train YOLOv8 on incrementally larger subsets of a traffic sign dataset and track:
- **F1 Score** and **Average Precision (AP@0.50)** per class
- Confusion matrices and confidence-thresholding behavior
- Visual inspection of misclassifications and missed detections

---

##  Dataset

- **Source**: Public traffic sign dataset from Kaggle
- **Content**: 14 traffic sign classes, including:
  `Green Light`, `Red Light`, `Stop`, `Speed Limit 20` through `Speed Limit 120`
- **Format**: YOLO-style object detection labels (normalized bounding boxes)
- **Size**:
  - Train: 3530 images
  - Validation: 801 images
  - Test: 638 images
- **Image Specs**: 416×416 resolution, RGB,  some dashcam-like perspectives

**Class Imbalance Handling**:
- Dropped severely underrepresented class `Speed Limit 10`
- Maintained rare but interesting classes like `Speed Limit 110` to enable error analysis

**Few-Shot Regime**:
- Minimum of 4 examples per class
- Maintains class distribution across all training percentages (6%, 13%, 25%, 100%)

---

##  Methodology

- **Base Model**: YOLOv8 (pretrained on COCO 2017)
- **Fine-Tuning**:
  - Loss: Cross Entropy (Focal Loss tested but not needed)
  - Optimizer: AdamW
  - No data augmentation
  - All layers unfrozen during training
  - Training conducted on NVIDIA GeForce RTX 3050 GPU
- **Evaluation Metrics**:
  - Precision, Recall, F1 Score
  - AP@0.50 (loose box overlap)
  - AP@0.50:0.95 (stricter localization)

---

## Interesting Results

- Results show that YOLOv8 can learn meaningful traffic sign representations with very few examples. 
- Under such extreme data scarcity, data quality is crucial. Some classes training signal was poor and it affected the best performing ones, lowering their precision. 
- Confidence-threshold tuning adapted to each class appears to be necessary in low data regimes. 
- **Traffic lights** underperformed despite being the most frequent class — possibly due to non-contrastive backgrounds
- **Best performing class** (few-shot, 11 examples in training): `Stop` — 95% Precision, 77% Recall

Plots in the repo include:
- F1 Score and AP@0.50 evolution across dataset sizes
- Confusion matrices and per-class confidence calibration
- Visual inspection of predictions

---

## Authors:
- Tirdod Behbehani (tirdod.behbehani@bse.eu)
- Alejandro Delgado (alejandro.delgado@bse.eu)
- Alex Malo (alex.malo@bse.eu)
