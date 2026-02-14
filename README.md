# 🎯 Pixel Coordinate Prediction

> Deep learning model to predict **(x, y)** coordinates of a single bright pixel in 50×50 grayscale images.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)

---

## Problem

Given a 50×50 image with one pixel at 255 and the rest at 0, predict the exact **(x, y)** location—a supervised regression task with spatial structure.

## Approach

| Step | Choice |
|------|--------|
| **Model** | Fully convolutional CNN → 50×50 heatmap output |
| **Why heatmap?** | Direct regression + pooling loses spatial precision (~20–35 px error). Heatmaps preserve full resolution → ~1–3 px error. |
| **Robustness** | Noise augmentation (clean + noisy training) for better performance on noisy inputs |

Coordinates are extracted from the heatmap via **soft-argmax**—standard in pose estimation and keypoint detection.

## Quick Start

```bash
git clone <repo-url>
cd deepedge
pip install -r requirements.txt
jupyter notebook pixel_coordinate_prediction.ipynb
```

## Results

- **Clean data:** ~1–3 px average error, R² ≈ 1.0  
- **Noisy data:** Robust after augmentation  
- **Architecture:** No MaxPooling, no Flatten—spatial resolution preserved end-to-end

## Structure

```
├── pixel_coordinate_prediction.ipynb   # Main notebook
├── requirements.txt
└── (best_model.keras, training_history.csv, etc. — generated on run)
```

---

*ML Assignment — Supervised Regression*
