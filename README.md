# 🐶 Dog Emotion Detection using YOLOv8 

## 📌 Project Overview
This project combines **YOLOv8** (for dog detection) with **ResNet50** (for feature extraction) to build a pipeline that can detect dogs in images/videos and extract **emotion-related features**.  
The extracted features are aggregated over time using **temporal pooling** (average or max) to generate a final **video-level feature vector**.  
These features can later be used for **emotion classification** or other downstream ML tasks.

---

## 🚀 Features
- Train YOLOv8 on a custom dog emotion dataset.
- Detect dogs in images, videos, or live webcam streams.
- Extract deep features using ResNet50 (pre-trained on ImageNet).
- Apply **temporal pooling** across frames to build a video-level feature vector.
- Save the extracted features for use in classification models (SVM, MLP, etc.).
- Real-time detection and feature extraction using OpenCV.

---

## 🛠 Requirements
Make sure you have the following installed:

```bash
python >= 3.8
torch >= 2.0.0
torchvision >= 0.15.0
ultralytics >= 8.0.0
opencv-python
numpy
pillow
matplotlib
seaborn
scikit-learn
