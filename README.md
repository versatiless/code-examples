# 🛣️ Street Object Detection with YOLOv10

This project provides a full training pipeline for **street object detection** using the [YOLOv10](https://github.com/THU-MIG/yolov10) deep learning model. It supports custom datasets with automatic data augmentation, multi-GPU training, fine-tuning, hyperparameter tuning, and real-time GPU monitoring.

## 🚀 Features

- Train YOLOv10 on custom street datasets (e.g., cracks, signs, barriers)
- Apply rich **image augmentations** (rain, snow, fog, blur, contrast, scaling, etc.) using [Albumentations](https://albumentations.ai/)
- Supports both **low-quality** and **high-quality** image folders with separate augmentation strategies
- Easily **fine-tune** from pretrained weights or best checkpoints
- Tune hyperparameters via YAML configuration and custom search spaces
- Monitor GPU usage in real-time with `GPUtil`
- Built with **PyTorch**, **Ray**, and **Ultralytics YOLOv10**

## 🧠 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- albumentations
- ray
- supervision
- GPUtil
- [YOLOv10](https://github.com/THU-MIG/yolov10) (must be cloned locally)

## 📂 Directory Structure

```bash
StreetObjectDetection/
├── config/                  # Dataset YAML files
├── weights/                 # Pretrained or trained model weights
├── runs/train/              # Training outputs
├── yolov10/                 # YOLOv10 repo (cloned manually)
├── object_detection.py      # Main training logic
