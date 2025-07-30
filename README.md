### Street Object Detection with YOLOv10

The script yolo_custom_model.py provides a full training pipeline for street object detection using the [YOLOv10](https://github.com/THU-MIG/yolov10) deep learning model. It supports custom datasets with automatic data augmentation, multi-GPU training, fine-tuning, hyperparameter tuning, and real-time GPU monitoring.

### Street Object Detection and Multi-Class Tracking with YOLOv10 and BYTETrack
The script object_tracker_mot.py performs high-performance object detection and multi-class tracking on dashcam or street-level video data using YOLOv10 and BYTETrack. It processes multiple videos, applies two trained YOLOv10 models (e.g., for street objects and road markings), and outputs detection results in both tracking and raw detection formats.
