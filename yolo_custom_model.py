import os
import os.path as osp
from glob import glob
import cv2
import supervision as sv
import sys
home_path = osp.expanduser("~")
sys.path.append(osp.join(home_path, "StreetObjectDetection/yolov10"))
from ultralytics import YOLOv10
import torch
import time
import GPUtil
from torch.utils.data import Dataset, DataLoader
import ray
ray.init(num_cpus=2)
import gc
import random
from ultralytics.data.augment import Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
import torch.nn as nn


def __init__(self, p=0.1):
    self.p = p
    self.transform = None
    T = [
        # A.RandAugment(num_ops=3, magnitude=3, p=0.5),
        A.MotionBlur(p=0.2),  # Apply motion blur with 20% probability
        A.RandomRain(p=0.2, brightness_coefficient=0.8, drop_width=1, blur_value=3),  # Simulate rain
        A.RandomSnow(p=0.2, brightness_coeff=1.2),  # Simulate snow
        A.RandomFog(p=0.2),  # Simulate fog
        A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        A.RandomScale(scale_limit=0.3, p=0.4),  # Slightly scale the image
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # Add noise
        A.ToGray(p=0.1),  # Occasionally make grayscale
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.3),
        ToTensorV2()
    ]
    self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo",label_fields=["class_labels"]))



class CustomDataset:
    def __init__(self, dataset_yaml, mode='train'):
        self.augment = (mode == "train")

        # Load dataset paths from YAML file
        with open(dataset_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)

        self.image_paths = []
        self.label_paths = []

        self.low_quality_folders = ['fix_only_marking_v6']
        self.high_quality_folders = ['ceymo']

        for img_dir in data_cfg[mode]:
            for img_file in os.listdir(img_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(img_dir, img_file))
                    self.label_paths.append(os.path.join(img_dir, img_file.replace(".jpg", ".txt")))

        # Define augmentation strategies
        self.high_quality_transform = A.Compose([
            A.MotionBlur(p=0.3),
            A.RandomRain(p=0.3),
            A.RandomSnow(p=0.3),
            A.RandomFog(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomScale(scale_limit=0.3, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        self.low_quality_transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_path = self.label_paths[index]
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read labels (YOLO format)
        with open(label_path, "r") as file:
            labels = [line.strip().split() for line in file.readlines()]

        h, w = image.shape[:2]
        bboxes = []
        class_labels = []
        for label in labels:
            class_id = int(label[0])
            x_center, y_center, width, height = map(float, label[1:])
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w)
            y_max = int((y_center + height / 2) * h)
            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(class_id)

        # Apply different augmentations based on the folder
        if self.augment:
            tag = 'low'
            for folder in self.high_quality_folders:
                if folder in img_path:
                    tag = 'high'
                    break
            if tag == "high":
                transformed = self.high_quality_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            else:
                transformed = self.low_quality_transform(image=image, bboxes=bboxes, class_labels=class_labels)

            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        return image, bboxes, class_labels



class ObjectDetection:
    def __init__(self):
        home_dir = os.path.expanduser('~')
        self.dir = os.path.join(home_dir, "StreetObjectDetection/StreetObjYOLO")


    def train(self, runName, config_file, device="0,1", epoch=1000, batch=48, lr=1e-3,
              model_path=None, fixed_hyperparams=None, resume=False, imgsz=640):
        if model_path is None:
            model_path = os.path.join(self.dir, "weights/yolov10l.pt")
        model = YOLOv10(model_path)
        yaml_path = os.path.join(self.dir, config_file)
        # Albumentations.__init__ == __init__
        if fixed_hyperparams is not None:
            for key, val in fixed_hyperparams.items():
                model.overrides[key] = val
        results = model.train(
            data= yaml_path,         # Path to your dataset config file
            batch = batch,               # Training batch size
            imgsz= imgsz,                   # Input image size
            epochs= epoch,                  # Number of training epochs
            optimizer= 'Adam',             # Optimizer, can be 'Adam', 'SGD', etc.
            lr0= lr,                    # Initial learning rate
            lrf= 0.1,                     # Final learning rate factor
            weight_decay= 0.0005,         # Weight decay for regularization
            momentum= 0.937,              # Momentum (SGD-specific)
            verbose= True,                # Verbose output
            device= device,                  # GPU device index or 'cpu'
            workers= 8,                   # Number of workers for data loading
            project= 'runs/train',        # Output directory for results
            name= runName,                  # Experiment name
            exist_ok= False,              # Overwrite existing project/name directory
            rect= False,                  # Use rectangular training (speed optimization)
            resume= resume,                # Resume training from the last checkpoint
            multi_scale= False,           # Use multi-scale training
            single_cls= False,             # Treat data as single-class
            patience=100,
            augment=True,
        )

    def fine_tune(self, run_name, config_file, device, epoch, batch, model_folder, lr_factor=0.1):
        args_yaml = osp.join(model_folder, 'args.yaml')
        model_path = osp.join(model_folder, 'weights/best.pt')
        with open(args_yaml, 'r') as f:
            hyp = yaml.safe_load(f)
        hyp['lr0'] *= lr_factor
        hyp['data'] = config_file
        hyp['name'] = run_name
        hyp['device'] = device
        hyp['epochs'] = epoch
        hyp['batch'] = batch
        model = YOLOv10(model_path)

        # Train the model using all parameters from args.yaml
        model.train(**hyp)

    def train_from_best(self, runName, config_file, device="0,1", epoch=1000, batch=48, model_path=None,
                        best_hyper=None, fixed_hyperparams=None, imgsz=640, num_workers=8):
        if model_path is None:
            model_path = os.path.join(self.dir, "weights/yolov10l.pt")
        model = YOLOv10(model_path)

        yaml_path = os.path.join(self.dir, config_file)
        with open(yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)

        # Albumentations.__init__ == __init__

        if fixed_hyperparams is not None:
            for key, val in fixed_hyperparams.items():
                model.overrides[key] = val
        if best_hyper is not None:
            with open(best_hyper, 'r') as file:
                hyperparameters = yaml.safe_load(file)
        else:
            hyperparameters = {}
        hyperparameters.update(fixed_hyperparams)

        model.train(
            data=yaml_path,         # Path to your dataset config file
            batch=batch,               # Training batch size
            imgsz=imgsz,                   # Input image size
            epochs=epoch,                  # Number of training epochs
            optimizer='Adam',             # Optimizer, can be 'Adam', 'SGD', etc.
            verbose=True,                # Verbose output
            device=device,                  # GPU device index or 'cpu'
            workers=num_workers,                   # Number of workers for data loading
            project='runs/train',        # Output directory for results
            name=runName,                  # Experiment name
            exist_ok=False,              # Overwrite existing project/name directory
            rect=False,                  # Use rectangular training (speed optimization)
            resume=False,                # Resume training from the last checkpoint
            multi_scale=False,           # Use multi-scale training
            single_cls=False,             # Treat data as single-class
            patience=100,
            augment=True,
            **hyperparameters,
        )

    def train_different_augments(self, runName, config_file, device="0,1", epoch=1000, batch=48, model_path=None,
                        best_hyper=None, fixed_hyperparams=None):
        if model_path is None:
            model_path = os.path.join(self.dir, "weights/yolov10l.pt")

        model = YOLOv10(model_path)
        if fixed_hyperparams is not None:
            for key, val in fixed_hyperparams.items():
                model.overrides[key] = val

        with open(best_hyper, 'r') as file:
            hyperparameters = yaml.safe_load(file)
        print(f"{hyperparameters=}")
        tmp = (dir(model))
        for item in tmp:
            if "optim" in item or 'lr' in item or 'learn' in item:
                print(item)
        print('-------------------------')
        save_dir = osp.join(self.dir, f"runs/train/{runName}")
        os.makedirs(save_dir, exist_ok=True)

        yaml_path = os.path.join(self.dir, config_file)

        train_dataset = CustomDataset(dataset_yaml=yaml_path, mode='train')
        val_dataset = CustomDataset(dataset_yaml=yaml_path, mode='val')

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr0'])
        criterion = nn.MSELoss()  # Replace with appropriate loss function for YOLO
        best_score = float('-inf')  # Track best validation fitness score
        train_losses = []
        val_losses = []

        for epoch_idx in range(epoch):
            model.train()
            epoch_train_loss = 0.0
            for batch_idx, (images, bboxes, labels) in enumerate(train_loader):
                images = torch.stack(images).to(device)
                optimizer.zero_grad()
                loss = model(images, targets={"bboxes": bboxes, "labels": labels})
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation step
            model.eval()
            epoch_val_loss = 0.0
            fitness_score = 0.0  # Custom metric to track best model

            metrics, fitness = model.validate()
            print(f"{metrics=}")
            print(f"{fitness=}")

            print(f"Epoch [{epoch_idx + 1}/{epoch}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Fitness: {fitness_score:.4f}")


        # Save trained model
        save_path = os.path.join(self.dir, "weights", f"{runName}_trained.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


        model.train(
            data=yaml_path,         # Path to your dataset config file
            batch=batch,               # Training batch size
            imgsz=640,                   # Input image size
            epochs=epoch,                  # Number of training epochs
            optimizer='Adam',             # Optimizer, can be 'Adam', 'SGD', etc.
            verbose=True,                # Verbose output
            device=device,                  # GPU device index or 'cpu'
            workers=8,                   # Number of workers for data loading
            project='runs/train',        # Output directory for results
            name=runName,                  # Experiment name
            exist_ok=False,              # Overwrite existing project/name directory
            rect=False,                  # Use rectangular training (speed optimization)
            resume=False,                # Resume training from the last checkpoint
            multi_scale=False,           # Use multi-scale training
            single_cls=False,             # Treat data as single-class
            patience=100,
            augment=True,
            **hyperparameters,
        )

    def tune(self, config_file, custom_space, fixed_hyperparams, epoch=100, batch=32, device='0,1,2', init_best_file=None):
        model = YOLOv10(os.path.join(self.dir, "weights/yolov10l.pt"))
        yaml_path = os.path.join(self.dir,  config_file)

        space = custom_space
        if fixed_hyperparams is not None:
            for key, val in fixed_hyperparams.items():
                # del space[key]
                model.overrides[key] = val
        model.overrides['fliplr'] = 0

        param_to_best = {}
        if init_best_file is not None:
            with open(init_best_file, 'r') as file:
                param_to_best = yaml.safe_load(file)
        else:
            param_to_best = {
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'box': 7.5,
                'cls': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degree': 0,
                'translate': 0.1,
                'scale':0.5,
                'shear': 0,
                'mixup':0,
                'erasing': 0.4,
                'auto_augment': 'randaugment'
            }

        print("searching space--------------------")
        for key, val in space.items():
            print(f"{key}: {val[0]} ~ {val[1]}")
            if key in param_to_best:
                model.overrides[key] = param_to_best[key]
                assert val[0] <= param_to_best[key] <= val[1]
                print(f"Load Best parameter '{key}: {param_to_best[key]}")
            else:
                model.overrides[key] = (val[0] + val[1]) / 3
        print("-----------------------------------\n\n")
        # Albumentations.__init__ == __init__
        model.tune(data=yaml_path,
                   epochs=epoch,
                   iterations=100,
                   optimizer="AdamW",
                   plots=False,
                   save=False,
                   val=True,
                   use_ray=False,
                   batch=batch,
                   device=device,
                   space=space
        )


    def monitor_gpu_usage(self, gap=5*60):
        """Monitor GPU usage in a separate thread."""

        def monitor():
            print("Monitoring GPU usage... Press Ctrl+C to stop.")
            try:
                while True:
                    gpus = GPUtil.getGPUs()
                    print("\n ============ GPU usage ============")
                    for gpu in gpus:
                        print(f"GPU {gpu.id}: {gpu.name}")
                        print(f"  Memory Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                        print(f"  Utilization: {gpu.load * 100:.1f}%")
                    time.sleep(gap)  # Update every 10 seconds
            except KeyboardInterrupt:
                print("GPU monitoring stopped.")

        from threading import Thread
        monitor_thread = Thread(target=monitor, daemon=True)
        monitor_thread.start()
        gc.collect()

if __name__ == "__main__":
    config_file = "config/majority_v1.yaml"

    ObjectDetection.tune_custom_dataset(config_file),


