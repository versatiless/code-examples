import json
import os
import cv2
import numpy as np
import pandas as pd
import torch
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.byte_tracker import STrack
from tqdm import tqdm
import time
import warnings
from ultralytics import YOLOv10
import os.path as osp
import matplotlib.pyplot as plt
from glob import glob
import time
import datetime

warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)


class ByteTrackArgs:
    track_thresh = 0.4
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = -1
    min_box_area = 10
    mot20 = False


def ask_user_for_rotation(frame, save_path):
    cv2.imwrite(save_path, frame)
    print(f"Image saved at {save_path}. Open it manually and check.")
    key = input("Press 'r' to rotate, any other key to continue: ").strip().lower()
    return key == 'r'

def run_batch_od_inference(batch_frames, type_to_model, type_to_imgsz, type_to_old_to_new_idx):
    detections_batch = [[] for _ in range(len(batch_frames))]  # list of detections per frame

    for tp, model in type_to_model.items():
        imgsz = type_to_imgsz[tp]
        old_to_new_idx = type_to_old_to_new_idx[tp]
        device = model.device

        # Preprocess all frames
        frames_resized = [cv2.resize(f, (imgsz, imgsz)) for f in batch_frames]
        frames_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames_resized
        ])  # (B, 3, H, W)
        frames_tensor = frames_tensor.to(device)

        # Run batch inference
        with torch.cuda.device(device):
            results = model(frames_tensor, verbose=False, imgsz=imgsz)

        for bidx, result in enumerate(results):
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                assert max(x2, y2) > 1

                new_cls = old_to_new_idx[int(cls)]
                detections_batch[bidx].append([x1/imgsz, y1/imgsz, x2/imgsz, y2/imgsz, conf, new_cls])

    return detections_batch, device



def load_models(type_to_model_path):
    names = []
    type_to_model = {}
    type_to_imgsz = {}
    type_to_old_to_new_idx = {}

    for tp, model_path in type_to_model_path.items():
        model = YOLOv10(model_path).to('cuda')

        classes = model.names
        imgsz = model.overrides.get("imgsz", 640)  # Default to 640 if not available
        # print(f"Using imgsz={imgsz} for model {tp}")
        old_to_new_idx = {}
        for old_idx in range(len(classes)):
            cls = classes[old_idx]
            old_to_new_idx[old_idx] = len(names)
            names.append(cls)
        type_to_old_to_new_idx[tp] = old_to_new_idx

        type_to_model[tp] = model
        type_to_imgsz[tp] = imgsz
    return type_to_model, type_to_imgsz, type_to_old_to_new_idx, names

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def run_detection_tracker(type_to_model, type_to_imgsz, type_to_old_to_new_idx, names, video_path, save_file, batch_size=16):
    first_det_frame = None

    basename = osp.basename(video_path).split(".")[0]
    with open(f"{basename}_obj.names", 'w') as obj_fout:
        for idx, cls in enumerate(names):
            obj_fout.write(cls + '\n')
            # print(idx,":", cls)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, first_frame = cap.read()
    cap.release()
    # print('load first frame')

    save_folder = "tmp"
    os.makedirs(save_folder, exist_ok=True)
    level1_folder = video_path.split("/")[-2]
    if level1_folder not in folder_to_rotate:
        rotate_video = True
    else:
        rotate_video = folder_to_rotate[level1_folder]

    save_img = osp.join(save_folder, f"{level1_folder}_{osp.basename(video_path).split('.')[0]}_first_r18{rotate_video}.jpg")
    cv2.imwrite(save_img, first_frame)

    # ask_user_for_rotation(first_frame, save_img)

    cap = cv2.VideoCapture(video_path)

    STrack.track_id_count = 0
    class_to_tracker = {cls_id: BYTETracker(ByteTrackArgs()) for cls_id in range(len(names))}
    class_to_track_offset = {cls_id: cls_id * 100000 for cls_id in range(len(names))}

    records = []
    batch_frames = []
    batch_indices = []
    fid_to_detections = {}

    progress_checkpoint = max(1, total_frames // 50)
    last_reported = -1
    device = None
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if rotate_video:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        batch_frames.append(frame)
        batch_indices.append(frame_idx)

        if len(batch_frames) == batch_size or frame_idx == total_frames - 1:
            detections_batch, device = run_batch_od_inference(batch_frames, type_to_model, type_to_imgsz, type_to_old_to_new_idx)

            for i, detections in enumerate(detections_batch):
                fid = batch_indices[i]
                frame = batch_frames[i]
                fid_to_detections[fid] = [list(map(float, d)) for d in detections]

                for cls_id in range(len(names)):
                    cls_dets = [d for d in detections if int(d[5]) == cls_id]
                    det_tensor = torch.from_numpy(np.array(cls_dets, dtype=np.float32)) if cls_dets else torch.empty((0, 6), dtype=torch.float32)
                    targets = class_to_tracker[cls_id].update(det_tensor, frame.shape[:2], frame.shape[:2])
                    for t in targets:
                        if not t.is_activated:
                            continue
                        x1, y1, w, h = t.tlwh
                        x2, y2 = x1 + w, y1 + h
                        conf = t.score
                        global_id = class_to_track_offset[cls_id] + t.track_id
                        cur_line = f"{fid},{global_id},{x1},{y1},{x2},{y2},{conf},{cls_id},-1,-1\n"
                        records.append(cur_line)
            batch_frames.clear()
            batch_indices.clear()

        if frame_idx // progress_checkpoint > last_reported:
            last_reported = frame_idx // progress_checkpoint
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{now}][PID {os.getpid()}][{device}] {basename}: {frame_idx}/{total_frames} ({last_reported}%)", flush=True)

    det_file = save_file.replace("_MOT.txt", "_DET.json")
    with open(det_file, 'w') as file:
        json.dump(fid_to_detections, file)

    cap.release()

    with open(save_file, 'w') as fout:
        for rec in records:
            fout.write(rec)

folder_to_rotate = {
    "11242024": False,
    '10132024': True,
    "10202024":True,
    "1026202410272024": True,
    '1028202411022024': True,
    "1105202411092024": True,
    '1119202411211024': True,
}

def main():
    home_path = osp.expanduser("~")
    project_path = osp.join(home_path, "StreetObjectDetection/StreetObjYOLO")
    video_root_folder = "/mnt/d/202*Video/Tacoma/Right/*/"
    cdds = glob(osp.join(video_root_folder, "*.MP4"))
    print(f"[INFO] Total videos: {len(cdds)}")

    type_to_model_path = {"street_obj": osp.join(project_path, f"runs/train/objV3_augV2_1280/weights/best.pt"),
                          "road_marking": osp.join(project_path, f"runs/train/rm_selectedp2_finalv1/weights/best.pt")}
    type_to_model, type_to_imgsz, type_to_old_to_new_idx, names = load_models(type_to_model_path)

    for video_path in cdds:
        video_basename = osp.basename(video_path).replace(".MP4", "")
        os.makedirs("mot_v2", exist_ok=True)
        save_file = f"mot_v2/{video_basename}_MOT.txt"
        if not osp.exists(save_file):
            s1 = time.time()
            run_detection_tracker(type_to_model, type_to_imgsz, type_to_old_to_new_idx, names, video_path, save_file)
            duration = round(time.time() - s1, 2)
            print(f"[FINISHED]: {save_file=} IN {duration}s ")
        else:
            print(f"[SKIPPED]: Already exist '{save_file}'")



if __name__ == "__main__":
    main()
