import torch
import numpy as np
import cv2
from ultralytics import YOLO
from unik3d.models import UniK3D  
import os
from pathlib import Path
import argparse
import time

parser = argparse.ArgumentParser(description="YOLO + UniK3D on video")
parser.add_argument("--video", "-v", default="videos/Waterloo.mp4", help="Path to input video, default videos/Waterloo.mp4")
parser.add_argument("--fps", "-f", type=float, default=1.0, help="Output FPS to process, default 1.0")
parser.add_argument("--conf", "-c", type=float, default=0.2, help="YOLO confidence threshold (0–1), default 0.2")
args = parser.parse_args()

# Variables
video_path = args.video
output_fps = args.fps
conf = args.conf

# Load models
model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl") #large model
yolo_model = YOLO("yolo_models/yolo11n-uav-vehicle-bbox.pt")

# Jetson-friendly settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
try:
    yolo_model.to(device)
except Exception:
    pass
try:
    model.resolution_level = 5
except Exception:
    pass
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

src_fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output folder
os.makedirs("output", exist_ok=True)

# Output at desired FPS
output_path = f"output/{Path(video_path).stem}_{int(output_fps)}fps.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

# Process at desired FPS
step = max(1, int(round(src_fps / max(1e-6, output_fps))))
frame_idx = 0

while True:
    start_time = time.time()
    if frame_idx % step != 0:
        ok = cap.grab()
        if not ok:
            break
        frame_idx += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    short_side = min(height, width) if min(height, width) > 0 else 640
    imgsz = min(640, short_side) 
    yres = yolo_model(frame,conf=conf,verbose=False,device=0 if device.type == "cuda" else "cpu")[0] # Adding in imgsz=imgsz significantly reduces accuracy

    boxes = yres.boxes
    if boxes is None or len(boxes) == 0:
        out.write(frame)
        frame_idx += 1
        continue

    # UniK3D depth estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_tensor = (torch.from_numpy(frame_rgb)
                  .to(device)  
                  .permute(2, 0, 1).unsqueeze(0).float()
                  .to(memory_format=torch.channels_last))

    with torch.no_grad():
        pred = model.infer(rgb_tensor)
        depth_map = pred["depth"].squeeze()
        depth_map = torch.nn.functional.interpolate(
            depth_map[None, None], size=(height, width), mode="nearest"
        ).squeeze().float().cpu().numpy()

    H, W = depth_map.shape
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        det_conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        label = yolo_model.names[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        depth_crop = depth_map[y1:y2, x1:x2]
        valid = depth_crop[np.isfinite(depth_crop) & (depth_crop > 0)]
        label_text = f"{label} {det_conf:.2f}, {np.median(valid):.1f}m" if valid.size else f"{label} {det_conf:.2f}, n/a"
        cv2.putText(frame, label_text, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_idx += 1
    elapsed_time = time.time() - start_time
    print(f"Processed frame {frame_idx} at {elapsed_time:.2f} seconds")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved to {output_path}")