import torch
import numpy as np
import cv2
from ultralytics import YOLO
from unik3d.models import UniK3D
import os, csv, time
from pathlib import Path
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="YOLO/Color/Manual + UniK3D with simple tracking IDs")
parser.add_argument("--video", "-v", default="videos/Waterloo.mp4",
                    help="Path to input video, default videos/Waterloo.mp4")
parser.add_argument("--conf", "-c", type=float, default=0.2,
                    help="YOLO confidence threshold (0–1), default 0.2")
parser.add_argument("--mode", choices=["yolo", "color"], default="yolo",
                    help="Detection mode: yolo (default) or color (HSV red mask; application for stadium videos)")
parser.add_argument("--every", type=int, default=1, help="Process every Nth frame (sampling)")
parser.add_argument("--start", type=int, default=0, help="First frame to process (inclusive)")
parser.add_argument("--end", type=int, default=-1, help="Last frame index (exclusive, -1 = to the end)")
parser.add_argument("--central_crop", type=float, default=0.0,
                    help="Optional central crop ratio (0..0.45) for depth inside bbox (robustness)")
parser.add_argument("--max_age", type=int, default=30,
                    help="Frames to keep a track alive without a matching detection")
parser.add_argument("--iou_match", type=float, default=0.3,
                    help="IoU threshold for detection-to-track assignment")

args = parser.parse_args()

# Variables
video_path = args.video
conf = args.conf
mode = args.mode
every = max(1, int(args.every))
f_start, f_end = int(args.start), int(args.end)
CENTRAL_CROP_RATIO = float(args.central_crop)
if CENTRAL_CROP_RATIO < 0.0 or CENTRAL_CROP_RATIO >= 0.5:
    CENTRAL_CROP_RATIO = 0.0
MAX_AGE = int(args.max_age)
IOU_THRESH = float(args.iou_match)

# Load models
model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")  # large model
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
if not src_fps or src_fps <= 1:   # fallback if FPS is missing/invalid
    src_fps = 30.0
print(f"[info] source FPS detected: {src_fps:.2f}")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output folder
os.makedirs("output", exist_ok=True)

# Output writer follows source FPS
output_path = f"output/{Path(video_path).stem}_{int(src_fps)}fps_{mode}_ids.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

# === CSV outputs: put under output/timings and include video basename ===
stats_dir = os.path.join(os.path.dirname(output_path), "stats")
os.makedirs(stats_dir, exist_ok=True)
video_base = os.path.splitext(os.path.basename(output_path))[0]
timing_csv_path = os.path.join(stats_dir, f"timings_{video_base}.csv")
depth_csv_path  = os.path.join(stats_dir, f"depths_{video_base}.csv")
stats_csv_path  = os.path.join(stats_dir, f"depth_stats_by_id_{video_base}.csv")

# Timing buffers
yolo_times_ms, unik3d_times_ms, total_times_ms = [], [], []
csv_rows = []
depth_rows = []           # per-frame rows to write once at the end
object_depths = []        # for overall summary statistics

# Tracker state (very light IoU-based)
next_id = 1
tracks = {}  # id -> dict(bbox, label, last_frame)
_tracker = None  # for manual ROI

def _resolve_label_from_yolo(yolo_names, cls_id):
    if hasattr(yolo_names, "get"):
        return yolo_names.get(cls_id, str(cls_id))
    try:
        return yolo_names[cls_id]
    except Exception:
        return str(cls_id)

def _median_depth_in_bbox(depth_map, x1, y1, x2, y2, central_ratio=0.0):
    H, W = depth_map.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    if central_ratio > 0.0:
        rx = int((x2 - x1) * central_ratio)
        ry = int((y2 - y1) * central_ratio)
        cx1, cy1 = x1 + rx, y1 + ry
        cx2, cy2 = x2 - rx, y2 - ry
        if cx2 > cx1 and cy2 > cy1:
            x1, y1, x2, y2 = cx1, cy1, cx2, cy2

    crop = depth_map[y1:y2, x1:x2]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))

def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def _assign_ids(dets, frame_idx):
    """
    dets: list of dict {bbox:(x1,y1,x2,y2), label:str, conf:float}
    returns: list of dict {id:int, bbox:..., label:..., conf:...}
    """
    global next_id, tracks
    assigned = []

    # Build list of live tracks
    live_ids = [tid for tid, t in tracks.items() if frame_idx - t["last_frame"] <= MAX_AGE]

    used_tracks = set()

    for d in dets:
        db = d["bbox"]
        # find best matching live track by IoU
        best_iou, best_tid = 0.0, None
        for tid in live_ids:
            if tid in used_tracks:  # one detection per track
                continue
            tb = tracks[tid]["bbox"]
            iou = _iou(db, tb)
            if iou > best_iou:
                best_iou, best_tid = iou, tid
        if best_tid is not None and best_iou >= IOU_THRESH:
            # update track
            tracks[best_tid]["bbox"] = db
            tracks[best_tid]["label"] = d["label"]
            tracks[best_tid]["last_frame"] = frame_idx
            assigned.append({"id": best_tid, **d})
            used_tracks.add(best_tid)
        else:
            # create new track
            tid = next_id; next_id += 1
            tracks[tid] = {"bbox": db, "label": d["label"], "last_frame": frame_idx}
            assigned.append({"id": tid, **d})
            used_tracks.add(tid)

    # Optionally, we could prune dead tracks here (not necessary for logging)
    return assigned

# Jump to start
cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_start))
frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

while True:
    if f_end > 0 and frame_idx >= f_end:
        break

    # sampling: grab frames we don't want to process
    if (frame_idx - f_start) % every != 0:
        ok = cap.grab()
        if not ok:
            break
        frame_idx += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    dets = []  # list of {bbox:(x1,y1,x2,y2), label:str, conf:float}
    yolo_ms = 0.0

    if mode == "yolo":
        yres = yolo_model(frame, conf=conf, verbose=False,
                          device=0 if device.type == "cuda" else "cpu")[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        yolo_ms = (t1 - t0) * 1000.0

        if yres.boxes is not None and len(yres.boxes) > 0:
            xyxy = yres.boxes.xyxy.cpu().numpy()
            confs = yres.boxes.conf.cpu().numpy()
            clses = yres.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), cf, ci in zip(xyxy, confs, clses):
                label = _resolve_label_from_yolo(yolo_model.names, int(ci))
                dets.append({"bbox": (int(x1),int(y1),int(x2),int(y2)), "label": label, "conf": float(cf)})
    elif mode == "color":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1, upper1 = (0, 70, 60), (10, 255, 255)
        lower2, upper2 = (170, 70, 60), (180, 255, 255)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.medianBlur(mask, 5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 500:
                dets.append({"bbox": (x, y, x + w, y + h), "label": "red_object", "conf": 1.0})
        t1 = time.perf_counter()
    else:
        t1 = time.perf_counter()

    unik_ms = None
    t_end = t1

    if len(dets) == 0:
        # no detections this frame
        out.write(frame)
        total_ms = (t_end - t0) * 1000.0
        yolo_times_ms.append(yolo_ms)
        total_times_ms.append(total_ms)
        csv_rows.append([frame_idx, f"{yolo_ms:.3f}", "", f"{total_ms:.3f}"])
        frame_idx += 1
        continue

    # Assign IDs to detections
    assigned = _assign_ids(dets, frame_idx)

    # UniK3D depth estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_tensor = (torch.from_numpy(frame_rgb)
                  .to(device)
                  .permute(2, 0, 1).unsqueeze(0).float()
                  .to(memory_format=torch.channels_last))

    with torch.no_grad():
        torch.cuda.synchronize()
        t2a = time.perf_counter()

        pred = model.infer(rgb_tensor)

        torch.cuda.synchronize()
        t2b = time.perf_counter()
        unik_ms = (t2b - t1) * 1000.0
        t_end = t2b

        depth_map = pred["depth"].squeeze()
        depth_map = torch.nn.functional.interpolate(
            depth_map[None, None], size=(height, width), mode="nearest"
        ).squeeze().float().cpu().numpy()

    # For each assigned detection (with ID), compute depth, draw, and log
    for item in assigned:
        tid = item["id"]
        (x1,y1,x2,y2) = item["bbox"]
        label = item["label"]
        confv = item["conf"]

        best_depth = _median_depth_in_bbox(depth_map, x1, y1, x2, y2, CENTRAL_CROP_RATIO)
        if best_depth is not None:
            object_depths.append(best_depth)

        # Draw
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        txt = f"ID {tid} | {label} {confv:.2f} | {best_depth:.1f}m" if best_depth is not None else f"ID {tid} | {label} {confv:.2f} | n/a"
        cv2.putText(frame, txt, (int(x1), max(15, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Per-frame CSV row (one row per detection)
        depth_rows.append([
            frame_idx, tid, label, int(x1), int(y1), int(x2), int(y2), "" if best_depth is None else f"{best_depth:.4f}", f"{confv:.3f}"
        ])

    out.write(frame)

    # timings
    total_ms = (t_end - t0) * 1000.0
    yolo_times_ms.append(yolo_ms)
    if unik_ms is not None:
        unik3d_times_ms.append(unik_ms)
    total_times_ms.append(total_ms)
    csv_rows.append([
        frame_idx,
        f"{yolo_ms:.3f}",
        f"{unik_ms:.3f}" if unik_ms is not None else "",
        f"{total_ms:.3f}"
    ])

    frame_idx += 1

cap.release()
out.release()

try:
    cv2.destroyAllWindows()
except cv2.error as e:
    print(f"[warn] cv2.destroyAllWindows() skipped: {e}")

# Write CSV once at the end
with open(timing_csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_idx", "yolo_ms", "unik3d_ms", "total_ms"])
    w.writerows(csv_rows)

with open(depth_csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_idx", "id", "class_name", "x1", "y1", "x2", "y2", "depth_m", "conf"])
    w.writerows(depth_rows)

# Print & save per-ID stats for convenience
from collections import defaultdict
by_id = defaultdict(list)
for r in depth_rows:
    # r: [frame_idx, id, label, x1, y1, x2, y2, depth, conf]
    if r[7] != "":
        by_id[r[1]].append(float(r[7]))

with open(stats_csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "class_name_example", "frames_with_depth", "depth_avg_m", "depth_median_m", "depth_std_m", "depth_p95_m"])
    for tid, vals in by_id.items():
        arr = np.array(vals, dtype=np.float64)
        avg = float(arr.mean())
        med = float(np.median(arr))
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        p95 = float(np.percentile(arr, 95))
        # fetch a label example from rows
        label_example = next((r[2] for r in depth_rows if r[1] == tid), "")
        w.writerow([tid, label_example, len(vals), f"{avg:.4f}", f"{med:.4f}", f"{std:.4f}", f"{p95:.4f}"])

def _avg(xs):
    return (sum(xs) / len(xs)) if xs else 0.0

avg_yolo   = _avg(yolo_times_ms)
avg_unik3d = _avg(unik3d_times_ms)
avg_total  = _avg(total_times_ms)

print("\n=== Inference timing (ms) ===")
print(f"Frames processed: {len(total_times_ms)}")
print(f"YOLO avg   : {avg_yolo:.3f} ms")
print(f"UniK3D avg : {avg_unik3d:.3f} ms" if unik3d_times_ms else "UniK3D avg : n/a (never ran)")
print(f"Total avg  : {avg_total:.3f} ms")
print(f"Per-frame timings saved to: {timing_csv_path}")
print(f"Video saved to {output_path}")

# --- Overall distance summary (all IDs together)
valid_depths = [d for d in object_depths if d is not None]
print("\n=== Object distance (meters) ===")
if valid_depths:
    arr = np.array(valid_depths, dtype=np.float64)
    avg_depth = float(arr.mean())
    med_depth = float(np.median(arr))
    std_depth = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    p95_depth = float(np.percentile(arr, 95))
    print(f"Frames with depth: {len(valid_depths)}")
    print(f"Average: {avg_depth:.3f} m   Median: {med_depth:.3f} m   Std: {std_depth:.3f} m   P95: {p95_depth:.3f} m")
    print(f"Per-frame depths saved to: {depth_csv_path}")
    print(f"Per-ID stats saved to: {stats_csv_path}")
else:
    print("No valid depth values were computed.")
