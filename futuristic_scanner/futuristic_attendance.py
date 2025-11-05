#!/usr/bin/env python3
"""
futuristic_attendance.py
- YOLOv8-face (preferred) or OpenCV DNN fallback
- Multi-image per-student training (folder per student)
- IoU tracker so identity persists through occlusion (hands on face)
- Auto-learn: auto-save high-quality samples for known & unknown
- Auto + Manual zoom, HUD, scanning line, attendance CSV, voice
"""

import os, sys, time, csv
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import face_recognition
import pyttsx3

# ---------- CONFIG ----------
KNOWN_DIR = "known_faces"
MODELS_DIR = "models"
YOLO_WEIGHTS = os.path.join(MODELS_DIR, "yolov8m-face.pt") 
DNN_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
DNN_WEIGHTS = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

IP_CAM = "http://192.168.1.15:8080/video"
ATTENDANCE_CSV = "attendance.csv"

DETECT_EVERY = 3          
MIN_CONF = 0.5            
IOU_THRESH = 0.35
HOLD_NAME_SEC = 2.0      
AUTO_LEARN_SAVE_EVERY = 5 
UNKNOWN_SAVE_DIR = "unknown_samples"
UNKNOWN_SAVE_THRESHOLD = 6  
MAX_MISSES = 12           
BOX_MARGIN = 0.18        
TOLERANCE = 0.48          

MIN_ZOOM = 1.0
MAX_ZOOM = 6.0
ZOOM_SPEED = 0.06
AUTO_ZOOM = True

# ------------------------------
os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- voice ----------
tts = pyttsx3.init()
tts.setProperty("rate", 170)

# ---------- load known faces (folder per person; multiple images) ----------
known_encodings = []
known_names = []

print("[INFO] Loading known faces from", KNOWN_DIR)
if os.path.isdir(KNOWN_DIR):
    for person in sorted(os.listdir(KNOWN_DIR)):
        person_folder = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(person_folder):
            continue
        for fname in os.listdir(person_folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(person_folder, fname)
            try:
                img = face_recognition.load_image_file(p)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(person)
            except Exception as e:
                print("  ! error loading", p, e)
print(f"[INFO] {len(known_encodings)} encodings loaded for {len(set(known_names))} people")

# ---------- attendance CSV ----------
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time", "Date"])
attendance_seen = set()

# ---------- choose detector ----------
use_yolo = False
yolo_model = None
net = None
try:
    if os.path.exists(YOLO_WEIGHTS):
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_WEIGHTS)
        use_yolo = True
        print("[INFO] Using YOLOv8-face:", YOLO_WEIGHTS)
    else:
        raise Exception("YOLO weights not found")
except Exception as e:
    print("[INFO] YOLO not used:", e)
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_WEIGHTS):
        net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_WEIGHTS)
        print("[INFO] Using OpenCV DNN face detector")
    else:
        print("[ERROR] No detector found. Put YOLO weights or DNN model files into models/.")
        sys.exit(1)

# ---------- video capture (threadless/simple) ----------
cap = cv2.VideoCapture(IP_CAM)
if not cap.isOpened():
    print("[ERROR] Cannot open camera:", IP_CAM)
    sys.exit(1)

# set moderate resolution for good zoom; change if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------- tracking data ----------
tracks = {}  # track_id -> {box, name, last_seen_time, misses, seen_count, samples deque, last_name_time}
next_track_id = 0

def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1]); xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1, (a[2]-a[0])) * max(1, (a[3]-a[1]))
    areaB = max(1, (b[2]-b[0])) * max(1, (b[3]-b[1]))
    denom = float(areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 1.0
    return inter / denom

def expand_box(box, frac, frame_w, frame_h):
    x1,y1,x2,y2 = box
    w = x2 - x1; h = y2 - y1
    pad_w = int(w * frac); pad_h = int(h * frac)
    nx1 = max(0, x1 - pad_w); ny1 = max(0, y1 - pad_h)
    nx2 = min(frame_w, x2 + pad_w); ny2 = min(frame_h, y2 + pad_h)
    return (nx1, ny1, nx2, ny2)

def save_unknown_sample(sample_img, tid):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(UNKNOWN_SAVE_DIR, f"unknown_{tid}_{ts}.jpg")
    cv2.imwrite(path, sample_img)
    print("[AUTO-SAVE] Unknown saved:", path)

def add_sample_to_person(name, sample_img):
    folder = os.path.join(KNOWN_DIR, name)
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{name}_{ts}.jpg")
    cv2.imwrite(path, sample_img)
    # compute encoding and add
    try:
        new_enc = face_recognition.face_encodings(face_recognition.load_image_file(path))[0]
        known_encodings.append(new_enc)
        known_names.append(name)
        print("[AUTO-LEARN] Added sample to", name)
    except Exception as e:
        print("[AUTO-LEARN] Encoding failed for", path, e)

# ---------- main loop ----------
frame_idx = 0
zoom_level = 1.0
auto_zoom = AUTO_ZOOM
scan_y = 0
scan_dir = 6

print("[INFO] Starting. Press Q to quit. Z/X manual zoom. A toggle auto-zoom.")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue
    frame_idx += 1
    H, W = frame.shape[:2]

    # apply zoom crop (centered)
    cx, cy = W//2, H//2
    zw = int(W / zoom_level); zh = int(H / zoom_level)
    x1 = max(0, cx - zw//2); y1 = max(0, cy - zh//2)
    x2 = min(W, cx + zw//2); y2 = min(H, cy + zh//2)
    frame_crop = frame[y1:y2, x1:x2]
    frame_disp = cv2.resize(frame_crop, (W, H))

    # lighten slightly for poor lighting
    frame_disp = cv2.convertScaleAbs(frame_disp, alpha=1.1, beta=10)

    detections = []  # (x1,y1,x2,y2,score) in display coordinates

    # detection stage
    if use_yolo:
        # ultralytics YOLO returns result object(s)
        results = yolo_model(frame_disp, verbose=False)
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                xy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else [1.0]*len(xy)
                for i, b in enumerate(xy):
                    bx1, by1, bx2, by2 = [int(v) for v in b[:4]]
                    sc = float(confs[i]) if i < len(confs) else 1.0
                    if sc >= MIN_CONF:
                        detections.append((bx1, by1, bx2, by2, sc))
    else:
        # use DNN on every Nth frame
        if frame_idx % DETECT_EVERY == 0:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame_disp, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
            net.setInput(blob)
            dets = net.forward()
            for i in range(dets.shape[2]):
                conf = float(dets[0,0,i,2])
                if conf < MIN_CONF:
                    continue
                box = dets[0,0,i,3:7] * np.array([W,H,W,H])
                xA,yA,xB,yB = box.astype("int")
                xA,yA = max(0,xA), max(0,yA)
                xB,yB = min(W,xB), min(H,yB)
                if xB-xA < 20 or yB-yA < 20:
                    continue
                detections.append((xA,yA,xB,yB,conf))

    # -------- match detections to existing tracks by IoU ----------
    used_det = set()
    if detections and tracks:
        tids = list(tracks.keys())
        iou_mat = np.zeros((len(tids), len(detections)), dtype=float)
        for ti, tid in enumerate(tids):
            tb = tracks[tid]['box']
            for di, det in enumerate(detections):
                iou_mat[ti, di] = iou(tb, det[:4])
        # greedy match
        while True:
            idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            ti, di = idx
            if iou_mat[ti, di] < IOU_THRESH:
                break
            tid = tids[ti]
            tracks[tid]['box'] = detections[di][:4]
            tracks[tid]['last_seen'] = time.time()
            tracks[tid]['misses'] = 0
            tracks[tid]['seen_count'] += 1
            # append sample crop
            bx1,by1,bx2,by2 = [int(v) for v in detections[di][:4]]
            crop = frame_disp[by1:by2, bx1:bx2].copy()
            if crop.size:
                tracks[tid]['samples'].append(crop)
            used_det.add(di)
            iou_mat[ti,:] = -1
            iou_mat[:,di] = -1

    # add unmatched detections as new tracks
    for di, det in enumerate(detections):
        if di in used_det:
            continue
        bx1,by1,bx2,by2,_ = det
        tid = next_track_id
        next_track_id += 1
        tracks[tid] = {
            'box': (int(bx1),int(by1),int(bx2),int(by2)),
            'name': None,
            'last_seen': time.time(),
            'misses': 0,
            'seen_count': 1,
            'samples': deque(maxlen=16),
            'last_name_time': 0.0
        }
        crop = frame_disp[by1:by2, bx1:bx2].copy()
        if crop.size:
            tracks[tid]['samples'].append(crop)

    # increase misses for non-detected tracks
    if not detections:
        for tid in list(tracks.keys()):
            tracks[tid]['misses'] += 1
            if tracks[tid]['misses'] > MAX_MISSES:
                del tracks[tid]

    # ---------- recognition (heavy) done every N frames ----------
    if frame_idx % DETECT_EVERY == 0:
        for tid, info in list(tracks.items()):
            bx1,by1,bx2,by2 = [int(v) for v in info['box']]
            # expand box a bit to capture ears/chin/top
            ex1, ey1, ex2, ey2 = expand_box((bx1,by1,bx2,by2), BOX_MARGIN, W, H)
            # clamp again
            ex1, ey1, ex2, ey2 = max(0,ex1), max(0,ey1), min(W,ex2), min(H,ey2)
            if ex2-ex1 < 20 or ey2-ey1 < 20:
                continue
            crop = frame_disp[ey1:ey2, ex1:ex2].copy()
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = []
            try:
                encs = face_recognition.face_encodings(rgb)
            except Exception:
                encs = []
            name_this = None
            if encs:
                enc = encs[0]
                if known_encodings:
                    dists = face_recognition.face_distance(known_encodings, enc)
                    best = int(np.argmin(dists))
                    best_dist = float(dists[best])
                    if best_dist < TOLERANCE:
                        name_this = known_names[best]
            # update track
            if name_this:
                prev = info.get('name', None)
                info['name'] = name_this
                info['last_name_time'] = time.time()
                # auto-add sample occasionally to person folder
                if info.get('seen_count',0) % AUTO_LEARN_SAVE_EVERY == 0:
                    add_sample_to_person(name_this, crop)
                # mark attendance once
                if name_this not in attendance_seen:
                    attendance_seen.add(name_this)
                    now = datetime.now()
                    with open(ATTENDANCE_CSV, "a", newline="") as f:
                        csv.writer(f).writerow([name_this, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")])
                    tts.say(f"{name_this} present")
                    tts.runAndWait()
            else:
                # if we had a recent name, hold it (occlusion tolerance)
                if info.get('name', None) and (time.time() - info.get('last_name_time',0)) < HOLD_NAME_SEC:
                    pass
                else:
                    info['name'] = None

    # ---------- handle unknown auto-save ----------
    for tid, info in list(tracks.items()):
        if info.get('name', None) is None and info['seen_count'] >= UNKNOWN_SAVE_THRESHOLD:
            sample = info['samples'][-1] if info['samples'] else None
            if sample is not None:
                save_unknown_sample(sample, tid)
            tracks[tid]['seen_count'] = 0

    # ---------- auto-zoom based on max face width ----------
    if auto_zoom and detections:
        max_w = max([det[2]-det[0] for det in detections])
        # choose target inversely proportional to face width (empirical)
        target = np.clip(700.0 / max_w, MIN_ZOOM, MAX_ZOOM)
        zoom_level += (target - zoom_level) * ZOOM_SPEED

    # ---------- display & HUD ----------
    disp = frame_disp.copy()
    # scanning line
    sy = int((frame_idx * scan_dir) % H)
    cv2.line(disp, (0, sy), (W, sy), (0,255,0), 2)

    # draw tracks with glow
    for tid, info in tracks.items():
        x1,y1,x2,y2 = [int(v) for v in info['box']]
        name = info.get('name', None)
        color = (0,255,0) if name else (0,0,255)
        # glow border
        for i in range(5,0,-1):
            cv2.rectangle(disp, (x1-i, y1-i), (x2+i, y2+i), (0, 255//i, 0), 1)
        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
        label = name if name else f"Unknown#{tid}"
        cv2.putText(disp, label, (x1, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

    # HUD
    hud = disp.copy()
    cv2.rectangle(hud, (10,10), (360, 158), (0,255,0), -1)
    disp = cv2.addWeighted(hud, 0.17, disp, 0.83, 0)
    now = datetime.now()
    lines = [
        f"Zoom: {zoom_level:.2f}x",
        f"Tracks: {len(tracks)}",
        f"Known present: {len(attendance_seen)}",
        f"Auto-zoom: {'ON' if auto_zoom else 'OFF'}",
        now.strftime("%H:%M:%S %d-%b-%Y")
    ]
    yy = 38
    for L in lines:
        cv2.putText(disp, L, (20, yy), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 1)
        yy += 26

    cv2.imshow("Futuristic Attendance (accurate v3)", disp)

    # keys
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('z'):
        zoom_level = min(zoom_level + 0.2, MAX_ZOOM)
        auto_zoom = False
    elif k == ord('x'):
        zoom_level = max(zoom_level - 0.2, MIN_ZOOM)
        auto_zoom = False
    elif k == ord('a'):
        auto_zoom = not auto_zoom

cap.release()
cv2.destroyAllWindows()
print("[INFO] stopped")











