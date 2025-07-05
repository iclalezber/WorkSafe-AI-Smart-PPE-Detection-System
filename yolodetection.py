from collections import Counter, defaultdict
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import subprocess
import time
import cv2
import os

# ====== 🧠 MODEL & CONFIGURATION ======

# Load the trained YOLOv8 model
model = YOLO("yolov8-model/best.pt")

# File path for detection logs
LOG_PATH = "results/log.txt"

# List of class names used in detection
classNames = [
    'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask', 'foot',
    'tool', 'glasses', 'gloves', 'helmet', 'hands', 'head',
    'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
]

# Unique BGR colors to each class
classColors = {
    'person': (255, 0, 0),          # Dark Blue
    'ear': (200, 200, 200),         # Light Gray
    'ear-mufs': (255, 150, 0),
    'face': (255, 0, 255),          # Pink
    'face-guard': (255, 255, 0),    
    'face-mask': (0, 165, 255),     # Orange
    'foot': (150, 255, 0),
    'tool': (100, 150, 255),        # Salmon
    'glasses': (255, 200, 0),       # Cyan
    'gloves': (255, 20, 147),       # Purple
    'helmet': (0, 255, 0),          # Green
    'hands': (144, 238, 144),       # Light Green
    'head': (0, 0, 255),            # Red
    'medical-suit': (0, 200, 255),  # Gold
    'shoes': (80, 180, 255),        # Orange
    'safety-suit': (0, 100, 255),
    'safety-vest': (0, 255, 255)    # Yellow
}

# Required PPE classes per environment
REQUIRED_PPE_SETS = {
    'Worksite': {'helmet', 'safety-vest', 'gloves'},
    'Laboratory': {'glasses', 'face-mask', 'medical-suit', 'gloves'}
}

# PPE weights for environment prediction
PPE_WEIGHTS = {
    'helmet': 5,
    'medical-suit': 5,
    'face-mask': 4,
    'safety-vest': 4,
    'glasses': 3,
    'gloves': 1,
}


# ====== 🛡️ ENVIRONMENT & SAFETY ANALYSIS ======

# Predicts the environment from detected PPE classes
def predict_environment(detected_classes):
    detected_set = set(detected_classes)
    best_env = None
    best_match_count = -1
    best_score = -1

    for env, required_ppe in REQUIRED_PPE_SETS.items():
        matched = detected_set & required_ppe
        match_count = len(matched)
        score = sum(PPE_WEIGHTS.get(ppe, 1) for ppe in matched)

        if match_count > best_match_count or (match_count == best_match_count and score > best_score):
            best_env = env
            best_match_count = match_count
            best_score = score

    return best_env


# Calculates the safety score based on missing PPE
def calculate_safety_score(detected, predicted_env):
    present = set(detected)
    required_ppe = REQUIRED_PPE_SETS.get(predicted_env, set())
    matched = present & required_ppe

    if not required_ppe:
        return 100.0, set()
    score = len(matched) / len(required_ppe)

    return round(score * 100, 2), required_ppe - present


# ====== 🎯 MAIN DETECTIONS ======

# Detects PPE in a single webcam or video frame
def process_frame(frame, source_type="VIDEO", filename="N/A"):
    results = model(frame, stream=True)
    overlay = frame.copy()
    detected_classes = []
    confidences = {}

    if source_type == "WEBCAM":
        box_thickness = 3
        font_scale = 0.6
        font_thickness = 1
        rect_padding = 10
    else:
        box_thickness = 15
        font_scale = 1.5
        font_thickness = 3
        rect_padding = 15

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            class_name = classNames[cls_id] if cls_id < len(classNames) else f"Class_{cls_id}"
            color = classColors.get(class_name, (128, 128, 128))

            if conf > 0.6:
                label = f"{class_name} {conf:.2f}"
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(overlay, (x1, y1 - text_h - rect_padding), (x1 + text_w, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            detected_classes.append(class_name)
            confidences.setdefault(class_name, []).append(conf)

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    predicted_env = predict_environment(detected_classes)
    score, missing_items = calculate_safety_score(detected_classes, predicted_env)

    for class_name in set(detected_classes):
        avg_conf = sum(confidences[class_name]) / len(confidences[class_name])
        log_detection(class_name, avg_conf, score, source_type=source_type, filename=filename, environment=predicted_env)
    
    status_color = (0, 255, 0) if score == 100 else (0, 0, 255)
    status_text = "SAFETY OK" if score == 100 else f"MISSING PPE: {', '.join(missing_items)}"
    
    if source_type == "WEBCAM":
        draw_rounded_rectangle(frame, (10, 10), (290, 43), (50, 50, 50), radius=6, thickness=-1)
        cv2.putText(frame, f"{predicted_env.upper()} | {score}%", (18, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
        cv2.putText(frame, status_text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.33, status_color, 1)

    else:
        draw_rounded_rectangle(frame, (10, 10), (780, 120), (50, 50, 50), radius=20, thickness=-1)
        cv2.putText(frame, f"{predicted_env.upper()} | {score}%", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 2)
        cv2.putText(frame, status_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, status_color, 2)
        
    return frame


# Detects PPE from live webcam stream
def detect_webcam():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # DirectShow backend

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame = process_frame(frame, source_type="WEBCAM", filename="real-time")
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        print()
        print_log("📷 Live webcam detection completed.", "success")


# Fixes FPS of the output video using FFmpeg
def fix_fps_with_ffmpeg(input_path, output_path, target_fps=25):
    subprocess.run([
        'ffmpeg', '-y',
        '-i', input_path,
        '-r', str(target_fps),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        output_path
    ], check=True)


# Detects PPE in an uploaded video file
def detect_video(path):
    cap = cv2.VideoCapture(path)
    original_filename = os.path.splitext(os.path.basename(path))[0]
    result_filename = f"detected_{original_filename}.mp4"
    os.makedirs("static/video_results", exist_ok=True)
    result_path = os.path.join("static/video_results", result_filename)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0 or np.isnan(fps):
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    try:
        frame_counter = 0
        dropped_frames = 0

        while True:
            success, frame = cap.read()
            if not success:
                dropped_frames += 1
                print_log("🛑 Video ended. Stopping detection.", "info")
                break

            frame_counter += 1
            frame = cv2.resize(frame, (width, height))

            frame = process_frame(frame, source_type="VIDEO", filename=result_filename)
            out.write(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        print_log(f"📊 Total processed frames: {frame_counter}", "info")
        print_log(f"⚠️  Dropped frames: {dropped_frames}", "warn")

    finally:
        cap.release()
        out.release()
        print()
        print_log("🎥 Video detection completed.", "success")
        print_log("Finalizing and correcting FPS... Please wait.", "info")
        print()

        try:
            fixed_path = result_path.replace(".mp4", "_fixed.mp4")
            fix_fps_with_ffmpeg(result_path, fixed_path)

            if os.path.exists(fixed_path):
                os.remove(result_path)
                os.rename(fixed_path, result_path)
                time.sleep(0.5)
                result_path = fixed_path
                print_log(f"🎉 FPS fixing completed! Detected video is available at: \033[95m{result_path}\033[0m", "success")
            else:
                print_log("Fixed file not found, FPS correction failed!", "error")
        except Exception:
            pass


# Detects PPE in a static photo file
def detect_photo(image_path):
    results = model.predict(image_path, conf=0.3)
    img = cv2.imread(image_path)

    os.makedirs("static/photo_results", exist_ok=True)
    result_filename = f"detected_{os.path.basename(image_path)}"
    result_path = os.path.join("static/photo_results", result_filename)

    detected_classes = []
    confidences = {}
    class_counts = {}

    overlay = img.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = classNames[cls_id]
        conf = float(box.conf[0])

        detected_classes.append(class_name)
        confidences.setdefault(class_name, []).append(conf)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = classColors.get(class_name, (255, 0, 0))
        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 15)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(overlay, (x1, y1 - text_h - 15), (x1 + text_w, y1), color, -1)
        cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    predicted_env = predict_environment(detected_classes)
    score, missing_items = calculate_safety_score(detected_classes, predicted_env)

    status_color = (0, 255, 0) if score == 100 else (0, 0, 255)
    status_text = "SAFETY OK" if score == 100 else f"MISSING PPE: {', '.join(missing_items)}"

    draw_rounded_rectangle(img, (10, 10), (780, 120), (50, 50, 50), radius=20, thickness=-1)
    cv2.putText(img, f"{predicted_env.upper()} | {score}%", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 2)
    cv2.putText(img, status_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, status_color, 2)

    cv2.imwrite(result_path, img)

    os.makedirs("static/photo", exist_ok=True)
    original_copy_path = os.path.join("static/photo", os.path.basename(image_path))
    cv2.imwrite(original_copy_path, cv2.imread(image_path))

    for class_name in set(detected_classes):
        avg_conf = sum(confidences[class_name]) / len(confidences[class_name])
        log_detection(class_name, avg_conf, score, source_type="PHOTO", filename=result_filename, environment=predicted_env)
    
    print()
    print_log(f"🖼️  Photo detection completed! Detected photo is available at: \033[95m{result_path}\033[0m", "success")
    
    return class_counts, result_path, score, predicted_env


# ====== 📝 LOGGING & STATISTICS ======

# Saves detection results to log file
def log_detection(class_name, confidence, score=None, source_type="VIDEO", filename="N/A", environment="N/A"):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(LOG_PATH, "a") as f:
        if score is not None:
            f.write(f"{now} [{source_type}] [{environment}] {filename} {class_name} {confidence:.2f} {score}%\n")
        else:
            f.write(f"{now} [{source_type}] [{environment}] {filename} {class_name} {confidence:.2f}\n")


# Analyzes logs to calculate stats and risk level
def analyze_safety_stats(lines, source_name):
    class_list = [line.split()[5] for line in lines if f"[{source_name}]" in line]
    environment_classes = defaultdict(list)

    for line in lines:
        if f"[{source_name}]" in line:
            parts = line.strip().split()
            if len(parts) >= 7:
                env = parts[3].strip("[]")
                cls = parts[5]
                environment_classes[env].append(cls)

    total = len(class_list)
    class_counts = Counter(class_list)
    most = class_counts.most_common(1)[0] if class_counts else ("N/A", 0)
    least = class_counts.most_common()[-1] if class_counts else ("N/A", 0)

    high_risk_envs = 0
    medium_risk_envs = 0
    low_risk_envs = 0

    for env, detected_classes in environment_classes.items():
        required_ppe = REQUIRED_PPE_SETS.get(env.capitalize(), set())
        detected_set = set(detected_classes)

        if not required_ppe:
            continue

        matched = detected_set & required_ppe
        missing = required_ppe - matched

        if len(missing) > (len(required_ppe) / 2):
            high_risk_envs += 1
        elif 0 < len(missing) <= (len(required_ppe) / 2):
            medium_risk_envs += 1
        else:
            low_risk_envs += 1

    if high_risk_envs > 0:
        risk_level = "high"
        risk_message = "🔴 High Risk: Many critical PPEs missing!"
    elif medium_risk_envs > 0:
        risk_level = "medium"
        risk_message = "🟡 Medium Risk: Some PPEs missing."
    else:
        risk_level = "low"
        risk_message = "🟢 All required PPEs are properly detected."

    return {
        "total": total,
        "most": most,
        "least": least,
        "avg_score": 0,
        "risk_level": risk_level,
        "risk_message": risk_message
    }


# ====== 🎨 VISUALIZATION & UTILITY ======

# Draws a rounded box for environment and score text
def draw_rounded_rectangle(img, top_left, bottom_right, color, radius=20, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)

    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


# Converts class colors from BGR to HEX
def get_class_colors():
    return {
        name: '#%02x%02x%02x' % (r, g, b)
        for name, (b, g, r) in classColors.items()
    }


# Prints colored log messages to terminal
def print_log(message, level="info"):
    colors = {
        "info": "\033[1;94m[INFO]\033[0m",
        "success": "\033[1;92m[✓]\033[0m",
        "error": "\033[1;91m[ERROR]\033[0m",
        "warn": "\033[1;93m[WARN]\033[0m"
    }
    print(f"{colors.get(level, '[INFO]')} {message}", flush=True)


# Color source tags in terminal logs
def colorize_source(source):
    colors = {
        "ALL": "\033[94m",
        "VIDEO": "\033[93m",
        "WEBCAM": "\033[96m",
        "PHOTO": "\033[91m"
    }
    color = colors.get(source.upper(), "\033[0m")
    return f"{color}'{source}'\033[0m"