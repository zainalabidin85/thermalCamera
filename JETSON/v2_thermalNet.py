#!/usr/bin/env python3
import cv2
import sys
import argparse
import requests
import threading
import queue
import time
from flask import Flask, jsonify
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, videoOutput, Log

# --- Animal normal temperature library (°C) ---
# --- these temperature ranges are estimation only
ANIMAL_TEMP_RANGES = {
    "person": (36.5, 37.5),
    "dog": (38.3, 39.2),
    "cat": (38.1, 39.2),
    "cow": (38.0, 39.3),
    "chicken": (40.6, 41.7),
    "sheep": (38.3, 39.9),
    "horse": (37.5, 38.5),
    "goat": (38.5, 39.7)
}

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Calibrate thermal camera using object detection and Jetson.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=detectNet.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("--output", type=str, default="display://0", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="Detection model to use")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="Overlay types: box,labels,conf")
parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--temp_url", type=str, default="http://raspberrypi.local:8080/ground_truth_temp", help="URL to fetch real-world temp")
parser.add_argument("--post_url", type=str, default="http://raspberrypi.local:8080/calibrate", help="Flask server calibration endpoint")
parser.add_argument("--input", type=str, default="http://10.42.0.1:8080/video_feed", help="Video input source")

args = parser.parse_args()
cap_source = args.input

# --- Load detection model ---
net = detectNet(args.network, sys.argv, args.threshold)
output = videoOutput(args.output, argv=sys.argv)

# --- Start Flask status server ---
app = Flask(__name__)
video_ready = False

@app.route('/status')
def status():
    return jsonify({"status": "active"})

@app.route('/video_ready')
def video_ready_endpoint():
    return jsonify({"ok": video_ready})

threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8081), daemon=True).start()

# --- Attempt to open video stream ---
cap = cv2.VideoCapture(cap_source)
video_ready = cap.isOpened()

if not video_ready:
    print(f"Warning: Failed to open video stream: {cap_source}")
    while True:
        pass

print("Starting object detection and temperature inference...")

# --- Main detection loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = cudaFromNumpy(rgba)

    detections = net.Detect(img, overlay=args.overlay)
    print(f"Detected {len(detections)} objects.")

    for detection in detections:
        label_name = net.GetClassDesc(detection.ClassID).lower()
        if label_name not in ANIMAL_TEMP_RANGES:
            print(f"Skipped: '{label_name}' not in temp library.")
            continue

        center_x = int((detection.Left + detection.Right) / 2)
        center_y = int((detection.Top + detection.Bottom) / 2)
        if not (0 <= center_x < gray.shape[1] and 0 <= center_y < gray.shape[0]):
            print("Detection center out of bounds.")
            continue

        intensity = int(gray[center_y, center_x])
        if intensity == 0:
            print("Skipped: pixel intensity is 0.")
            continue

        try:
            temp_data = requests.get(args.temp_url, timeout=1).json()
            true_temp = temp_data["temp"]
        except Exception as e:
            print("Failed to fetch ground truth temp:", e)
            continue

        scale = true_temp / intensity
        est_temp = round(scale * intensity, 1)

        normal_min, normal_max = ANIMAL_TEMP_RANGES[label_name]
        feverish = est_temp > (normal_max + 1.0)

        print(f"[{label_name}] @ ({center_x},{center_y}) | I={intensity} | Real={true_temp:.1f}°C | Est={est_temp:.1f}°C | {'FEVERISH' if feverish else 'Normal'}")

        payload = {
            "x": center_x,
            "y": center_y,
            "temp": est_temp,
            "intensity": intensity,
            "animal": label_name,
            "fever": feverish
        }

        try:
            r = requests.post(args.post_url, json=payload, timeout=1)
            if r.ok:
                print("✔️ Sent to Flask.")
            else:
                print(f"❌ Flask error: {r.status_code}")
        except Exception as e:
            print("POST error:", e)

    output.Render(img)
    output.SetStatus(f"Jetson Inference | FPS: {net.GetNetworkFPS():.1f}")
    net.PrintProfilerTimes()

    if not output.IsStreaming():
        break
