#!/usr/bin/env python3
import cv2
import sys
import argparse
import requests
import threading
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
parser = argparse.ArgumentParser(description="Detect animals and estimate temperature using the thermalcam Pi's API.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=detectNet.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("--output", type=str, default="display://0", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="Detection model to use")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="Overlay types: box,labels,conf")
parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--pixel_temp_url", type=str, default="http://thermalcam.local:5000/pixel_temp",
    help="thermalcam Pi endpoint that returns an estimated temp at a normalized x,y point")
parser.add_argument("--input", type=str, default="http://thermalcam.local:5000/video_feed?map=inferno",
    help="Video input source (thermalcam Pi's /video_feed)")

args = parser.parse_args()
cap_source = args.input

# --- Load detection model ---
net = detectNet(args.network, sys.argv, args.threshold)
output = videoOutput(args.output, argv=sys.argv)

# --- Start Flask status server ---
app = Flask(__name__)
video_ready = False
latest_detections = []
detections_lock = threading.Lock()

@app.route('/status')
def status():
    return jsonify({"status": "active"})

@app.route('/video_ready')
def video_ready_endpoint():
    return jsonify({"ok": video_ready})

@app.route('/detections')
def detections_endpoint():
    """Latest fever-check results, for anything that wants to poll the Jetson directly
    (there's no /calibrate receiver on the Pi side, so results live here instead)."""
    with detections_lock:
        return jsonify({"detections": list(latest_detections)})

threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8081), daemon=True).start()

# --- Attempt to open video stream ---
cap = cv2.VideoCapture(cap_source)
video_ready = cap.isOpened()

if not video_ready:
    print(f"Warning: Failed to open video stream: {cap_source}")
    while True:
        time.sleep(1)

print("Starting object detection with thermalcam Pi API calibration...")

# --- Main detection loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        continue

    frame_h, frame_w = frame.shape[:2]
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = cudaFromNumpy(rgba)

    detections = net.Detect(img, overlay=args.overlay)
    print(f"Detected {len(detections)} objects.")
    frame_results = []

    for detection in detections:
        label_name = net.GetClassDesc(detection.ClassID).lower()
        if label_name not in ANIMAL_TEMP_RANGES:
            print(f"Skipped: '{label_name}' not in temp library.")
            continue

        center_x = int((detection.Left + detection.Right) / 2)
        center_y = int((detection.Top + detection.Bottom) / 2)
        if not (0 <= center_x < frame_w and 0 <= center_y < frame_h):
            print("Detection center out of bounds.")
            continue

        # Ask the Pi directly for the temp at this point -- it computes this from
        # the true raw sensor frame, so it's more accurate than reconstructing
        # intensity from the (colorized, JPEG-compressed) video frame ourselves.
        x_norm = center_x / max(frame_w - 1, 1)
        y_norm = center_y / max(frame_h - 1, 1)

        try:
            resp = requests.get(args.pixel_temp_url, params={"x": x_norm, "y": y_norm}, timeout=1)
            est_temp = resp.json().get("temp")
        except Exception as e:
            print("Failed to fetch pixel temp from thermalcam Pi:", e)
            continue

        if est_temp is None:
            print(f"Skipped: Pi has no valid temp estimate right now (sensor error?).")
            continue

        normal_min, normal_max = ANIMAL_TEMP_RANGES[label_name]
        feverish = est_temp > (normal_max + 1.0)

        print(f"[{label_name}] @ ({center_x},{center_y}) | Est={est_temp:.1f}°C | {'FEVERISH' if feverish else 'Normal'}")

        frame_results.append({
            "x": center_x,
            "y": center_y,
            "temp": est_temp,
            "animal": label_name,
            "fever": feverish
        })

    with detections_lock:
        latest_detections = frame_results

    output.Render(img)
    output.SetStatus(f"Jetson Inference | FPS: {net.GetNetworkFPS():.1f}")
    net.PrintProfilerTimes()

    if not output.IsStreaming():
        break
