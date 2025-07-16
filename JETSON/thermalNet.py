#!/usr/bin/env python3
import cv2
import sys
import argparse
import requests
import threading
from flask import Flask, jsonify
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, videoOutput, Log

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Calibrate thermal camera using object detection and Jetson.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=detectNet.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("--output", type=str, default="display://0", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="Detection model to use")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="Overlay types: box,labels,conf")
parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--known_temp", type=float, default=30.0, help="Known temperature of the detected object (°C)")
parser.add_argument("--post_url", type=str, default="http://raspberrypi.local:8080/calibrate", help="Flask server calibration endpoint")
parser.add_argument("--input", type=str, default="http://10.42.0.1:8080/video_feed", help="Video input source")

args = parser.parse_args()
cap_source = args.input

# --- Load detection model ---
net = detectNet(args.network, sys.argv, args.threshold)

#  Custom modelNet
# net = detectNet(model="ssd-mobilenet.onnx", labels="labels.txt",
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes",
#                 threshold=args.threshold)
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

# Launch Flask server in background
threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8081), daemon=True).start()


# --- Attempt to open video stream ---
cap = cv2.VideoCapture(cap_source)
video_ready = cap.isOpened()

if not video_ready:
    print(f"Warning: Failed to open video stream: {cap_source}")
    print("Flask server will remain active for /status.")
    while True:
        # Keep script alive
        pass

print("Starting object detection and calibration...")

# --- Main detection loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        continue

    # Prepare image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = cudaFromNumpy(rgba)

    # Run detection
    detections = net.Detect(img, overlay=args.overlay)
    print(f"Detected {len(detections)} objects.")

    for detection in detections:
        center_x = int((detection.Left + detection.Right) / 2)
        center_y = int((detection.Top + detection.Bottom) / 2)
        
        # Retrieve the intensity of the pixel
        if 0 <= center_x < gray.shape[1] and 0 <= center_y < gray.shape[0]:
            intensity = int(gray[center_y, center_x])
            # Avoid dividing by zeros pixel intensity
            if intensity == 0:
                print("Skipping invalid intensity = 0")
                continue
                
            # Formula scale here
            scale = args.known_temp / intensity
            print(f"Calibrated @ ({center_x}, {center_y}) | I={intensity} | T={args.known_temp} | Scale={scale:.4f}")
            
            # package the variables as payload
            payload = {
                "x": center_x,
                "y": center_y,
                "temp": round(args.known_temp, 1),
                "intensity": intensity
            }
            
            # send payload to Flask
            try:
                r = requests.post(args.post_url, json=payload, timeout=1)
                if r.ok:
                    print("Sent calibration to Flask.")
                else:
                    print(f"Error from Flask: {r.status_code}")
            except requests.RequestException as e:
                print(f"Failed to send calibration: {e}")
        else:
            print("Detection center is out of frame bounds.")
    
    # showing object detection image
    output.Render(img)
    output.SetStatus(f"Jetson Calibration | FPS: {net.GetNetworkFPS():.1f}")
    net.PrintProfilerTimes()

    if not output.IsStreaming():
        break
