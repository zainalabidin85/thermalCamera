#!/usr/bin/env python3
"""feverDetection_jetson.py - Jetson entry point for the fever-detection engine.

There's no Jetson-specific detection code here: RPI/feverDetection.py
already has zero Raspberry-Pi-specific hardware calls (no local camera, no
GPIO/I2C — it only talks to thermalCam-Pi and the ESP32 over HTTP), and
Ultralytics automatically uses CUDA when it's available. So the exact same
script runs unmodified on Jetson. This file just launches it from a
Jetson-side entry point (e.g. its own systemd unit under JETSON/), so
there's one shared model and one shared codebase instead of a duplicate
Jetson-specific detection stack to keep in sync.

The one thing that differs per device is YOLO_WEIGHTS: point it at a
TensorRT .engine export of your trained model on Jetson for real-time
speed, vs. a plain .pt file on the Pi. Same training pipeline, same class
labels, one model — just export it differently per device:

    # one-time, using the trained .pt (from either device):
    from ultralytics import YOLO
    YOLO("feverDetection.pt").export(format="engine")  # -> feverDetection.engine

Run with:
    YOLO_WEIGHTS=feverDetection.engine THERMALCAM_HOST=<thermalcam-pi> \\
        python3 feverDetection_jetson.py
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RPI"))

if __name__ == "__main__":
    runpy.run_module("feverDetection", run_name="__main__")
