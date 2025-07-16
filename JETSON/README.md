# Jetson Object Detection and Temperature Calibration

This module uses `jetson-inference` to:
- Detect animals.
- Extract pixel intensities.
- Calibrate estimated temperature.
- POST results to Raspberry Pi for visualization.

## Run with:
```
python3 v2_thermalNet.py --input http://<pi_ip>:8080/video_feed
```
