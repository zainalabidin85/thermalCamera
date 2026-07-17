# Jetson Object Detection and Temperature Calibration

This module uses `jetson-inference` to:
- Pull the video feed from the thermalcam Pi's `/video_feed`.
- Detect animals.
- Query the Pi's `/pixel_temp` endpoint for each detection's center point to get an
  estimated temperature (computed server-side from the true raw sensor frame).
- Flag detections above the species' normal range as feverish.
- Serve the latest results on its own `/detections` endpoint (port 8081) for anything
  that wants to poll the Jetson directly.

## Run with:
```
python3 v2_thermalNet.py --input http://<pi_ip>:5000/video_feed --pixel_temp_url http://<pi_ip>:5000/pixel_temp
```

Defaults assume the Pi is reachable at `thermalcam.local:5000` (see `ThermalCam Pi/thermal_overlay.py`).
