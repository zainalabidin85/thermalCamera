# Inference Engine — Fever Detection (feverDetection.py)

This device (a Raspberry Pi 5 or Jetson) has **no camera or MLX90614
attached**. Sensor hardware lives on a separate device/project,
**thermalCam-Pi** (a USB camera + MLX90614 IR point sensor, its own repo,
default `http://<host>:5000`), which streams video and answers per-point
temperature queries over HTTP. This device just consumes that API and runs
inference. Three stages, kept deliberately separate:

1. **Video** — pulled from thermalCam-Pi's `/video_feed` (always 640x480,
   already color-mapped) via `cv2.VideoCapture(url)`.
2. **Detection stage** (YOLO) — outputs **species/class labels only**
   (person, dog, cat, ...). No temperature or fever logic happens here.
3. **Fever stage** (`fever_estimator.py`) — for each detection, asks
   thermalCam-Pi's `/pixel_temp?x=<0-1>&y=<0-1>` for the estimated
   temperature at that point (computed server-side from the true raw
   sensor frame, not a compressed copy), then decides per-species whether
   it's feverish using normal temperature ranges. No calibration math
   happens on this device — thermalCam-Pi already did it.

A fourth helper, `pointer_mapper.py`, turns a feverish detection's pixel
coordinate into a correctly-scaled `/move` command for the ESP32 laser
pointer, normalizing into the fixed 640x480 reference frame the ESP32
firmware expects and rate-limiting how often the pointer is redirected.

## Files

- `feverDetection.py` — main Flask app; orchestrates remote video capture,
  detection, fever evaluation, and ESP32 pointer control. Hosts the browser
  UI.
- `fever_estimator.py` — species temperature ranges + fever verdict, driven
  by `thermalcam_client.py` queries (rate-limited to the top few
  highest-confidence detections per cycle, to avoid hammering thermalCam-Pi
  with an HTTP round trip per detection per frame).
- `thermalcam_client.py` — thin HTTP client for thermalCam-Pi's API
  (`/pixel_temp`, `/temperature_data`, `/temp_range`, `/system_status`,
  `/video_feed`).
- `pointer_mapper.py` — pixel-to-reference-frame normalization and
  rate-limited move queue for the ESP32.
- `device_scanner.py` — LAN scan to auto-discover the ESP32 by MAC vendor
  prefix.
- `static/feverDetection.html` — web UI: live video feed, click-to-point,
  live detection table (species/confidence/temp/fever), thermalCam-Pi
  connection status, servo range controls.

## Run with:
```
python3 feverDetection.py
```

Serves on `http://<this-device-ip>:8080`. Key environment variables (all
optional):

| Variable | Default | Purpose |
|---|---|---|
| `THERMALCAM_HOST` / `THERMALCAM_PORT` | `pi4.local` / `5000` | thermalCam-Pi's address |
| `THERMALCAM_BASE` | `http://<host>:<port>` | Overrides host/port with a full base URL |
| `THERMALCAM_COLOR_MAP` | `inferno` | Color map requested from thermalCam-Pi's `/video_feed` |
| `THERMALCAM_STATUS_POLL_INTERVAL` | `2.0` | Seconds between background polls of thermalCam-Pi's status (HUD/UI only) |
| `DETECT_ENABLE` | `1` | Enable/disable the detection thread |
| `YOLO_WEIGHTS` | `yolo11n.pt` | Path to YOLO weights |
| `DETECT_IMGSZ` / `DETECT_CONF` / `DETECT_IOU` | `320` / `0.35` / `0.5` | YOLO inference params |
| `DETECT_CLASSES` | (all) | Comma-separated class allowlist |
| `FEVER_MOVE_COOLDOWN` | `2.0` | Seconds between laser-pointer moves |
| `FEVER_MARGIN_C` | `1.0` | °C above species normal-max before flagging fever |
| `FEVER_QUERY_INTERVAL` | `0.3` | Minimum seconds between `/pixel_temp` query batches to thermalCam-Pi |
