#!/usr/bin/env python3
# v6_thermalCam.py — Pi 5, ESP32 pointer, MLX90614 as ground truth

from flask import Flask, Response, send_from_directory, jsonify, request
import cv2
import numpy as np
import threading
import time
import requests
import socket
import os
import atexit
from collections import deque

# Import the OOP MLX90614 reader
from mlx90614_reader import create_mlx_reader

# ---------------- Optional: user detector module ----------------
def _try_import_yolo_detect():
    try:
        import importlib
        return importlib.import_module("yolo_detect")
    except Exception:
        return None

from device_scanner import scan_devices

app = Flask(__name__, static_folder="static")

# ---------------- Config (env overridable) ----------------
BUFFER_SIZE = int(os.getenv("PTR_BUFFER_SIZE", "60"))
SEND_INTERVAL = float(os.getenv("PTR_SEND_INTERVAL", "0.25"))#000000#000000
DETECT_ENABLE = os.getenv("DETECT_ENABLE", "1") != "0"
DETECT_IMGSZ = int(os.getenv("DETECT_IMGSZ", "320"))
DETECT_CONF = float(os.getenv("DETECT_CONF", "0.35"))
DETECT_IOU = float(os.getenv("DETECT_IOU", "0.5"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo11n.pt")
CLASS_FILTER_ENV = os.getenv("DETECT_CLASSES","").strip()
CLASS_FILTER = set([c for c in CLASS_FILTER_ENV.split(",") if c]) or None

# --- Pointer trigger queue (only on fever or manual click) ---
FEVER_MOVE_COOLDOWN = float(os.getenv("FEVER_MOVE_COOLDOWN", "2.0"))
cmd_queue = deque() # queue of payloads to send to ESP32
_last_fever_move_ts = 0.0 # rate-limit fever-triggered moves

# Fever thresholds (°C)
FEVER_THRESH_PERSON = float(os.getenv("FEVER_THRESH_PERSON", "37.5"))
FEVER_THRESH_ANIMAL = float(os.getenv("FEVER_THRESH_ANIMAL", "39.0"))

# MLX configuration
MLX_OFFSET_C = float(os.getenv("MLX90614_OFFSET_C", "0.0"))
MLX_WINDOW_SIZE = int(os.getenv("MLX_WINDOW_SIZE", "5"))
MLX_POLL_INTERVAL = float(os.getenv("MLX_POLL_INTERVAL", "0.5"))
MLX_I2C_BUS = int(os.getenv("MLX_I2C_BUS", "1"))

# ---------------- Shared state ----------------
latest_frame: np.ndarray | None = None # colorized frame for display
latest_gray: np.ndarray | None = None # raw gray (uint8)
det_input_frame: np.ndarray | None = None # BGR built from gray for YOLO

# Calibration info (scale maps gray->°C)
calibrated_point = None # {"scale": float}

coord_buffer = deque(maxlen=BUFFER_SIZE)
lock = threading.Lock()

# ESP32 info
esp32_base = ""
esp32_status = ""
esp32_move = ""
esp32_config = ""
esp32_locked = False

# Detection results (all detections)
det_lock = threading.RLock()  # Changed to RLock for better thread safety
det_results = [] # list of dicts: {"x1","y1","x2","y2","cx","cy","cls","conf"}
det_fps = 0.0
detector_ready = False

# MLX anchor + auto-cal (DEFAULTS: center + auto-cal enabled using object temp)
mlx_anchor_mode = os.getenv("MLX_ANCHOR_MODE", "center") # center|detection|custom
mlx_anchor_custom = [None, None] # x,y for custom
mlx_offset_c_runtime = 0.0 # small runtime bias
auto_cal_enable = os.getenv("MLX_AUTO_CAL_ENABLE", "1") != "0" # <-- default ON
auto_cal_alpha = float(os.getenv("MLX_AUTO_CAL_ALPHA", "0.05"))
auto_cal_use = os.getenv("MLX_AUTO_CAL_USE", "object") # object|ambient

# ---------------- Initialize MLX Reader ----------------
try:
    mlx_reader = create_mlx_reader(
        window_size=MLX_WINDOW_SIZE,
        poll_interval=MLX_POLL_INTERVAL,
        i2c_bus=MLX_I2C_BUS
    )
    print("[MLX90614] Reader initialized successfully")
except Exception as e:
    print(f"[MLX90614] Failed to initialize: {e}")
    # Create a dummy reader that will fail gracefully
    mlx_reader = None

# ---------------- Helpers ----------------
def get_mlx_anchor_point(shape_hw):
    """Return (ax, ay) where MLX is assumed to point."""
    H, W = shape_hw[:2]
    if mlx_anchor_mode == "detection":
        with det_lock:
            ds = det_results[:]
            if ds:
                best = max(ds, key=lambda d: d["conf"])
                return best["cx"], best["cy"]
        return W//2, H//2
    elif mlx_anchor_mode == "custom":
        x, y = mlx_anchor_custom
        if x is not None and y is not None:
            return int(x), int(y)
        return W//2, H//2
    else:
        return W//2, H//2 # center

def auto_calibration_step(gray_img):
    """EMA adjust 'scale' so gray(anchor) * scale ~= MLX temp."""
    global calibrated_point
    if not auto_cal_enable or gray_img is None or mlx_reader is None:
        return
    
    H, W = gray_img.shape[:2]
    ax, ay = get_mlx_anchor_point((H, W, 1))
    if not (0 <= ax < W and 0 <= ay < H):
        return
    
    g = float(gray_img[ay, ax])
    if g <= 1.0:
        return
    
    amb, obj, ok = mlx_reader.get_latest(fallback_read=False)
    chosen = obj if auto_cal_use != "ambient" else amb
    if not ok or chosen is None:
        return
    
    chosen = float(chosen) + float(mlx_offset_c_runtime)
    if calibrated_point is None:
        calibrated_point = {"scale": 0.5}
    
    old = float(calibrated_point.get("scale", 0.5))
    target = chosen / g
    new = (1.0 - auto_cal_alpha) * old + auto_cal_alpha * target
    calibrated_point["scale"] = new

def resolve_device_ips():
    """Auto-find ESP32 on LAN (optional)."""
    global esp32_base, esp32_status, esp32_move, esp32_config, esp32_locked
    if esp32_locked:
        return
    try:
        for d in scan_devices():
            ip = d['ip']; hostname = d['hostname'].lower(); vendor = d['vendor']
            if 'esp32' in hostname or 'Espressif Inc.' in vendor:
                esp32_base = f"http://{ip}"
                esp32_status = f"{esp32_base}/status"
                esp32_move = f"{esp32_base}/move"
                esp32_config = f"{esp32_base}/config"
                esp32_locked = True
                print(f"[ESP32] Resolved {esp32_base}")
    except Exception as e:
        print(f"[ESP32] scan error: {e}")

def find_working_camera(max_devices=5):
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return i
    return None

def validate_config():
    """Validate critical configuration parameters"""
    issues = []
    
    if DETECT_ENABLE and not os.path.exists(YOLO_WEIGHTS):
        issues.append(f"YOLO weights file not found: {YOLO_WEIGHTS}")
    
    if FEVER_THRESH_PERSON < 35 or FEVER_THRESH_PERSON > 42:
        issues.append(f"Unusual fever threshold for person: {FEVER_THRESH_PERSON}")
    
    if FEVER_THRESH_ANIMAL < 35 or FEVER_THRESH_ANIMAL > 45:
        issues.append(f"Unusual fever threshold for animal: {FEVER_THRESH_ANIMAL}")
    
    if auto_cal_alpha <= 0 or auto_cal_alpha > 1:
        issues.append(f"Auto-cal alpha should be between 0 and 1: {auto_cal_alpha}")
    
    if mlx_reader is None or not mlx_reader.backend:
        issues.append("MLX90614 not available - check connections and dependencies")
    
    if issues:
        print("Configuration warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration validation passed")

# ---------------- Detection loop ----------------
def _ultra_predictor(weights):
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
    except Exception as e:
        print(f"[DET] Ultralytics load failed: {e}")
        return None, None

    def _predict(frame_bgr):
        r = model.predict(source=frame_bgr, imgsz=DETECT_IMGSZ,
                         conf=DETECT_CONF, iou=DETECT_IOU, verbose=False)[0]
        out = []
        names = getattr(r, "names", {})
        if len(r.boxes):
            for b in r.boxes:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                cls_nm = names.get(cls_id, str(cls_id))
                out.append((x1,y1,x2,y2,cls_nm,conf, frame_bgr.shape[1], frame_bgr.shape[0]))
        return out

    return model, _predict

def detection_loop():
    """Fill det_results (list of all detections) + det_fps."""
    global det_results, det_fps, detector_ready
    if not DETECT_ENABLE:
        print("[DET] disabled")
        return

    # Prefer user module if present
    yd = _try_import_yolo_detect()
    predictor = None
    if yd:
        try:
            model = yd.init(os.getenv("YOLO_WEIGHTS", None))
            def _predict(frame): # must return list of (x1,y1,x2,y2,cls,conf, sw, sh)
                try:
                    return yd.predict(model, frame, conf=DETECT_CONF, iou=DETECT_IOU, imgsz=DETECT_IMGSZ)
                except TypeError:
                    return yd.predict(model, frame)
            predictor = _predict
            print("[DET] using yolo_detect.py")
        except Exception as e:
            print(f"[DET] yolo_detect init failed: {e}")

    if predictor is None:
        model, predictor = _ultra_predictor(YOLO_WEIGHTS)
        if predictor: 
            print(f"[DET] using Ultralytics: {YOLO_WEIGHTS}")
        else:
            print("[DET] no detector available (ultralytics missing or bad weights path)")
            return

    detector_ready = True

    while True:
        if det_input_frame is None:
            time.sleep(0.02); continue
        frame_for_det = det_input_frame.copy()

        t0 = time.time()
        preds = predictor(frame_for_det)

        with det_lock:
            det_results.clear()
            if preds:
                H, W = frame_for_det.shape[:2]
                for p in preds:
                    if len(p) == 8:
                        x1,y1,x2,y2,cls,conf, sw, sh = p
                        sx, sy = W/float(sw), H/float(sh)
                        x1,y1,x2,y2 = x1*sx, y1*sy, x2*sx, y2*sy
                    else:
                        x1,y1,x2,y2,cls,conf = p[:6]
                    if CLASS_FILTER and (str(cls) not in CLASS_FILTER):
                        continue
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    det_results.append({
                        "x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                        "cx":cx,"cy":cy,"cls":str(cls),"conf":float(conf)
                    })

        dt = max(1e-3, time.time()-t0)
        det_fps = 0.9*det_fps + 0.1*(1.0/dt)

# ---------------- Capture & overlay ----------------
def _fever_threshold_for(cls_name: str) -> float:
    return FEVER_THRESH_PERSON if (cls_name or "").lower() == "person" else FEVER_THRESH_ANIMAL

def capture_thermal_feed():
    global latest_frame, latest_gray, det_input_frame, calibrated_point

    cam_index = find_working_camera()
    if cam_index is None:
        print("No camera found.")
        latest_frame = np.zeros((480,640,3), dtype=np.uint8)
        return

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    camera_retries = 0
    max_retries = 5

    while True:
        try:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, reinitializing...")
                camera_retries += 1
                if camera_retries >= max_retries:
                    print("Max camera retries reached, exiting camera loop")
                    break
                
                cap.release()
                time.sleep(1)
                cam_index = find_working_camera()
                if cam_index is not None:
                    cap = cv2.VideoCapture(cam_index)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                continue
            
            camera_retries = 0  # Reset retry counter on successful read

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            latest_gray = gray

            # Dedicated detector input (BGR from raw gray)
            det_input_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Display colormap
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            # HUD
            fps = 1.0 / max(1e-6, (time.time() - t0))
            cv2.putText(colored, f"FPS: {fps:.2f} | YOLO FPS: {det_fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # MLX temperature display
            if mlx_reader:
                amb, obj, ok_mlx = mlx_reader.get_latest(fallback_read=False)
                if ok_mlx and amb is not None:
                    mlx_amb_disp = float(amb) + float(mlx_offset_c_runtime)
                    cv2.putText(colored, f"Ambient: {mlx_amb_disp:.2f} C",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    
                    # Show calibration status
                    if calibrated_point and calibrated_point.get("scale", 0) > 0:
                        cv2.putText(colored, f"Calibrated: YES",
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Center crosshair
            H, W = gray.shape[:2]
            center = (W//2, H//2)
            cv2.drawMarker(colored, center, (255,255,255), cv2.MARKER_CROSS, 24, 2)

            # Auto-cal toward MLX (default ON)
            auto_calibration_step(gray)

            # Draw detections; show °C + FEVER if scale>0
            scale = calibrated_point.get("scale", 0.0) if calibrated_point else 0.0
            with det_lock:
                ds = det_results[:]
                for d in ds:
                    x1,y1,x2,y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                    cx, cy = d["cx"], d["cy"]
                    cls_nm, conf = d["cls"], d["conf"]
                    color = (0,255,0)
                    label = f"{cls_nm} {conf:.2f}"
                    if scale > 0 and 0 <= cx < W and 0 <= cy < H:
                        est_temp = round(float(gray[cy, cx]) * scale, 1)
                        thr = _fever_threshold_for(cls_nm)
                        fever = est_temp >= thr
                        label += f" | Temp: {est_temp}C"
                        if fever:
                            label += " | FEVER"
                            color = (0,0,255)
                        global _last_fever_move_ts
                        now = time.time()
                        if now - _last_fever_move_ts >= FEVER_MOVE_COOLDOWN:
                            queue_move(cx, cy, est_temp)
                            _last_fever_move_ts = now
                    cv2.rectangle(colored, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(colored, label, (x1, max(20, y1-10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            latest_frame = colored

        except Exception as e:
            print(f"Camera loop error: {e}")
            time.sleep(1)
            continue

        time.sleep(0.005)
    
    if cap.isOpened():
        cap.release()

# ---------------- ESP32 pointer ----------------
def _send_http(url, payload, retries=2):
    """Send HTTP request with retries and timeout"""
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=1.0)
            if r.ok:
                return True
        except Exception as e:
            if attempt == retries:
                print(f"[ESP32] Final attempt failed: {e}")
            time.sleep(0.1)
    return False

def queue_move(x: int, y: int, temp_c: float | None = None):
    """Queue a one-shot move command for the ESP32 with rate limiting"""
    global _last_fever_move_ts
    
    now = time.time()
    if now - _last_fever_move_ts < FEVER_MOVE_COOLDOWN:
        return
    
    payload = {"x": int(x), "y": int(y)}
    if temp_c is not None:
        payload["temp"] = float(temp_c)
    
    with lock:
        # Limit queue size to prevent memory issues
        if len(cmd_queue) < 100:
            cmd_queue.append(payload)
            _last_fever_move_ts = now

def send_to_esp32_loop():
    global esp32_move
    # Wait up to ~10s for auto-discovery
    for _ in range(40):
        if esp32_move.startswith("http"):
            break
        resolve_device_ips()
        time.sleep(0.25)

    while True:
        payload = None
        with lock:
            if cmd_queue:
                payload = cmd_queue.popleft()
        if payload is None:
            time.sleep(0.05)
            continue

        if esp32_move:
            if not _send_http(esp32_move, payload):
                print("[ESP32] HTTP send failed:", payload)
                # If send failed, put it back in queue for retry
                with lock:
                    if len(cmd_queue) < 100:
                        cmd_queue.appendleft(payload)
        else:
            # If ESP not discovered yet, keep it in the queue's front
            with lock:
                if len(cmd_queue) < 100:
                    cmd_queue.appendleft(payload)
            time.sleep(0.25)

# ---------------- Health Monitoring ----------------
def health_monitor():
    """Monitor system health and log issues"""
    health_check_interval = 30  # seconds
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        time.sleep(health_check_interval)
        
        issues = []
        
        # Check camera feed
        if latest_frame is None:
            issues.append("No frames from camera")
        
        # Check MLX90614
        if mlx_reader and not mlx_reader.ok:
            issues.append("MLX90614 not responding")
        elif not mlx_reader:
            issues.append("MLX90614 not available")
        
        # Check detection performance
        if DETECT_ENABLE and det_fps < 1.0:
            issues.append(f"Low detection FPS: {det_fps:.1f}")
        
        # Check ESP32 connection
        esp_alive = False
        if esp32_status:
            try:
                esp_alive = requests.get(esp32_status, timeout=1.0).ok
            except:
                pass
        if not esp_alive and esp32_status:
            issues.append("ESP32 not responding")
        
        if issues:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print("Health: CRITICAL - Multiple issues detected:")
                for issue in issues:
                    print(f"  - {issue}")
        else:
            consecutive_failures = 0
            print("Health: All systems normal")

# ---------------- Utilities & Routes ----------------
def get_wlan_ip():
    try:
        iface = "wlan0"
        import fcntl, struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(), 0x8915, struct.pack('256s', iface.encode('utf-8'))
        )[20:24])
    except Exception:
        return "Unavailable"

@app.route("/")
def index():
    return send_from_directory("static", "v3_index.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = latest_frame if latest_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/ground_truth_temp")
def ground_truth_temp():
    mode = request.args.get("mode", "object").lower()
    
    if not mlx_reader:
        return jsonify({
            "ok": False,
            "error": "MLX90614 not available",
            "ts": time.time()
        }), 503
    
    amb, obj, ok = mlx_reader.get_latest(fallback_read=True)
    chosen = obj if mode != "ambient" else amb
    if chosen is not None:
        chosen = float(chosen) + float(mlx_offset_c_runtime)
    
    return jsonify({
        "ok": bool(ok and chosen is not None),
        "unit": "C",
        "temp": round(chosen, 3) if chosen is not None else None,
        "ambient": round(amb, 3) if amb is not None else None,
        "object": round(obj, 3) if obj is not None else None,
        "offset_runtime": mlx_offset_c_runtime,
        "ts": time.time(),
    }), (200 if ok and chosen is not None else 503)

@app.route("/mlx_stats")
def mlx_stats():
    """Get MLX90614 statistics and health."""
    if not mlx_reader:
        return jsonify({
            "ok": False,
            "error": "MLX90614 not available"
        }), 503
    
    stats = mlx_reader.get_stats()
    return jsonify({
        "ok": True,
        "stats": stats
    })

@app.route("/status.json")
def status_json():
    resolve_device_ips()
    esp_alive = False
    try:
        if esp32_status:
            esp_alive = requests.get(esp32_status, timeout=0.5).ok
    except Exception:
        pass
    scale_val = calibrated_point.get("scale", -1) if calibrated_point else -1
    
    # Get MLX data
    amb, obj, ok_mlx = None, None, False
    if mlx_reader:
        amb, obj, ok_mlx = mlx_reader.get_latest(fallback_read=False)

    with det_lock:
        ds = det_results[:]
        best_det = None
        det_list = []
        s = calibrated_point.get("scale", 0.0) if calibrated_point else 0.0
        for d in ds[:5]:
            item = {"cls": d["cls"], "conf": float(d["conf"]), "temp": None, "fever": None}
            if latest_gray is not None and s > 0:
                H, W = latest_gray.shape[:2]
                cx, cy = d["cx"], d["cy"]
                if 0 <= cx < W and 0 <= cy < H:
                    t = float(latest_gray[cy, cx]) * s
                    item["temp"] = round(t, 2)
                    item["fever"] = (t >= _fever_threshold_for(d["cls"]))
            det_list.append(item)
        if det_list:
            best_det = max(det_list, key=lambda x: x["conf"])

    return jsonify({
        "esp32": esp_alive,
        "ip_esp32": esp32_base,
        "ip_server": f"http://{get_wlan_ip()}:8080",
        "scale": round(scale_val, 4) if scale_val > 0 else "-",
        "ref_temp": round(obj, 2) if obj is not None else None,
        "det_fps": round(float(det_fps), 1),
        "detector_active": bool(DETECT_ENABLE and detector_ready),
        "best_det": best_det,
        "detections": det_list,
        "mlx_anchor": {"mode": mlx_anchor_mode, "custom": mlx_anchor_custom},
        "auto_cal": {"enable": auto_cal_enable, "alpha": auto_cal_alpha, "use": auto_cal_use},
        "mlx_available": mlx_reader is not None and mlx_reader.backend is not None,
    })

@app.route("/calibrate", methods=["POST"])
def receive_calibration():
    """Set scale directly: POST {"temp":C, "intensity":gray}"""
    global calibrated_point
    data = request.get_json(force=True, silent=True) or {}
    inten = float(data.get("intensity", 0.0))
    tempc = float(data.get("temp", 0.0))
    if inten <= 0:
        return jsonify({"ok": False, "err": "intensity must be > 0"}), 400
    scale = tempc / inten
    calibrated_point = {"scale": scale, "x": data.get("x", 0), "y": data.get("y", 0)}
    return jsonify({"ok": True, "scale": scale})

@app.route("/mlx_anchor", methods=["GET","POST"])
def mlx_anchor():
    """GET -> current; POST {"mode":"center|detection|custom","x":..,"y":..}"""
    global mlx_anchor_mode, mlx_anchor_custom
    if request.method == "GET":
        return jsonify({"mode": mlx_anchor_mode, "custom": mlx_anchor_custom})
    data = request.get_json(force=True, silent=True) or {}
    mode = str(data.get("mode", "center")).lower()
    if mode not in ("center","detection","custom"):
        return jsonify({"ok": False, "err": "mode must be center|detection|custom"}), 400
    mlx_anchor_mode = mode
    if mode == "custom":
        if "x" not in data or "y" not in data:
            return jsonify({"ok": False, "err": "x & y required for custom"}), 400
        mlx_anchor_custom = [int(data["x"]), int(data["y"])]
    return jsonify({"ok": True, "mode": mlx_anchor_mode, "custom": mlx_anchor_custom})

@app.route("/auto_calibrate_mlx", methods=["GET","POST"])
def auto_calibrate_mlx():
    """GET -> state; POST {"enable":bool,"alpha":0.05,"use":"object|ambient"}"""
    global auto_cal_enable, auto_cal_alpha, auto_cal_use
    if request.method == "GET":
        return jsonify({"enable": auto_cal_enable, "alpha": auto_cal_alpha, "use": auto_cal_use})
    data = request.get_json(force=True, silent=True) or {}
    if "enable" in data: auto_cal_enable = bool(data["enable"])
    if "alpha" in data:
        try: auto_cal_alpha = float(data["alpha"])
        except: return jsonify({"ok": False, "err": "alpha must be number"}), 400
    if "use" in data:
        v = str(data["use"]).lower()
        if v not in ("object","ambient"):
            return jsonify({"ok": False, "err": "use must be object|ambient"}), 400
        auto_cal_use = v
    return jsonify({"ok": True, "enable": auto_cal_enable, "alpha": auto_cal_alpha, "use": auto_cal_use})

@app.route("/set_mlx_offset", methods=["GET","POST"])
def set_mlx_offset():
    """Set small runtime bias (°C) added to MLX readings: POST {"offset_c":-0.5}"""
    global mlx_offset_c_runtime
    if request.method == "GET":
        return jsonify({"offset_c": mlx_offset_c_runtime})
    data = request.get_json(force=True, silent=True) or {}
    try:
        mlx_offset_c_runtime = float(data.get("offset_c", 0.0))
    except:
        return jsonify({"ok": False, "err": "offset_c must be number"}), 400
    return jsonify({"ok": True, "offset_c": mlx_offset_c_runtime})

# ---------------- Cleanup ----------------
def cleanup():
    """Cleanup resources on exit"""
    print("Cleaning up resources...")
    if mlx_reader:
        mlx_reader.stop()
    cv2.destroyAllWindows()

# Register cleanup function
atexit.register(cleanup)

# ---------------- Main ----------------
if __name__ == "__main__":
    validate_config()
    
    print("[v6] Initializing...")
    
    threads = [
        threading.Thread(target=capture_thermal_feed, daemon=True),
        threading.Thread(target=detection_loop, daemon=True),
        threading.Thread(target=send_to_esp32_loop, daemon=True),
        threading.Thread(target=health_monitor, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    print("[v6] Running: auto-cal ENABLED by default")
    try:
        app.run(host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n[v6] Shutting down...")
    finally:
        cleanup()
