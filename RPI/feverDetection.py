#!/usr/bin/env python3
"""feverDetection.py - fever-detection inference engine (Jetson or Pi5).

This device has no thermal camera or MLX90614 attached — those live on a
separate device, thermalCam-Pi (its own project/repo), which streams video
and answers per-point temperature queries over HTTP. Pipeline, kept
deliberately in three stages:

  1) Video: pulled from thermalCam-Pi's /video_feed (always 640x480,
     already color-mapped) via cv2.VideoCapture(url).
  2) Detection stage (YOLO): outputs species/class labels only (person,
     dog, cat, ...) — no temperature or fever logic here.
  3) Fever stage (fever_estimator.FeverEstimator): for each detection, asks
     thermalCam-Pi's /pixel_temp for the estimated temperature at that
     point, then decides per-species whether it's feverish.

A fourth helper, pointer_mapper.py, turns a feverish detection's pixel
coordinate into a correctly-scaled ESP32 /move command.
"""

from flask import Flask, Response, send_from_directory, jsonify, request
import cv2
import numpy as np
import threading
import time
import requests
import socket
import os
import atexit
import importlib

from fever_estimator import FeverEstimator
from pointer_mapper import PointerMapper
from thermalcam_client import ThermalCamClient

app = Flask(__name__, static_folder="static")

# ---------------- Config (env overridable) ----------------
DETECT_ENABLE = os.getenv("DETECT_ENABLE", "1") != "0"
DETECT_IMGSZ = int(os.getenv("DETECT_IMGSZ", "320"))
DETECT_CONF = float(os.getenv("DETECT_CONF", "0.35"))
DETECT_IOU = float(os.getenv("DETECT_IOU", "0.5"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo11n.pt")
CLASS_FILTER_ENV = os.getenv("DETECT_CLASSES", "").strip()
CLASS_FILTER = set([c for c in CLASS_FILTER_ENV.split(",") if c]) or None
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

FEVER_MOVE_COOLDOWN = float(os.getenv("FEVER_MOVE_COOLDOWN", "2.0"))
FEVER_MARGIN_C = float(os.getenv("FEVER_MARGIN_C", "1.0"))
FEVER_QUERY_INTERVAL = float(os.getenv("FEVER_QUERY_INTERVAL", "0.3"))

# thermalCam-Pi: the device with the camera + MLX90614 attached
THERMALCAM_HOST = os.getenv("THERMALCAM_HOST", "thermalcam.local")
THERMALCAM_PORT = int(os.getenv("THERMALCAM_PORT", "5000"))
THERMALCAM_BASE = os.getenv("THERMALCAM_BASE", f"http://{THERMALCAM_HOST}:{THERMALCAM_PORT}")
THERMALCAM_COLOR_MAP = os.getenv("THERMALCAM_COLOR_MAP", "inferno")
THERMALCAM_STATUS_POLL_INTERVAL = float(os.getenv("THERMALCAM_STATUS_POLL_INTERVAL", "2.0"))

# ESP32 pointer: bound by mDNS hostname (esp32/thermalPointer/src/main.cpp
# advertises "thermalpointer.local"), not by IP/MAC scanning — this is the
# same pattern as THERMALCAM_HOST above. Whatever ESP32 board runs that
# firmware answers to this name, so swapping the physical board (which
# changes its MAC) doesn't break discovery like the old ARP+vendor-prefix
# scan did.
ESP32_HOST = os.getenv("ESP32_HOST", "thermalpointer.local")
ESP32_BASE = os.getenv("ESP32_BASE", f"http://{ESP32_HOST}")

# ---------------- Shared state ----------------
latest_frame: np.ndarray | None = None      # annotated display frame
det_input_frame: np.ndarray | None = None   # frame handed to YOLO
frame_size = (640, 480)                     # actual (w,h) of latest captured frame

det_lock = threading.RLock()
det_results = []   # species-only: [{"x1","y1","x2","y2","cx","cy","cls","conf"}]
det_fps = 0.0
detector_ready = False

eval_lock = threading.Lock()
latest_evaluated = []   # det_results + {"temp","fever"}, set once per camera-loop frame

model_lock = threading.RLock()
current_weights = YOLO_WEIGHTS   # currently active weights path
active_predictor = {"fn": None, "backend": None}   # swappable predictor closure
model_classes = []               # class names the current model can output
active_class_filter = CLASS_FILTER   # None = no filter (all classes shown)

thermalcam_cache_lock = threading.Lock()
thermalcam_temp_cache = {}     # last /temperature_data response
thermalcam_status_cache = {}   # last /system_status response

# ESP32 info — fixed at ESP32_BASE (mDNS hostname), resolved fresh by the
# OS on every request rather than scanned/cached. See ESP32_BASE above.
esp32_base = ESP32_BASE
esp32_status = f"{esp32_base}/status"
esp32_move = f"{esp32_base}/move"
esp32_config_url = f"{esp32_base}/config"
esp32_servo_url = f"{esp32_base}/servo"

# ---------------- Initialize helpers ----------------
thermalcam = ThermalCamClient(THERMALCAM_BASE)

fever = FeverEstimator(
    client=thermalcam,
    fever_margin=FEVER_MARGIN_C,
    query_interval=FEVER_QUERY_INTERVAL,
)

pointer = PointerMapper(move_cooldown=FEVER_MOVE_COOLDOWN)

# ---------------- Helpers ----------------
def validate_config():
    issues = []
    if DETECT_ENABLE and not os.path.exists(YOLO_WEIGHTS):
        issues.append(f"YOLO weights file not found: {YOLO_WEIGHTS}")
    if issues:
        print("Configuration warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration validation passed")


def list_available_models():
    """Weights files the UI is allowed to switch to: the configured default
    plus any .pt/.engine file dropped in MODELS_DIR."""
    found = set()
    if os.path.exists(YOLO_WEIGHTS):
        found.add(os.path.abspath(YOLO_WEIGHTS))
    if os.path.isdir(MODELS_DIR):
        for fn in os.listdir(MODELS_DIR):
            if fn.endswith((".pt", ".engine")):
                found.add(os.path.abspath(os.path.join(MODELS_DIR, fn)))
    return sorted(found)


def thermalcam_status_loop():
    """Periodically cache thermalCam-Pi's temperature/system status for the
    HUD and /status.json, decoupled from the per-detection /pixel_temp
    queries fever_estimator makes."""
    global thermalcam_temp_cache, thermalcam_status_cache
    while True:
        temp_data = thermalcam.get_temperature_data()
        status_data = thermalcam.get_system_status()
        with thermalcam_cache_lock:
            if temp_data is not None:
                thermalcam_temp_cache = temp_data
            if status_data is not None:
                thermalcam_status_cache = status_data
        time.sleep(THERMALCAM_STATUS_POLL_INTERVAL)


# ---------------- Detection stage (species only) ----------------
def _try_import_yolo_detect():
    try:
        return importlib.import_module("yolo_detect")
    except Exception:
        return None


def _names_to_list(names_map):
    if isinstance(names_map, dict):
        return sorted(set(str(n) for n in names_map.values()))
    if isinstance(names_map, (list, tuple, set)):
        return sorted(set(str(n) for n in names_map))
    return []


def _ultra_predictor(weights):
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
    except Exception as e:
        err = f"Ultralytics load failed: {e}"
        print(f"[DET] {err}")
        return None, None, [], err

    task = getattr(model, "task", None)
    if task != "detect":
        err = f"unsupported model task '{task}' — need a detection (-det) checkpoint, not classification/segmentation/pose/obb"
        print(f"[DET] {weights}: {err}")
        return None, None, [], err

    class_names = _names_to_list(getattr(model, "names", {}))

    def _predict(frame_bgr):
        r = model.predict(source=frame_bgr, imgsz=DETECT_IMGSZ,
                           conf=DETECT_CONF, iou=DETECT_IOU, verbose=False)[0]
        out = []
        names = getattr(r, "names", {})
        if len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                cls_nm = names.get(cls_id, str(cls_id))
                out.append((x1, y1, x2, y2, cls_nm, conf, frame_bgr.shape[1], frame_bgr.shape[0]))
        return out

    return model, _predict, class_names, None


def _load_predictor(weights):
    """Build a predict(frame) closure for `weights`. Tries the optional
    yolo_detect.py hook first (e.g. custom Jetson/TensorRT loader), then
    falls back to plain Ultralytics. Returns (predict_fn, backend_name,
    class_names, err) — predict_fn is None and err is set on failure
    (e.g. bad path, or a non-detection checkpoint like -cls/-seg/-pose)."""
    yd = _try_import_yolo_detect()
    if yd:
        try:
            model = yd.init(weights)

            def _predict(frame):
                try:
                    return yd.predict(model, frame, conf=DETECT_CONF, iou=DETECT_IOU, imgsz=DETECT_IMGSZ)
                except TypeError:
                    return yd.predict(model, frame)

            class_names = _names_to_list(getattr(model, "names", None) or getattr(model, "class_names", None))
            return _predict, "yolo_detect", class_names, None
        except Exception as e:
            print(f"[DET] yolo_detect init failed for {weights}: {e}")

    _, predict, class_names, err = _ultra_predictor(weights)
    if predict:
        return predict, "ultralytics", class_names, None
    return None, None, [], err


def _apply_loaded_model(predict_fn, backend, class_names, weights):
    """Install a freshly-loaded model as active. Resets the class filter if
    it no longer matches the new model's classes. Caller holds no lock."""
    global current_weights, active_class_filter
    with model_lock:
        active_predictor["fn"] = predict_fn
        active_predictor["backend"] = backend
        current_weights = weights
        model_classes[:] = class_names
        if active_class_filter and not active_class_filter.issubset(set(model_classes)):
            active_class_filter = None


def detection_loop():
    """Fill det_results with species-only detections + det_fps."""
    global det_results, det_fps, detector_ready
    if not DETECT_ENABLE:
        print("[DET] disabled")
        return

    predict_fn, backend, class_names, err = _load_predictor(current_weights)
    if predict_fn is None:
        print(f"[DET] no detector available: {err or 'ultralytics missing or bad weights path'}")
        return

    _apply_loaded_model(predict_fn, backend, class_names, current_weights)
    print(f"[DET] using {backend}: {current_weights} ({len(class_names)} classes)")
    detector_ready = True

    while True:
        if det_input_frame is None:
            time.sleep(0.02)
            continue
        frame_for_det = det_input_frame.copy()

        t0 = time.time()
        with model_lock:
            predictor = active_predictor["fn"]
            class_filter = active_class_filter
        preds = predictor(frame_for_det)

        with det_lock:
            det_results.clear()
            if preds:
                H, W = frame_for_det.shape[:2]
                for p in preds:
                    if len(p) == 8:
                        x1, y1, x2, y2, cls, conf, sw, sh = p
                        sx, sy = W / float(sw), H / float(sh)
                        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                    else:
                        x1, y1, x2, y2, cls, conf = p[:6]
                    if class_filter and (str(cls) not in class_filter):
                        continue
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    det_results.append({
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "cx": cx, "cy": cy, "cls": str(cls), "conf": float(conf),
                    })

        dt = max(1e-3, time.time() - t0)
        det_fps = 0.9 * det_fps + 0.1 * (1.0 / dt)


# ---------------- Capture (remote) + fever overlay ----------------
def capture_thermal_feed():
    global latest_frame, det_input_frame, frame_size, latest_evaluated

    video_url = thermalcam.video_feed_url(color_map=THERMALCAM_COLOR_MAP)
    print(f"[CAM] Connecting to thermalCam-Pi: {video_url}")
    cap = cv2.VideoCapture(video_url)

    consecutive_errors = 0
    max_consecutive_errors = 10

    while True:
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if not ok or frame is None:
                consecutive_errors += 1
                print(f"[CAM] Read failed from thermalCam-Pi ({consecutive_errors})")
                if consecutive_errors >= max_consecutive_errors:
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(video_url)
                    consecutive_errors = 0
                time.sleep(0.1)
                continue

            consecutive_errors = 0
            H, W = frame.shape[:2]
            frame_size = (W, H)
            det_input_frame = frame  # already BGR, thermalCam-Pi handles the color map

            colored = frame.copy()

            cv2.drawMarker(colored, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)

            # ---- stage 3: fever secondary pass over species-only detections ----
            with det_lock:
                species_dets = det_results[:]
            evaluated = fever.evaluate(species_dets, W, H)
            with eval_lock:
                latest_evaluated = evaluated

            for d in evaluated:
                x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                cx, cy = d["cx"], d["cy"]
                label = f"{d['cls']} {d['conf']:.2f}"
                color = (0, 255, 0)
                if d["temp"] is not None:
                    label += f" | {d['temp']}C"
                if d["fever"]:
                    label += " | FEVER"
                    color = (0, 0, 255)
                    pointer.queue_move(cx, cy, W, H, temp_c=d["temp"], species=d["cls"])
                cv2.rectangle(colored, (x1, y1), (x2, y2), color, 2)
                cv2.putText(colored, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            latest_frame = colored

        except Exception as e:
            print(f"Camera loop error: {e}")
            time.sleep(1)
            continue

        time.sleep(0.005)


# ---------------- ESP32 pointer sender ----------------
def _send_http(url, payload, retries=2):
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


def send_to_esp32_loop():
    while True:
        payload = pointer.pop()
        if payload is None:
            time.sleep(0.05)
            continue

        if esp32_move:
            if not _send_http(esp32_move, payload):
                print("[ESP32] HTTP send failed:", payload)
                pointer.requeue_front(payload)
        else:
            pointer.requeue_front(payload)
            time.sleep(0.25)


# ---------------- Utilities & Routes ----------------
def get_wlan_ip():
    try:
        iface = "wlan0"
        import fcntl
        import struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(), 0x8915, struct.pack('256s', iface.encode('utf-8'))
        )[20:24])
    except Exception:
        return "Unavailable"


@app.route("/")
def index():
    return send_from_directory("static", "feverDetection.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = latest_frame if latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/thermalcam_status")
def thermalcam_status():
    """Cached view of thermalCam-Pi's own temperature/system status."""
    with thermalcam_cache_lock:
        return jsonify({
            "base_url": THERMALCAM_BASE,
            "temperature": dict(thermalcam_temp_cache) or None,
            "system": dict(thermalcam_status_cache) or None,
        })


@app.route("/status.json")
def status_json():
    esp_alive = False
    try:
        esp_alive = requests.get(esp32_status, timeout=0.5).ok
    except Exception:
        pass

    with eval_lock:
        evaluated = latest_evaluated[:5]
    best_det = max(evaluated, key=lambda x: x["conf"]) if evaluated else None

    with thermalcam_cache_lock:
        temp_data = dict(thermalcam_temp_cache)
        status_data = dict(thermalcam_status_cache)

    return jsonify({
        "esp32": esp_alive,
        "ip_esp32": esp32_base,
        "ip_server": f"http://{get_wlan_ip()}:8080",
        "ip_thermalcam": THERMALCAM_BASE,
        "thermalcam_connected": bool(status_data.get("mlx_connected") or status_data.get("camera_running")),
        "thermalcam_temp": temp_data or None,
        "det_fps": round(float(det_fps), 1),
        "detector_active": bool(DETECT_ENABLE and detector_ready),
        "best_det": best_det,
        "detections": evaluated,
        "fever": fever.status(),
        "pointer_queue_len": len(pointer),
    })


@app.route("/manual_point", methods=["POST"])
def manual_point():
    """UI click-to-point: {"x":.., "y":..} in the 640x480 canvas the UI renders."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        x, y = int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "err": "x & y required"}), 400
    queued = pointer.queue_move(x, y, pointer.reference_w, pointer.reference_h, force=True)
    return jsonify({"ok": queued})


@app.route("/models")
def list_models():
    with model_lock:
        current = current_weights
        backend = active_predictor["backend"]
    return jsonify({
        "current": current,
        "backend": backend,
        "available": list_available_models(),
        "detector_active": bool(DETECT_ENABLE and detector_ready),
    })


@app.route("/select_model", methods=["POST"])
def select_model():
    """Swap the running detector's weights. Restricted to files returned by
    /models (the configured default + MODELS_DIR contents) so this can't be
    used to load an arbitrary path off the filesystem."""
    if not DETECT_ENABLE:
        return jsonify({"ok": False, "err": "detection disabled"}), 503

    data = request.get_json(force=True, silent=True) or {}
    weights = (data.get("weights") or "").strip()
    if weights not in list_available_models():
        return jsonify({"ok": False, "err": "unknown model; must be one of /models' available list"}), 400

    predict_fn, backend, class_names, err = _load_predictor(weights)
    if predict_fn is None:
        return jsonify({"ok": False, "err": err or f"failed to load model: {weights}"}), 400

    _apply_loaded_model(predict_fn, backend, class_names, weights)
    print(f"[DET] switched to {backend}: {weights} ({len(class_names)} classes)")
    return jsonify({"ok": True, "current": weights, "backend": backend, "classes": class_names})


@app.route("/classes")
def list_classes():
    """Classes the current model can output, and which of them are active
    (empty active list means no filter — all classes are shown)."""
    with model_lock:
        return jsonify({
            "available": list(model_classes),
            "active": sorted(active_class_filter) if active_class_filter else [],
        })


@app.route("/select_classes", methods=["POST"])
def select_classes():
    """Set which detected classes are kept. Body: {"classes": [...]},
    empty list clears the filter (all classes shown)."""
    global active_class_filter
    if not DETECT_ENABLE:
        return jsonify({"ok": False, "err": "detection disabled"}), 503

    data = request.get_json(force=True, silent=True) or {}
    classes = data.get("classes")
    if not isinstance(classes, list):
        return jsonify({"ok": False, "err": "classes must be a list (empty = all)"}), 400

    with model_lock:
        valid = set(model_classes)
        requested = set(str(c) for c in classes)
        unknown = requested - valid
        if unknown:
            return jsonify({"ok": False, "err": f"unknown classes: {sorted(unknown)}"}), 400
        active_class_filter = requested or None
        return jsonify({"ok": True, "active": sorted(requested)})


@app.route("/proxy_config", methods=["POST"])
def proxy_config():
    """Forward servo angle-range config to the ESP32."""
    if not esp32_config_url:
        return jsonify({"ok": False, "err": "ESP32 not discovered"}), 503
    try:
        r = requests.post(esp32_config_url, data=request.form, timeout=1.5)
        return (r.text, r.status_code)
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 502


@app.route("/proxy_servo")
def proxy_servo():
    """Fetch current servo angles from the ESP32."""
    if not esp32_servo_url:
        return jsonify({"x": None, "y": None})
    try:
        r = requests.get(esp32_servo_url, timeout=1.0)
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    except Exception:
        return jsonify({"x": None, "y": None})


# ---------------- Cleanup ----------------
def cleanup():
    print("Cleaning up resources...")
    cv2.destroyAllWindows()


atexit.register(cleanup)

# ---------------- Main ----------------
if __name__ == "__main__":
    validate_config()
    print("[feverDetection] Initializing...")
    print(f"[feverDetection] thermalCam-Pi: {THERMALCAM_BASE}")

    threads = [
        threading.Thread(target=capture_thermal_feed, daemon=True),
        threading.Thread(target=detection_loop, daemon=True),
        threading.Thread(target=send_to_esp32_loop, daemon=True),
        threading.Thread(target=thermalcam_status_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    print("[feverDetection] Running")
    try:
        app.run(host="0.0.0.0", port=8080, threaded=True)
    except KeyboardInterrupt:
        print("\n[feverDetection] Shutting down...")
    finally:
        cleanup()
