from flask import Flask, Response, send_from_directory, jsonify, request
import cv2, numpy as np, threading, time, requests, socket, os
from collections import deque

# --- MLX90614 backends (Adafruit preferred, smbus2 fallback) ---
_HAS_ADAFRUIT = False
_HAS_SMBUS2 = False
try:
    import board, busio
    from adafruit_mlx90614 import MLX90614
    _HAS_ADAFRUIT = True
except Exception:
    pass

try:
    from smbus2 import SMBus
    _HAS_SMBUS2 = True
except Exception:
    pass

from device_scanner import scan_devices

app = Flask(__name__, static_folder="static")

latest_frame = None
calibrated_point = None
coord_buffer = []
buffer_size = 60
send_interval = 2.0
lock = threading.Lock()

esp32_base = ""
esp32_status = ""
esp32_move = ""
esp32_config = ""
esp32_locked = False
jetson_ip = ""
jetson_endpoint = ""

# ---------------- MLX Helper ----------------
MLX_ADDR = int(os.getenv("MLX90614_ADDR", "0x5A"), 16)
MLX_OFFSET_C = float(os.getenv("MLX90614_OFFSET_C", "0.0"))  # small calibration tweak
REG_TA = 0x06
REG_TOBJ1 = 0x07

class MLXReader:
    """
    Thread-safe MLX90614 reader with moving average smoothing.
    Uses Adafruit library if present; otherwise smbus2 direct register reads.
    """
    def __init__(self, window=5, poll_s=0.5):
        self.window = max(1, int(window))
        self.poll_s = poll_s
        self.ok = False
        self.backend = None
        self._amb_hist = deque(maxlen=self.window)
        self._obj_hist = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr = None

        # Lazy init of hardware in start() to avoid import errors at import time
        self._i2c = None  # for Adafruit
        self._mlx = None  # for Adafruit
        self._bus = None  # for smbus2

    def _init_hw(self):
        if _HAS_ADAFRUIT:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._mlx = MLX90614(self._i2c)
            self.backend = "adafruit"
            return
        if _HAS_SMBUS2:
            self._bus = SMBus(1)
            self.backend = "smbus2"
            return
        self.backend = None

    def _read_once(self):
        if self.backend == "adafruit":
            amb = float(self._mlx.ambient_temperature) + MLX_OFFSET_C
            obj = float(self._mlx.object_temperature) + MLX_OFFSET_C
        elif self.backend == "smbus2":
            # read_word_data returns big-endian for MLX -> swap
            rawa = self._bus.read_word_data(MLX_ADDR, REG_TA)
            rawa = ((rawa & 0xFF) << 8) | (rawa >> 8)
            amb = rawa * 0.02 - 273.15 + MLX_OFFSET_C

            rawo = self._bus.read_word_data(MLX_ADDR, REG_TOBJ1)
            rawo = ((rawo & 0xFF) << 8) | (rawo >> 8)
            obj = rawo * 0.02 - 273.15 + MLX_OFFSET_C
        else:
            raise RuntimeError("No MLX backend available")
        return amb, obj

    def read_once(self):
        """Public: read and update buffers, returning current values."""
        amb, obj = self._read_once()
        with self._lock:
            self._amb_hist.append(amb)
            self._obj_hist.append(obj)
        self.ok = True
        return amb, obj

    def get_latest(self, fallback_read=True):
        with self._lock:
            have = len(self._obj_hist) > 0
        if not have and fallback_read and self.backend:
            try:
                self.read_once()
            except Exception:
                pass
        with self._lock:
            amb = sum(self._amb_hist) / len(self._amb_hist) if self._amb_hist else None
            obj = sum(self._obj_hist) / len(self._obj_hist) if self._obj_hist else None
        return amb, obj, self.ok

    def start(self):
        if self._thr:
            return
        self._init_hw()
        if not self.backend:
            print("MLX90614: no backend available (install adafruit-circuitpython-mlx90614 or smbus2)")
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        if self._thr:
            self._stop.set()
            self._thr.join(timeout=1.0)
            self._thr = None

    def _loop(self):
        # take a couple of warmup reads
        for _ in range(2):
            try: self.read_once()
            except Exception: pass
            time.sleep(self.poll_s)
        while not self._stop.is_set():
            try:
                amb, obj = self._read_once()
                # simple outlier reject: skip if >15°C off rolling average
                with self._lock:
                    avg_obj = (sum(self._obj_hist)/len(self._obj_hist)) if self._obj_hist else obj
                if self._obj_hist and abs(obj - avg_obj) > 15:
                    # skip this spike
                    pass
                else:
                    with self._lock:
                        self._amb_hist.append(amb)
                        self._obj_hist.append(obj)
                self.ok = True
            except Exception as e:
                self.ok = False
                # keep running; next loop may recover
            time.sleep(self.poll_s)

mlx_reader = MLXReader(window=5, poll_s=0.5)

# -------------- existing device discovery --------------
def resolve_device_ips():
    global esp32_base, esp32_status, esp32_move, esp32_config, esp32_locked
    global jetson_ip, jetson_endpoint
    if esp32_locked:
        return
    devices = scan_devices()
    for d in devices:
        ip = d['ip']
        hostname = d['hostname'].lower()
        vendor = d['vendor']
        if 'esp32' in hostname or 'Espressif Inc.' in vendor:
            esp32_base = f"http://{ip}"
            esp32_status = f"{esp32_base}/status"
            esp32_move = f"{esp32_base}/move"
            esp32_config = f"{esp32_base}/config"
            esp32_locked = True
            print(f"ESP32 IP resolved: {esp32_base}")
        elif 'nvidia' in vendor.lower() or 'orin' in hostname.lower():
            jetson_ip = f"http://{ip}:8081"
            jetson_endpoint = f"http://{ip}:8081/status"

def find_working_camera(max_devices=5):
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def capture_thermal_feed():
    global latest_frame, calibrated_point
    cam_index = find_working_camera()
    if cam_index is None:
        print("No camera found.")
        latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        cv2.putText(colored, f"Coords: {maxLoc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(colored, f"FPS: {1/(time.time()-start):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.circle(colored, maxLoc, 10, (0,255,255), 2)

        if calibrated_point:
            scale = calibrated_point.get("scale", 0)
            est_temp = round(scale * maxVal, 1)
            position = (maxLoc[0] + 15, maxLoc[1] - 15)
            cv2.putText(colored, f"Max Temp : {est_temp:.1f}C", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            x, y = calibrated_point.get("x", 0), calibrated_point.get("y", 0)
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                obj_temp = round(float(gray[y, x]) * scale, 1)
                fever = calibrated_point.get("fever", False)
                label = calibrated_point.get("animal", "Object").title()
                cv2.putText(colored, f"Temp: {obj_temp:.1f}C", (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(colored, "FEVERISH" if fever else label, (x+10, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if fever else (0,255,0), 2)
                cv2.circle(colored, (x, y), 8, (0,255,0), 2)

        latest_frame = colored.copy()
        with lock:
            coord_buffer.append(maxLoc)
            if len(coord_buffer) > buffer_size:
                coord_buffer.pop(0)
        time.sleep(0.01)

def send_to_esp32_loop():
    global esp32_move
    while not esp32_move.startswith("http"):
        time.sleep(0.5)
    while True:
        time.sleep(send_interval)
        with lock:
            if len(coord_buffer) < buffer_size:
                continue
            avg_x = sum(p[0] for p in coord_buffer) // buffer_size
            avg_y = sum(p[1] for p in coord_buffer) // buffer_size
        try:
            r = requests.post(esp32_move, json={"x": avg_x, "y": avg_y}, timeout=0.2)
            if r.ok:
                print(f"Sent: ({avg_x}, {avg_y})")
        except:
            print("ESP32 not reachable.")

# ---------------- HTTP routes ----------------
@app.route('/')
def index():
    return send_from_directory("static", "index.html")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = latest_frame if latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ground_truth_temp')
def ground_truth_temp():
    """
    GET /ground_truth_temp?mode=object|ambient
    Returns JSON with smoothed MLX readings.
    """
    mode = request.args.get("mode", "object").lower()
    amb, obj, ok = mlx_reader.get_latest(fallback_read=True)
    chosen = obj if mode != "ambient" else amb
    return jsonify({
        "ok": bool(ok and chosen is not None),
        "unit": "C",
        "temp": round(float(chosen), 3) if (chosen is not None) else None,
        "ambient": round(float(amb), 3) if (amb is not None) else None,
        "object": round(float(obj), 3) if (obj is not None) else None,
        "ts": time.time()
    }), (200 if ok and chosen is not None else 503)

@app.route('/status.json')
def status_json():
    resolve_device_ips()
    esp_alive = False
    jet_alive = False
    try:
        esp_alive = requests.get(esp32_status, timeout=0.5).ok
    except:
        pass
    try:
        jet_alive = requests.get(jetson_endpoint, timeout=0.5).ok
    except:
        pass
    scale_val = calibrated_point.get("scale", -1) if calibrated_point else -1
    # include latest reference temperature for UI/debug
    amb, obj, ok = mlx_reader.get_latest(fallback_read=False)
    return jsonify({
        "esp32": esp_alive,
        "jetson": jet_alive,
        "ip_esp32": esp32_base,
        "ip_jetson": jetson_ip,
        "ip_server": f"http://{get_wlan_ip()}:8080",
        "scale": round(scale_val, 4) if scale_val > 0 else "-",
        "ref_temp": round(obj, 2) if (obj is not None) else None
    })

@app.route('/calibrate', methods=['POST'])
def receive_calibration():
    global calibrated_point
    data = request.json
    if data.get("intensity", 0) == 0:
        return jsonify({"error": "Zero intensity"}), 400
    scale = data["temp"] / data["intensity"]
    data["scale"] = scale
    calibrated_point = data
    print(f"Received: {calibrated_point}")
    return jsonify({"status": "ok"})

def get_wlan_ip():
    try:
        iface = "wlan0"
        import fcntl, struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,
            struct.pack('256s', iface.encode('utf-8'))
        )[20:24])
    except:
        return "Unavailable"

if __name__ == "__main__":
    mlx_reader.start()  # <<< start MLX background smoothing
    threading.Thread(target=capture_thermal_feed, daemon=True).start()
    threading.Thread(target=send_to_esp32_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)
