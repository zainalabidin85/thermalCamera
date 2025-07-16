from flask import Flask, Response, send_from_directory, jsonify, request
import cv2
import numpy as np
import threading
import time
import requests
import socket

try:
    import board
    import busio
    from adafruit_mlx90614 import MLX90614
    has_mlx = True
except ImportError:
    print("MLX90614 libraries not found.")
    has_mlx = False

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

latest_mlx_temp = 0.0

def read_mlx_loop():
    global latest_mlx_temp
    if not has_mlx:
        return
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        mlx = MLX90614(i2c)
    except Exception as e:
        print("Failed to initialize MLX90614:", e)
        return
    while True:
        try:
            latest_mlx_temp = mlx.object_temperature
        except Exception as e:
            print("MLX read error:", e)
        time.sleep(1)

@app.route('/ground_truth_temp')
def ground_truth_temp():
    return jsonify({"temp": round(latest_mlx_temp, 2)})

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

@app.route('/status.json')
def status_json():
    resolve_device_ips()
    esp_alive = False
    jet_alive = False
    try:
        esp_alive = requests.get(esp32_status, timeout=0.5).ok
    except: pass
    try:
        jet_alive = requests.get(jetson_endpoint, timeout=0.5).ok
    except: pass
    scale_val = calibrated_point.get("scale", -1) if calibrated_point else -1
    return jsonify({
        "esp32": esp_alive,
        "jetson": jet_alive,
        "ip_esp32": esp32_base,
        "ip_jetson": jetson_ip,
        "ip_server": f"http://{get_wlan_ip()}:8080",
        "scale": round(scale_val, 4) if scale_val > 0 else "-"
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
    threading.Thread(target=read_mlx_loop, daemon=True).start()
    threading.Thread(target=capture_thermal_feed, daemon=True).start()
    threading.Thread(target=send_to_esp32_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)
